import json
import os
from typing import Dict, List

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from dataset import PuzzleDatasetMetadata


cli = ArgParser()


class ArcDatasetConfig(BaseModel):
    source_dir: str = "data/raw-arc3"
    output_dir: str = "data/arc3"
    train_fraction: float = 0.9
    downsample_factor: int = 2
    seed: int = 0


def _init_split_store() -> Dict[str, List]:
    return {
        "inputs": [],
        "labels": [],
        "puzzle_indices": [0],
        "group_indices": [0],
        "puzzle_identifiers": [],
        "puzzle_counts": [],
    }


def _stack_examples(examples: List[np.ndarray], seq_len: int) -> np.ndarray:
    if not examples:
        return np.zeros((0, seq_len), dtype=np.int32)
    return np.concatenate(examples, axis=0).astype(np.int32, copy=False)


def _majority_vote(values: np.ndarray, rng: np.random.Generator) -> int:
    counts = np.bincount(values)
    max_count = counts.max()
    candidates = np.flatnonzero(counts == max_count)
    if candidates.size == 1 or rng is None:
        return int(candidates[0])
    idx = rng.integers(0, candidates.size)
    return int(candidates[idx])


def _downsample_grid(
    grid: np.ndarray, factor: int, rng: np.random.Generator
) -> np.ndarray:
    if factor == 1:
        return grid.copy()
    h, w = grid.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(
            f"Grid size {grid.shape} is not divisible by factor {factor}."
        )
    new_h = h // factor
    new_w = w // factor
    reshaped = grid.reshape(new_h, factor, new_w, factor)
    reshaped = reshaped.transpose(0, 2, 1, 3).reshape(new_h, new_w, factor * factor)
    out = np.empty((new_h, new_w), dtype=grid.dtype)
    for i in range(new_h):
        for j in range(new_w):
            block = reshaped[i, j].reshape(-1)
            out[i, j] = _majority_vote(block, rng)
    return out


def _downsample_frames(
    frames: np.ndarray, factor: int, rng: np.random.Generator
) -> np.ndarray:
    if factor == 1:
        return frames.copy()
    downsampled = np.empty(
        (frames.shape[0], frames.shape[1] // factor, frames.shape[2] // factor),
        dtype=frames.dtype,
    )
    for idx, frame in enumerate(frames):
        downsampled[idx] = _downsample_grid(frame, factor, rng)
    return downsampled


def build_arc_dataset(config: ArcDatasetConfig):
    source_dir = config.source_dir
    files = sorted(
        [
            f
            for f in os.listdir(source_dir)
            if f.endswith(".npz") and os.path.isfile(os.path.join(source_dir, f))
        ]
    )
    if not files:
        raise ValueError(f"No .npz files found under {source_dir}")

    seq_len = None
    vocab_max = 0
    splits = {"train": _init_split_store(), "test": _init_split_store()}
    puzzle_names: List[str] = []
    rng = np.random.default_rng(config.seed)

    for idx, filename in enumerate(files):
        path = os.path.join(source_dir, filename)
        puzzle_name = os.path.splitext(filename)[0]
        puzzle_names.append(puzzle_name)
        npz = np.load(path)
        frames = np.asarray(npz["frames"], dtype=np.int32)
        actions = np.asarray(npz["actions"], dtype=np.int32)

        if frames.ndim != 3 or frames.shape[1] != frames.shape[2]:
            raise ValueError(f"Unexpected frame shape {frames.shape} in {filename}")
        if frames.shape[0] != actions.shape[0]:
            raise ValueError(f"Mismatched frames/actions lengths in {filename}")
        if frames.shape[0] < 2:
            continue

        frames = _downsample_frames(frames, max(1, int(config.downsample_factor)), rng)
        grid_size = frames.shape[1]
        current_seq_len = grid_size * grid_size
        if seq_len is None:
            seq_len = current_seq_len
        elif seq_len != current_seq_len:
            raise ValueError(
                f"Inconsistent seq_len {current_seq_len} encountered in {filename}"
            )

        frames[:, 0, 0] = actions
        vocab_max = max(vocab_max, int(frames.max()))

        flat = frames.reshape(frames.shape[0], -1).astype(np.int32)
        inputs = flat[:-1]
        labels = flat[1:]
        num_pairs = inputs.shape[0]
        if num_pairs == 0:
            continue

        train_fraction = float(np.clip(config.train_fraction, 0.0, 1.0))
        if train_fraction <= 0.0:
            train_count = 0
        elif train_fraction >= 1.0 or num_pairs == 1:
            train_count = num_pairs
        else:
            train_count = int(num_pairs * train_fraction)
            train_count = max(1, train_count)
            train_count = min(num_pairs - 1, train_count)

        puzzle_identifier = idx + 1
        train_inputs = inputs[:train_count]
        train_labels = labels[:train_count]
        test_inputs = inputs[train_count:]
        test_labels = labels[train_count:]

        def add_examples(split_name: str, inp: np.ndarray, lab: np.ndarray):
            if inp.size == 0:
                return
            split = splits[split_name]
            split["inputs"].append(inp)
            split["labels"].append(lab)
            split["puzzle_indices"].append(split["puzzle_indices"][-1] + inp.shape[0])
            split["group_indices"].append(split["group_indices"][-1] + 1)
            split["puzzle_identifiers"].append(puzzle_identifier)
            split["puzzle_counts"].append(inp.shape[0])

        add_examples("train", train_inputs, train_labels)
        add_examples("test", test_inputs, test_labels)

    if seq_len is None:
        raise ValueError("No valid puzzles found.")
    vocab_size = vocab_max + 1
    identifier_names = ["<blank>", *puzzle_names]

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(identifier_names, f)

    for split_name, split_store in splits.items():
        split_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        inputs = _stack_examples(split_store["inputs"], seq_len)
        labels = _stack_examples(split_store["labels"], seq_len)
        puzzle_indices = np.asarray(split_store["puzzle_indices"], dtype=np.int32)
        group_indices = np.asarray(split_store["group_indices"], dtype=np.int32)
        puzzle_identifiers = np.asarray(
            split_store["puzzle_identifiers"], dtype=np.int32
        )

        if split_store["puzzle_counts"]:
            mean_puzzle_examples = float(np.mean(split_store["puzzle_counts"]))
        else:
            mean_puzzle_examples = 0.0

        metadata = PuzzleDatasetMetadata(
            seq_len=seq_len,
            vocab_size=vocab_size,
            pad_id=0,
            ignore_label_id=None,
            blank_identifier_id=0,
            num_puzzle_identifiers=len(identifier_names),
            total_groups=max(len(group_indices) - 1, 0),
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=len(split_store["puzzle_counts"]),
            sets=["all"],
        )

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

        np.save(os.path.join(split_dir, "all__inputs.npy"), inputs)
        np.save(os.path.join(split_dir, "all__labels.npy"), labels)
        np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(split_dir, "all__group_indices.npy"), group_indices)
        np.save(
            os.path.join(split_dir, "all__puzzle_identifiers.npy"),
            puzzle_identifiers,
        )


@cli.command(singleton=True)
def build(config: ArcDatasetConfig):
    build_arc_dataset(config)


if __name__ == "__main__":
    cli()
