import os
from typing import List, Tuple

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from dataset import DatasetMetadata


cli = ArgParser()


class TestDatasetConfig(BaseModel):
    output_dir: str = "data/player-16x16"
    grid_size: int = 16
    player_size: int = 4
    train_fraction: float = 0.5
    seed: int = 0


def _player_grid(grid_size: int, player_size: int, pos: Tuple[int, int]) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    x, y = pos
    grid[y : y + player_size, x : x + player_size] = 1
    return grid.reshape(-1)


def _move_player(
    x: int,
    y: int,
    action: int,
    grid_size: int,
    player_size: int,
) -> Tuple[int, int]:
    if action == 0:  # up
        y = max(0, y - 1)
    elif action == 1:  # down
        y = min(grid_size - player_size, y + 1)
    elif action == 2:  # left
        x = max(0, x - 1)
    elif action == 3:  # right
        x = min(grid_size - player_size, x + 1)
    return x, y


def _build_examples(config: TestDatasetConfig) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    examples: List[Tuple[np.ndarray, np.ndarray, int]] = []
    max_pos = config.grid_size - config.player_size
    for y in range(max_pos + 1):
        for x in range(max_pos + 1):
            for action in range(4):
                start = _player_grid(config.grid_size, config.player_size, (x, y))
                new_x, new_y = _move_player(x, y, action, config.grid_size, config.player_size)
                end = _player_grid(config.grid_size, config.player_size, (new_x, new_y))
                examples.append((start, end, action))
    return examples


def _save_split(
    split: str,
    data: List[Tuple[np.ndarray, np.ndarray, int]],
    config: TestDatasetConfig,
):
    inputs = np.stack([sample[0] for sample in data]).astype(np.int32)
    labels = np.stack([sample[1] for sample in data]).astype(np.int32)
    actions = np.array([[sample[2]] for sample in data], dtype=np.int32)

    total_examples = inputs.shape[0]
    puzzle_indices = np.arange(total_examples + 1, dtype=np.int32)
    puzzle_identifiers = np.zeros(total_examples, dtype=np.int32)
    group_indices = np.arange(total_examples + 1, dtype=np.int32)

    metadata = DatasetMetadata(
        seq_len=config.grid_size * config.grid_size,
        vocab_size=2,
        pad_id=0,
        ignore_label_id=None,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=total_examples,
        mean_puzzle_examples=1,
        total_puzzles=total_examples,
        sets=["all"],
        num_actions=1,
        action_vocab_size=4,
    )

    split_dir = os.path.join(config.output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    with open(os.path.join(split_dir, "dataset.json"), "w") as f:
        f.write(metadata.model_dump_json())

    np.save(os.path.join(split_dir, "all__inputs.npy"), inputs)
    np.save(os.path.join(split_dir, "all__labels.npy"), labels)
    np.save(os.path.join(split_dir, "all__actions.npy"), actions)
    np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
    np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(split_dir, "all__group_indices.npy"), group_indices)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        f.write("[\"<blank>\"]")


@cli.command(singleton=True)
def build_test_dataset(config: TestDatasetConfig):
    examples = _build_examples(config)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(examples)

    total = len(examples)
    train_count = int(round(total * config.train_fraction))
    train_count = min(max(train_count, 1), total - 1)
    train_split = examples[:train_count]
    test_split = examples[train_count:]

    _save_split("train", train_split, config)
    _save_split("test", test_split, config)

    print(f"train examples: {len(train_split)}")
    print(f"test examples: {len(test_split)}")


if __name__ == "__main__":
    cli()
