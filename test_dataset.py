import os
import math
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors


from dataset import PuzzleDataset, PuzzleDatasetConfig

CHARSET = "# SGo"
CHAR_COLORS = {
    "#": "black",  
    " ": "white",  
    "S": "green",  
    "G": "red", 
    "o": "orange",  
}


def build_id2char():
    id2char = ["?"] * (len(CHARSET) + 1)
    id2char[0] = "."
    for i, ch in enumerate(CHARSET, start=1):
        id2char[i] = ch
    return id2char


def show_pair(input_tokens: np.ndarray, output_tokens: np.ndarray, seq_len: int):
    id2char = build_id2char()

    def to_color_grid(tokens):
        chars = np.array([id2char[int(t)] for t in tokens])
        n = int(np.sqrt(seq_len))
        grid = chars.reshape(n, n)
        color_grid = np.zeros((n, n, 3))
        for ch, color in CHAR_COLORS.items():
            mask = grid == ch
            color_grid[mask] = colors.to_rgb(color)
        return color_grid

    c1 = to_color_grid(input_tokens)
    c2 = to_color_grid(output_tokens)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(c1, interpolation="none")
    plt.title("Input Maze")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(c2, interpolation="none")
    plt.title("Output Maze")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def count_total_examples(dataset: PuzzleDataset) -> int:
    """Use the internal _data mapping to count examples."""
    dataset._lazy_load_dataset()
    total = 0
    for set_name, data in dataset._data.items():
        total += data["inputs"].shape[0]
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/maze-30x30-hard-1k",
        help="Directory where preprocess_data wrote the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to inspect",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Global batch size to request from the dataset",
    )
    parser.add_argument(
        "--num_examples_to_print",
        type=int,
        default=3,
        help="How many examples from the first batch to print",
    )
    args = parser.parse_args()

    config = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[args.output_dir],
        global_batch_size=args.batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    dataset = PuzzleDataset(config=config, split=args.split)

    total_examples = count_total_examples(dataset)
    meta = dataset.metadata

    print("=== Puzzle Dataset Info ===")
    print(f"Output dir        : {args.output_dir}")
    print(f"Split             : {args.split}")
    print(f"Total examples    : {total_examples}")
    print(f"Seq len           : {meta.seq_len}")
    print(f"Vocab size        : {meta.vocab_size}")
    print(f"Pad id            : {meta.pad_id}")
    print(f"Ignore label id   : {meta.ignore_label_id}")
    print(f"Blank ident id    : {meta.blank_identifier_id}")
    print(f"Num identifiers   : {meta.num_puzzle_identifiers}")
    print(f"Total groups      : {meta.total_groups}")
    print(f"Mean puzzle exs   : {meta.mean_puzzle_examples:.3f}")
    print(f"Total puzzles     : {meta.total_puzzles}")
    print(f"Sets in metadata  : {meta.sets}")
    print("===========================")

    it = iter(dataset)
    try:
        set_name, batch, effective_bs = next(it)
    except StopIteration:
        return

    inputs = batch["inputs"]
    labels = batch["labels"]

    print(f"\nFirst batch from set '{set_name}':")
    print(f"  Batch inputs shape : {tuple(inputs.shape)}")
    print(f"  Batch labels shape : {tuple(labels.shape)}")
    print(f"  Effective batch sz : {effective_bs}")

    num_to_print = min(args.num_examples_to_print, inputs.shape[0])

    print("\n=== Sample Mazes (input -> target) ===")
    for i in range(num_to_print):
        inp_tokens = inputs[i].numpy()
        out_tokens = labels[i].numpy()
        show_pair(inp_tokens, out_tokens, meta.seq_len)


if __name__ == "__main__":
    main()
