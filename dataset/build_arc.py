from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os
import json
import hashlib
import numpy as np

from pydantic import BaseModel

from dataset import DatasetMetadata

ARC_MAX_GRID_SIZE = 30
ARC_AUGMENT_RETRIES_FACTOR = 5
PUZZLE_ID_SEPARATOR = "|||"


class DataProcessConfig(BaseModel):
    input_file_prefix: str = "data/arc-agi/arc-agi"
    output_dir: str = "data/arc1concept-aug-1000"
    subsets: List[str] = ["training", "evaluation", "concept"]
    test_set_name: str = "evaluation"
    seed: int = 0
    num_aug: int = 1000
    puzzle_identifiers_start: int = 1


@dataclass
class ARCPuzzle:
    id: str
    examples: List[Tuple[np.ndarray, np.ndarray]]


def arc_grid_to_np(grid: List[List[int]]):
    arr = np.array(grid)
    assert arr.ndim == 2
    assert arr.shape[0] <= ARC_MAX_GRID_SIZE and arr.shape[1] <= ARC_MAX_GRID_SIZE
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""

    if tid == 0:
        return arr
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)
    elif tid == 5:
        return np.flipud(arr)
    elif tid == 6:
        return arr.T
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    else:
        return arr


def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])


def np_grid_to_seq_translational_augment(
    inp: np.ndarray, out: np.ndarray, do_translation: bool
):
    if do_translation:
        pad_r = np.random.randint(
            0, ARC_MAX_GRID_SIZE - max(inp.shape[0], out.shape[0]) + 1
        )
        pad_c = np.random.randint(
            0, ARC_MAX_GRID_SIZE - max(inp.shape[1], out.shape[1]) + 1
        )
    else:
        pad_r = pad_c = 0

    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        grid = np.pad(
            grid + 2,
            (
                (pad_r, ARC_MAX_GRID_SIZE - pad_r - nrow),
                (pad_c, ARC_MAX_GRID_SIZE - pad_c - ncol),
            ),
            constant_values=0,
        )

        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARC_MAX_GRID_SIZE:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARC_MAX_GRID_SIZE:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2
    assert grid.dtype == np.uint8
    buffer = [x.to_bytes(1, byteorder="big") for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: dict):
    hashes = []
    for _, example in puzzle.items():
        for input_grid, label_grid in example.examples:
            hashes.append(f"{grid_hash(input_grid)}|{grid_hash(label_grid)}")
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def aug(name: str):
    trans_id = np.random.randint(0, 8)
    mapping = np.concatenate(
        [
            np.arange(0, 1, dtype=np.uint8),
            np.random.permutation(np.arange(1, 10, dtype=np.uint8)),
        ]
    )
    name_with_aug_repr = f"{name}{PUZZLE_ID_SEPARATOR}t{trans_id}{PUZZLE_ID_SEPARATOR}{''.join(str(x) for x in mapping)}"

    def _map_grid(grid: np.ndarray):
        return dihedral_transform(mapping[grid], trans_id)

    return name_with_aug_repr, _map_grid


def inverse_aug(name: str):
    if PUZZLE_ID_SEPARATOR not in name:
        return name, lambda x: x

    trans_id, perm = name.split(PUZZLE_ID_SEPARATOR)[-2:]
    trans_id = int(trans_id[1:])
    inv_perm = np.argsort(list(perm)).astype(np.uint8)

    def _map_grid(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]

    return name.split(PUZZLE_ID_SEPARATOR)[0], _map_grid


def convert_single_arc_puzzle(
    results: dict,
    name: str,
    puzzle: dict,
    aug_count: int,
    dest_mapping: Dict[str, Tuple[str, str]],
):
    dests = set(dest_mapping.values())
    converted = {dest: ARCPuzzle(name, []) for dest in dests}
    for example_type, examples in puzzle.items():
        dest = dest_mapping[example_type]
        converted[dest].examples.extend(
            [
                (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
                for example in examples
            ]
        )

    group = [converted]

    if aug_count > 0:
        hashes = {puzzle_hash(converted)}
        for _trial in range(ARC_AUGMENT_RETRIES_FACTOR * aug_count):
            aug_name, _map_grid = aug(name)
            augmented = {
                dest: ARCPuzzle(
                    aug_name,
                    [
                        (_map_grid(input_grid), _map_grid(label_grid))
                        for (input_grid, label_grid) in puzzle_.examples
                    ],
                )
                for dest, puzzle_ in converted.items()
            }
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)
            if len(group) >= aug_count + 1:
                break
        if len(group) < aug_count + 1:
            print(f"[Puzzle {name}] augmentation not full, only {len(group)}")

    for dest in dests:
        dest_split, dest_set = dest
        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted[dest] for converted in group])


def load_puzzles_arcagi(config: DataProcessConfig):
    train_examples_dest = ("train", "all")
    test_examples_map = {
        config.test_set_name: [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))],
    }

    test_puzzles = {}
    results = {}
    total_puzzles = 0

    for subset_name in config.subsets:
        with open(
            f"{config.input_file_prefix}_{subset_name}_challenges.json", "r"
        ) as f:
            puzzles = json.load(f)

        sols_filename = f"{config.input_file_prefix}_{subset_name}_solutions.json"
        if os.path.isfile(sols_filename):
            with open(sols_filename, "r") as f:
                sols = json.load(f)
                for puzzle_id in puzzles.keys():
                    for idx, sol_grid in enumerate(sols[puzzle_id]):
                        puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            print(f"{subset_name} solutions not found, filling with dummy")
            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        puzzles = list(puzzles.items())
        np.random.shuffle(puzzles)

        for idx, (name, puzzle) in enumerate(puzzles):
            fraction = idx / len(puzzles)
            test_examples_dest = None
            for f, dest in test_examples_map.get(
                subset_name, test_examples_map["_default"]
            ):
                if fraction < f:
                    test_examples_dest = dest
                    break

            assert test_examples_dest is not None

            if test_examples_dest[0] == "test":
                test_puzzles[name] = puzzle

            convert_single_arc_puzzle(
                results,
                name,
                puzzle,
                config.num_aug,
                {"train": train_examples_dest, "test": test_examples_dest},
            )
            total_puzzles += 1

    print(f"Total puzzles: {total_puzzles}")
    return results, test_puzzles


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)
    data, test_puzzles = load_puzzles_arcagi(config)

    num_identifiers = config.puzzle_identifiers_start
    identifier_map: Dict[str, int] = {}

    for _, split in data.items():
        for _, subset in split.items():
            for group in subset:
                for puzzle in group:
                    if puzzle.id not in identifier_map:
                        identifier_map[puzzle.id] = num_identifiers
                        num_identifiers += 1

    print(f"Total puzzle IDs (including <blank>): {num_identifiers}")

    for split_name, split in data.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        enable_translational_augment = split_name == "train"

        total_examples = 0
        total_puzzles = 0
        total_groups = 0

        for subset_name, subset in split.items():
            results = {
                k: []
                for k in [
                    "inputs",
                    "labels",
                    "puzzle_identifiers",
                    "puzzle_indices",
                    "group_indices",
                ]
            }
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)

            example_id = 0
            puzzle_id = 0

            for group in subset:
                for puzzle in group:
                    no_aug_id = np.random.randint(0, len(puzzle.examples))
                    for _idx_ex, (inp, out) in enumerate(puzzle.examples):
                        inp_seq, out_seq = np_grid_to_seq_translational_augment(
                            inp,
                            out,
                            do_translation=enable_translational_augment
                            and _idx_ex != no_aug_id,
                        )
                        results["inputs"].append(inp_seq)
                        results["labels"].append(out_seq)
                        example_id += 1
                        total_examples += 1

                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(identifier_map[puzzle.id])

                    puzzle_id += 1
                    total_puzzles += 1

                results["group_indices"].append(puzzle_id)
                total_groups += 1

            for k, v in results.items():
                if k in {"inputs", "labels"}:
                    v = np.stack(v, 0)
                else:
                    v = np.array(v, dtype=np.int32)
                np.save(
                    os.path.join(
                        config.output_dir, split_name, f"{subset_name}__{k}.npy"
                    ),
                    v,
                )

        metadata = DatasetMetadata(
            seq_len=ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE,
            vocab_size=12,
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            total_puzzles=total_puzzles,
            sets=list(split.keys()),
        )

        with open(
            os.path.join(config.output_dir, split_name, "dataset.json"), "w"
        ) as f:
            json.dump(metadata.model_dump(), f)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        ids_mapping = {v: k for k, v in identifier_map.items()}
        json.dump([ids_mapping.get(i, "<blank>") for i in range(num_identifiers)], f)

    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w") as f:
        json.dump(test_puzzles, f)


def main() -> None:
    config = DataProcessConfig()
    convert_dataset(config)


if __name__ == "__main__":
    main()
