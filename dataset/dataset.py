import os
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from argdantic import ArgParser
from pydantic import BaseModel


IGNORE_LABEL_ID = -100


class DatasetMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: List[str]


class DatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int
    rank: int
    num_replicas: int


class Dataset(IterableDataset):
    def __init__(self, config: DatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += (
                current_metadata.mean_puzzle_examples * current_metadata.total_puzzles
            )
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = DatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets,
        )

        assert (
            self.config.global_batch_size % self.config.num_replicas == 0
        ), f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = (
            self.config.global_batch_size // self.config.num_replicas
        )

        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> DatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return DatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None,
        }

        self._data = {}
        for set_name in self.metadata.sets:
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                self._data[set_name_] = {
                    field_name: np.load(
                        os.path.join(
                            dataset_path, self.split, f"{set_name}__{field_name}.npy"
                        ),
                        mmap_mode=mmap_mode,
                    )
                    for field_name, mmap_mode in field_mmap_modes.items()
                }

    def _collate_batch(self, batch):
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        if self.metadata.ignore_label_id is not None:
            batch["labels"][
                batch["labels"] == self.metadata.ignore_label_id
            ] = IGNORE_LABEL_ID

        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
            }
            batch = {
                k: np.pad(
                    v,
                    ((0, pad_size),) + ((0, 0),) * (v.ndim - 1),
                    constant_values=pad_values[k],
                )
                for k, v in batch.items()
            }

        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):
            total_examples = len(dataset["inputs"])

            start_index = 0
            while start_index < total_examples:
                end_index = min(
                    total_examples, start_index + self.config.global_batch_size
                )

                local_start = start_index + self.config.rank * self.local_batch_size
                local_end = min(
                    start_index + (self.config.rank + 1) * self.local_batch_size,
                    end_index,
                )

                puzzle_indices = []
                puzzle_index = (
                    np.searchsorted(
                        dataset["puzzle_indices"], local_start, side="right"
                    )
                    - 1
                )
                for i in range(local_start, local_end):
                    while (
                        puzzle_index + 1 < len(dataset["puzzle_indices"])
                        and i >= dataset["puzzle_indices"][puzzle_index + 1]
                    ):
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)

                batch = self._collate_batch(
                    {
                        "inputs": dataset["inputs"][local_start:local_end],
                        "labels": dataset["labels"][local_start:local_end],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][
                            puzzle_indices
                        ],
                    }
                )

                yield set_name, batch, end_index - start_index

                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():
            self._iters += 1

            rng = np.random.Generator(
                np.random.Philox(seed=self.config.seed + self._iters)
            )

            group_order = np.concatenate(
                [
                    rng.permutation(dataset["group_indices"].size - 1)
                    for _i in range(self.config.epochs_per_iter)
                ]
            )
            start_index = 0

            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                global_effective_batch_size = batch_puzzle_indices.size

                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices = batch_indices[
                    self.config.rank
                    * self.local_batch_size : (self.config.rank + 1)
                    * self.local_batch_size
                ]
                batch_puzzle_indices = batch_puzzle_indices[
                    self.config.rank
                    * self.local_batch_size : (self.config.rank + 1)
                    * self.local_batch_size
                ]
                batch = self._collate_batch(
                    {
                        "inputs": dataset["inputs"][batch_indices],
                        "labels": dataset["labels"][batch_indices],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][
                            batch_puzzle_indices
                        ],
                    }
                )

                yield set_name, batch, global_effective_batch_size

    def __iter__(self):
        worker_info = get_worker_info()
        assert (
            worker_info is None or worker_info.num_workers == 1
        ), "Multithreaded data loading is not currently supported."

        self._lazy_load_dataset()

        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


class GroupDataset(IterableDataset):
    """
    IterableDataset for contrastive training over ARC groups.

    Each iteration yields:
        set_name: str
        batch: Dict[str, torch.Tensor]
            {
                "inputs_1": (B, K, seq_len) int32
                "labels_1": (B, K, seq_len) int32
                "inputs_2": (B, K, seq_len) int32
                "labels_2": (B, K, seq_len) int32
                "group_ids": (B,) int32        # indices into group_indices
            }
        global_group_batch_size: int   # number of groups across all ranks
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        examples_per_view: int = 4,
        num_views: int = 2,
    ):
        super().__init__()
        assert num_views == 2, "GroupDataset currently supports exactly 2 views."
        self.config = config
        self.split = split
        self.examples_per_view = examples_per_view
        self.num_views = num_views

        # --- metadata aggregation (copied from Dataset, unchanged) ---
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers

            mean_puzzle_examples += (
                current_metadata.mean_puzzle_examples * current_metadata.total_puzzles
            )
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = DatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets,
        )

        # global_batch_size is interpreted as "global number of groups per batch"
        assert (
            self.config.global_batch_size % self.config.num_replicas == 0
        ), f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_group_batch_size = (
            self.config.global_batch_size // self.config.num_replicas
        )

        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> DatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return DatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "puzzle_indices": None,
            "group_indices": None,
        }

        self._data = {}
        for set_name in self.metadata.sets:
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                self._data[set_name_] = {
                    field_name: np.load(
                        os.path.join(
                            dataset_path, self.split, f"{set_name}__{field_name}.npy"
                        ),
                        mmap_mode=mmap_mode,
                    )
                    for field_name, mmap_mode in field_mmap_modes.items()
                }

    def _collate_group_batch(
        self,
        inputs_1: np.ndarray,
        labels_1: np.ndarray,
        inputs_2: np.ndarray,
        labels_2: np.ndarray,
        group_ids: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        # Cast + map ignore_label_id -> IGNORE_LABEL_ID
        inputs_1 = inputs_1.astype(np.int32)
        inputs_2 = inputs_2.astype(np.int32)
        labels_1 = labels_1.astype(np.int32)
        labels_2 = labels_2.astype(np.int32)

        if self.metadata.ignore_label_id is not None:
            labels_1[labels_1 == self.metadata.ignore_label_id] = IGNORE_LABEL_ID
            labels_2[labels_2 == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        batch = {
            "inputs_1": torch.from_numpy(inputs_1),
            "labels_1": torch.from_numpy(labels_1),
            "inputs_2": torch.from_numpy(inputs_2),
            "labels_2": torch.from_numpy(labels_2),
            "group_ids": torch.from_numpy(group_ids.astype(np.int32)),
        }
        return batch

    def _sample_examples_for_group(
        self,
        rng: np.random.Generator,
        dataset: Dict[str, np.ndarray],
        group_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample two K-example views from one group."""
        group_indices = dataset["group_indices"]
        puzzle_indices = dataset["puzzle_indices"]

        # puzzles belonging to this group
        p_start = int(group_indices[group_id])
        p_end = int(group_indices[group_id + 1])
        puzzle_ids = np.arange(p_start, p_end, dtype=np.int32)

        # gather all example indices in this group
        ex_ids = []
        for p in puzzle_ids:
            e_start = int(puzzle_indices[p])
            e_end = int(puzzle_indices[p + 1])
            ex_ids.extend(range(e_start, e_end))
        ex_ids = np.asarray(ex_ids, dtype=np.int64)

        assert ex_ids.size > 0, "Group has no examples, which should not happen."

        K = self.examples_per_view
        replace = ex_ids.size < K

        # view 1
        idx1 = rng.choice(ex_ids, size=K, replace=replace)
        # view 2
        idx2 = rng.choice(ex_ids, size=K, replace=replace)

        inputs = dataset["inputs"]
        labels = dataset["labels"]
        inputs_1 = inputs[idx1]  # (K, seq_len)
        labels_1 = labels[idx1]
        inputs_2 = inputs[idx2]
        labels_2 = labels[idx2]
        return inputs_1, labels_1, inputs_2, labels_2

    def _iter_train(self):
        # One "epoch" over groups, repeated epochs_per_iter times, with random shuffling
        for set_name, dataset in self._data.items():
            self._iters += 1
            rng = np.random.Generator(
                np.random.Philox(seed=self.config.seed + self._iters)
            )

            num_groups = dataset["group_indices"].size - 1
            group_order = np.concatenate(
                [
                    rng.permutation(num_groups)
                    for _ in range(self.config.epochs_per_iter)
                ]
            )

            start_index = 0
            global_group_batch_size = self.config.global_batch_size

            while start_index < group_order.size:
                end_index = start_index + global_group_batch_size
                if end_index > group_order.size:
                    break  # not enough groups left for a full global batch

                global_groups = group_order[start_index:end_index]
                start_index = end_index

                # shard across replicas
                local_start = self.config.rank * self.local_group_batch_size
                local_end = (self.config.rank + 1) * self.local_group_batch_size
                local_groups = global_groups[local_start:local_end]

                inputs_1_list: List[np.ndarray] = []
                labels_1_list: List[np.ndarray] = []
                inputs_2_list: List[np.ndarray] = []
                labels_2_list: List[np.ndarray] = []
                group_ids_list: List[int] = []

                for g in local_groups:
                    i1, l1, i2, l2 = self._sample_examples_for_group(
                        rng, dataset, int(g)
                    )
                    inputs_1_list.append(i1)
                    labels_1_list.append(l1)
                    inputs_2_list.append(i2)
                    labels_2_list.append(l2)
                    group_ids_list.append(int(g))

                # Stack into (B, K, seq_len)
                inputs_1 = np.stack(inputs_1_list, axis=0)
                labels_1 = np.stack(labels_1_list, axis=0)
                inputs_2 = np.stack(inputs_2_list, axis=0)
                labels_2 = np.stack(labels_2_list, axis=0)
                group_ids = np.asarray(group_ids_list, dtype=np.int32)

                batch = self._collate_group_batch(
                    inputs_1, labels_1, inputs_2, labels_2, group_ids
                )

                # global_group_batch_size = number of groups across all ranks
                yield set_name, batch, global_group_batch_size

    def _iter_test(self):
        # Deterministic pass over groups; same batching logic but no shuffling
        for set_name, dataset in self._data.items():
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed))
            num_groups = dataset["group_indices"].size - 1
            group_order = np.arange(num_groups, dtype=np.int32)

            start_index = 0
            global_group_batch_size = self.config.global_batch_size

            while start_index < group_order.size:
                end_index = start_index + global_group_batch_size
                if end_index > group_order.size:
                    break

                global_groups = group_order[start_index:end_index]
                start_index = end_index

                local_start = self.config.rank * self.local_group_batch_size
                local_end = (self.config.rank + 1) * self.local_group_batch_size
                local_groups = global_groups[local_start:local_end]

                inputs_1_list: List[np.ndarray] = []
                labels_1_list: List[np.ndarray] = []
                inputs_2_list: List[np.ndarray] = []
                labels_2_list: List[np.ndarray] = []
                group_ids_list: List[int] = []

                for g in local_groups:
                    i1, l1, i2, l2 = self._sample_examples_for_group(
                        rng, dataset, int(g)
                    )
                    inputs_1_list.append(i1)
                    labels_1_list.append(l1)
                    inputs_2_list.append(i2)
                    labels_2_list.append(l2)
                    group_ids_list.append(int(g))

                inputs_1 = np.stack(inputs_1_list, axis=0)
                labels_1 = np.stack(labels_1_list, axis=0)
                inputs_2 = np.stack(inputs_2_list, axis=0)
                labels_2 = np.stack(labels_2_list, axis=0)
                group_ids = np.asarray(group_ids_list, dtype=np.int32)

                batch = self._collate_group_batch(
                    inputs_1, labels_1, inputs_2, labels_2, group_ids
                )

                yield set_name, batch, global_group_batch_size

    def __iter__(self):
        worker_info = get_worker_info()
        assert (
            worker_info is None or worker_info.num_workers == 1
        ), "Multithreaded data loading is not currently supported."

        self._lazy_load_dataset()

        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


def _sample_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    global_batch_size: int,
):
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(
            puzzle_start + np.random.choice(puzzle_size, append_size, replace=False)
        )

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)
