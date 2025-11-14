from typing import Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
from adam_atan2_pytorch import AdamAtan2

from dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from trm.model import Model
from trm.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from trm.ema import EMAHelper
from trm.losses import ACTLossHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    data_paths: List[str]
    global_batch_size: int
    epochs: int
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    project_name: str
    run_name: str
    seed: int
    ema_rate: float
    model: Dict[str, Any]


DEFAULT_CONFIG = TrainConfig(
    data_paths=["data/maze-30x30-hard-1k"],
    global_batch_size=192,
    epochs=50000,
    lr=1e-4,
    lr_min_ratio=1.0,
    lr_warmup_steps=2000,
    weight_decay=1.0,
    beta1=0.9,
    beta2=0.95,
    puzzle_emb_lr=1e-4,
    puzzle_emb_weight_decay=1.0,
    project_name="maze-act",
    run_name="default",
    seed=0,
    ema_rate=0.999,
    model=dict(
        halt_exploration_prob=0.1,
        halt_max_steps=16,
        H_cycles=3,
        L_cycles=4,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        num_heads=8,
        expansion=4,
        puzzle_emb_ndim=512,
        forward_dtype="bfloat16",
        puzzle_emb_len=16,
    ),
)


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: TrainConfig, split: str, **kwargs):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths,
            rank=0,
            num_replicas=1,
            **kwargs,
        ),
        split=split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader, dataset.metadata


def create_model(
    config: TrainConfig,
    train_metadata: PuzzleDatasetMetadata,
):
    model_cfg = dict(
        **config.model,
        batch_size=config.global_batch_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,
    )

    with torch.device(DEVICE):
        model: nn.Module = Model(model_cfg)
        model = ACTLossHead(model)
        model = model.to(DEVICE)
        model = torch.compile(model)

    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),
            lr=1e-4,
            weight_decay=config.puzzle_emb_weight_decay,
            world_size=1,
        ),
        AdamAtan2(
            model.parameters(),
            lr=1e-4,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        ),
    ]
    optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(
            0.0,
            (1 - min_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    )


def init_train_state(
    config: TrainConfig,
    train_metadata: PuzzleDatasetMetadata,
):
    total_steps = int(
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        / config.global_batch_size
    )

    model, optimizers, optimizer_lrs = create_model(config, train_metadata)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )


def compute_lr(base_lr: float, config: TrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )


def train_batch(
    config: TrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    if train_state.carry is None:
        with torch.device(DEVICE):
            train_state.carry = train_state.model.initial_carry(batch)

    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    ((1 / global_batch_size) * loss).backward()

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step
        optim.step()
        optim.zero_grad()

    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        metric_values = metric_values.cpu().numpy()
        reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
        count = max(reduced_metrics["count"], 1)
        reduced_metrics = {
            f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
            for k, v in reduced_metrics.items()
        }
        reduced_metrics["train/lr"] = lr_this_step
        return reduced_metrics


def train_loop(config: TrainConfig):
    torch.random.manual_seed(config.seed)

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=config.epochs,
        global_batch_size=config.global_batch_size,
    )

    train_state = init_train_state(config, train_metadata)

    progress_bar = tqdm.tqdm(total=train_state.total_steps)
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config={
            **config.__dict__,
            "model": config.model,
        },
        settings=wandb.Settings(_disable_stats=True),
    )
    wandb.log(
        {"num_params": sum(x.numel() for x in train_state.model.parameters())},
        step=0,
    )

    ema_helper = EMAHelper(mu=config.ema_rate)
    ema_helper.register(train_state.model)

    train_state.model.train()
    for _, batch, global_batch_size in train_loader:
        metrics = train_batch(
            config,
            train_state,
            batch,
            global_batch_size,
        )

        wandb.log(metrics, step=train_state.step)
        progress_bar.update(train_state.step - progress_bar.n)
        ema_helper.update(train_state.model)

        if train_state.step >= train_state.total_steps:
            break

    wandb.finish()


if __name__ == "__main__":
    train_loop(DEFAULT_CONFIG)
