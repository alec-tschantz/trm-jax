import math
from dataclasses import dataclass
from typing import Any, Dict, List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import torch
from torch.utils.data import DataLoader
import tqdm
import wandb

from dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from trm.losses import act_loss
from trm.model import Carry, Model
from trm.utils import EMAHelper
from trm.optim import adam_atan2, sparse_sign_sgd

jax.config.update("jax_enable_x64", True)


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
    grad_clip_norm: float | None
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
    grad_clip_norm=1.0,
    project_name="maze-act",
    run_name="default",
    seed=0,
    ema_rate=0.999,
    model=dict(
        halt_exploration_prob=0.1,
        halt_max_steps=16,
        H_cycles=3,
        L_cycles=4,
        L_layers=2,
        hidden_size=512,
        num_heads=8,
        expansion=4,
        puzzle_emb_ndim=512,
        forward_dtype="float32",
        puzzle_emb_len=16,
    ),
)


@dataclass
class TrainState:
    params: eqx.Module
    static: eqx.Module
    opt_state: optax.OptState
    carry: Carry | None
    step: int
    total_steps: int
    rng: jnp.ndarray


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
        num_workers=0,
        pin_memory=True,
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
    key = jax.random.PRNGKey(config.seed)
    model = Model(model_cfg, key=key)
    params, static = eqx.partition(model, eqx.is_array)

    def build_param_labels(p):
        labels = jax.tree.map(lambda _: 0, p)
        labels = eqx.tree_at(lambda tree: tree.inner.puzzle_emb.weight, labels, 1)
        return labels

    param_labels = build_param_labels(params)

    transforms = {
        0: adam_atan2(
            beta1=config.beta1,
            beta2=config.beta2,
            weight_decay=config.weight_decay,
        ),
        1: sparse_sign_sgd(weight_decay=config.puzzle_emb_weight_decay),
    }

    optimizer = optax.multi_transform(transforms, build_param_labels)
    opt_state = optimizer.init(params)
    return params, static, optimizer, opt_state, param_labels


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
    params, static, optimizer, opt_state, param_labels = create_model(
        config, train_metadata
    )
    rng = jax.random.PRNGKey(config.seed + 1)
    return (
        TrainState(
            params=params,
            static=static,
            opt_state=opt_state,
            carry=None,
            step=0,
            total_steps=total_steps,
            rng=rng,
        ),
        optimizer,
        param_labels,
    )


def compute_lr(base_lr: float, config: TrainConfig, train_state: TrainState):
    lr = cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )
    return jnp.array(lr, dtype=jnp.float32)


def batch_to_jnp(batch: Dict[str, torch.Tensor]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in batch.items()}


def prepare_carry(model: Model, carry: Carry, batch: Dict[str, jnp.ndarray]) -> Carry:
    new_inner_carry = model.inner.reset_carry(carry.halted, carry.inner_carry)
    new_steps = jnp.where(carry.halted, 0, carry.steps)
    halted = carry.halted
    new_current_data = {
        k: jnp.where(
            halted.reshape((-1,) + (1,) * (batch[k].ndim - 1)),
            batch[k],
            carry.current_data[k],
        )
        for k in batch
    }
    return Carry(
        inner_carry=new_inner_carry,
        steps=new_steps,
        halted=halted,
        current_data=new_current_data,
    )


def train_loop(config: TrainConfig):
    torch.random.manual_seed(config.seed)

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=config.epochs,
        global_batch_size=config.global_batch_size,
    )

    train_state, optimizer, param_labels = init_train_state(config, train_metadata)

    progress_bar = tqdm.tqdm(total=train_state.total_steps)
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config={**config.__dict__, "model": config.model},
        settings=wandb.Settings(_disable_stats=True),
    )
    wandb.log(
        {"num_params": sum(x.size for x in jtu.tree_leaves(train_state.params))},
        step=0,
    )

    ema_helper = EMAHelper(mu=config.ema_rate)
    ema_helper.register(eqx.combine(train_state.params, train_state.static))
    static_model = train_state.static
    max_grad_norm = config.grad_clip_norm
    clipper = (
        optax.clip_by_global_norm(max_grad_norm)
        if max_grad_norm is not None
        else None
    )
    clipper_state = optax.EmptyState() if clipper is not None else None

    @eqx.filter_jit
    def train_step(params, opt_state, carry, batch, rng, lr_main, lr_puzzle):
        gb = jnp.asarray(batch["global_batch_size"], dtype=jnp.float32)

        def loss_fn(p):
            model = eqx.combine(p, static_model)
            prepared_carry = prepare_carry(model, carry, batch["data"])
            new_carry, loss, metrics, _, _ = act_loss(
                model,
                prepared_carry,
                rng=rng,
                return_keys=(),
                training=True,
            )
            return loss / gb, (new_carry, metrics, loss)

        (loss, (new_carry, metrics, unscaled_loss)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(params)

        if clipper is not None:
            grads, _ = clipper.update(grads, clipper_state)

        updates, opt_state = optimizer.update(grads, opt_state, params)

        def scale_update(update, label):
            if update is None:
                return None
            lr = jnp.where(jnp.equal(label, 1), lr_puzzle, lr_main)
            return update * lr

        updates = jax.tree.map(scale_update, updates, param_labels)
        params = optax.apply_updates(params, updates)
        return params, opt_state, new_carry, unscaled_loss, metrics

    for _, batch, global_batch_size in train_loader:
        if train_state.step >= train_state.total_steps:
            break
        batch_jnp = batch_to_jnp(batch)

        if train_state.carry is None:
            model = eqx.combine(train_state.params, train_state.static)
            train_state.carry = model.initial_carry(batch_jnp)

        lr_main = compute_lr(config.lr, config, train_state)
        lr_puzzle = compute_lr(config.puzzle_emb_lr, config, train_state)
        rng, step_rng = jax.random.split(train_state.rng)
        batch_pack = {
            "data": batch_jnp,
            "global_batch_size": jnp.asarray(global_batch_size, dtype=jnp.float32),
        }
        (
            new_params,
            train_state.opt_state,
            train_state.carry,
            loss,
            metrics,
        ) = train_step(
            train_state.params,
            train_state.opt_state,
            train_state.carry,
            batch_pack,
            step_rng,
            lr_main,
            lr_puzzle,
        )
        train_state.params = new_params
        train_state.step += 1
        train_state.rng = rng

        ema_helper.update(eqx.combine(train_state.params, train_state.static))

        metric_values = {k: float(v) for k, v in metrics.items()}
        if len(metric_values):
            count = max(metric_values.get("count", 1.0), 1.0)
            logged = {
                f"train/{k}": metric_values[k]
                / (global_batch_size if k.endswith("loss") else count)
                for k in metric_values
            }
            logged["train/lr"] = float(lr_main)
            logged["train/puzzle_lr"] = float(lr_puzzle)
            wandb.log(logged, step=train_state.step)
            progress_bar.update(train_state.step - progress_bar.n)

    wandb.finish()


if __name__ == "__main__":
    train_loop(DEFAULT_CONFIG)
