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
import tyro

from dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from evaluate import evaluate_model, evaluate_logit_lens
from trm.model import Model
from trm.utils import EMAHelper
from trm.optim import adam_atan2



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
    grad_clip_norm: float | None
    project_name: str
    run_name: str
    seed: int
    ema_rate: float
    eval_every: int
    logit_lens_every: int
    min_outer_steps: int
    max_outer_steps: int
    model: Dict[str, Any]


DEFAULT_CONFIG = TrainConfig(
    data_paths=["data/maze-30x30-hard-1k"],
    global_batch_size=96,
    epochs=50000,
    lr=1e-4,
    lr_min_ratio=1.0,
    lr_warmup_steps=2000,
    weight_decay=1.0,
    beta1=0.9,
    beta2=0.95,
    grad_clip_norm=1.0,
    project_name="energy-trm",
    run_name="energy-logits",
    seed=0,
    ema_rate=0.999,
    eval_every=1000,
    logit_lens_every=200,
    min_outer_steps=6,
    max_outer_steps=20,
    model=dict(
        y_cycles=3,
        z_cycles=4,
        num_layers=2,
        hidden_size=512,
        num_heads=8,
        expansion=4,
        z_vocab_size=64,
        forward_dtype="bfloat16",
        energy_step_size_min=0.05,
        energy_step_size_max=0.15,
        energy_noise_scale=0.1,
        max_outer_steps=20,
    ),
)


@dataclass
class TrainState:
    params: eqx.Module
    static: eqx.Module
    opt_state: optax.OptState
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
    )
    key = jax.random.PRNGKey(config.seed)
    model = Model(model_cfg, key=key)
    params, static = eqx.partition(model, eqx.is_array)

    optimizer = adam_atan2(
        beta1=config.beta1,
        beta2=config.beta2,
        weight_decay=config.weight_decay,
    )
    opt_state = optimizer.init(params)
    return params, static, optimizer, opt_state


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


def create_train_state(
    config: TrainConfig,
    train_metadata: PuzzleDatasetMetadata,
):
    total_steps = int(
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        / config.global_batch_size
    )
    params, static, optimizer, opt_state = create_model(config, train_metadata)
    rng = jax.random.PRNGKey(config.seed + 1)
    return (
        TrainState(
            params=params,
            static=static,
            opt_state=opt_state,
            step=0,
            total_steps=total_steps,
            rng=rng,
        ),
        optimizer,
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


def compute_outer_steps(config: TrainConfig, train_state: TrainState) -> int:
    if train_state.total_steps <= 1:
        return config.max_outer_steps
    progress = min(train_state.step / float(train_state.total_steps - 1), 1.0)
    span = config.max_outer_steps - config.min_outer_steps
    return int(round(config.min_outer_steps + progress * span))


def sample_step_size(rng: jnp.ndarray, model_config) -> jnp.ndarray:
    cfg = model_config
    return jax.random.uniform(
        rng,
        (),
        minval=cfg.energy_step_size_min,
        maxval=cfg.energy_step_size_max,
        dtype=jnp.float32,
    )


def infinite_dataloader(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def train_loop(config: TrainConfig):
    torch.random.manual_seed(config.seed)

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=config.epochs,
        global_batch_size=config.global_batch_size,
    )
    test_loader, test_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )

    test_loader_iter = infinite_dataloader(test_loader)

    train_state, optimizer = create_train_state(config, train_metadata)

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
    eval_rng = jax.random.PRNGKey(config.seed + 42)
    logit_lens_rng = jax.random.PRNGKey(config.seed + 4242)
    max_grad_norm = config.grad_clip_norm
    clipper = (
        optax.clip_by_global_norm(max_grad_norm) if max_grad_norm is not None else None
    )
    clipper_state = optax.EmptyState() if clipper is not None else None

    @eqx.filter_jit
    def train_step(params, opt_state, batch, rng, lr_main, num_outer_steps, step_size):
        rollout_steps = jnp.asarray(num_outer_steps, dtype=jnp.int32)

        def loss_fn(p, key):
            model = eqx.combine(p, static_model)
            loss, metrics = model.loss(
                batch["inputs"],
                batch["labels"],
                rng=key,
                num_outer_steps=rollout_steps,
                step_size=step_size,
                training=True,
            )
            return loss, metrics

        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            params, rng
        )

        if clipper is not None:
            grads, _ = clipper.update(grads, clipper_state)

        updates, opt_state = optimizer.update(grads, opt_state, params)

        def scale_update(update):
            if update is None:
                return None
            return update * lr_main

        updates = jax.tree.map(scale_update, updates)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    for _, batch, _ in train_loader:
        if train_state.step >= train_state.total_steps:
            break
        batch_jnp = batch_to_jnp(batch)

        lr_main = compute_lr(config.lr, config, train_state)
        train_state.rng, step_rng = jax.random.split(train_state.rng)
        step_rng, loss_rng = jax.random.split(step_rng)
        outer_steps = compute_outer_steps(config, train_state)
        step_size = sample_step_size(step_rng, static_model.config)
        (
            new_params,
            train_state.opt_state,
            loss,
            metrics,
        ) = train_step(
            train_state.params,
            train_state.opt_state,
            batch_jnp,
            loss_rng,
            lr_main,
            outer_steps,
            step_size,
        )
        train_state.params = new_params
        train_state.step += 1

        ema_helper.update(eqx.combine(train_state.params, static_model))

        metric_values = {k: float(v) for k, v in metrics.items()}
        if len(metric_values):
            logged = {
                "train/loss": metric_values["loss"],
                "train/token_accuracy": metric_values["token_accuracy"],
                "train/seq_accuracy": metric_values["seq_accuracy"],
                "train/lr": float(lr_main),
                "train/outer_steps": float(outer_steps),
                "train/step_size": float(step_size),
            }
            wandb.log(logged, step=train_state.step)
            progress_bar.update(train_state.step - progress_bar.n)

        if config.eval_every > 0 and train_state.step % config.eval_every == 0:
            eval_model = ema_helper.ema_copy()
            eval_rng, eval_step_rng = jax.random.split(eval_rng)
            eval_logs = evaluate_model(
                eval_model,
                test_loader,
                batch_converter=batch_to_jnp,
                rng=eval_step_rng,
                num_outer_steps=outer_steps,
            )
            if eval_logs:
                wandb.log(eval_logs, step=train_state.step)
        if (
            config.logit_lens_every > 0
            and train_state.step % config.logit_lens_every == 0
        ):
            lens_model = ema_helper.ema_copy()
            _, sampled_batch, _ = next(test_loader_iter)
            test_batch = batch_to_jnp(sampled_batch)
            logit_lens_rng, lens_step_rng = jax.random.split(logit_lens_rng)
            evaluate_logit_lens(
                lens_model,
                test_batch,
                test_metadata,
                step=train_state.step,
                rng=lens_step_rng,
            )

    wandb.finish()


def main(config: TrainConfig = DEFAULT_CONFIG):
    train_loop(config)


if __name__ == "__main__":
    tyro.cli(main)()
