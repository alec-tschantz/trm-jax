import math
from dataclasses import dataclass
from typing import Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import torch
import tyro
import tqdm
import wandb
from torch.utils.data import DataLoader

from dataset import Dataset, DatasetConfig, DatasetMetadata
from evaluate import evaluate_model, evaluate_logit_lens
from trm.losses import act_loss
from trm.model import Carry, Model, ModelConfig
from trm.optim import adam_atan2, sparse_sign_sgd, cosine_warmup_schedule
from trm.utils import EMAHelper


@dataclass
class TrainConfig:
    data_path: str = "data/maze-30x30-hard-1k"
    global_batch_size: int = 192
    epochs: int = 50000
    lr: float = 1e-4
    lr_min_ratio: float = 1.0
    lr_warmup_steps: int = 2000
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    task_emb_lr: float = 1e-4
    task_emb_weight_decay: float = 1.0
    grad_clip_norm: float = 1.0
    project_name: str = "trm-arc"
    run_name: str = "trm-maze"
    seed: int = 0
    ema_rate: float = 0.999
    eval_every: int = 1000
    logit_lens_every: int = 200
    halt_exploration_prob: float = 0.1
    halt_max_steps: int = 16
    y_cycles: int = 3
    z_cycles: int = 4
    num_layers: int = 2
    hidden_size: int = 512
    num_heads: int = 8
    expansion: int = 4
    task_emb_ndim: int = 512
    forward_dtype: str = "bfloat16"
    task_emb_len: int = 16


@dataclass
class TrainState:
    params: eqx.Module
    static: eqx.Module
    opt_state: optax.OptState
    carry: Carry | None
    step: int
    total_steps: int
    rng: jnp.ndarray


def create_model(
    config: TrainConfig,
    train_metadata: DatasetMetadata,
    *,
    key: jnp.ndarray,
):
    model_cfg = ModelConfig(
        batch_size=config.global_batch_size,
        seq_len=train_metadata.seq_len,
        task_emb_ndim=config.task_emb_ndim,
        num_task_identifiers=train_metadata.num_puzzle_identifiers,
        vocab_size=train_metadata.vocab_size,
        y_cycles=config.y_cycles,
        z_cycles=config.z_cycles,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        expansion=config.expansion,
        num_heads=config.num_heads,
        halt_max_steps=config.halt_max_steps,
        halt_exploration_prob=config.halt_exploration_prob,
        forward_dtype=config.forward_dtype,
        task_emb_len=config.task_emb_len,
    )

    model = Model(model_cfg, key=key)
    params, static = eqx.partition(model, eqx.is_array)

    def build_param_labels(p):
        labels = jtu.tree_map(lambda _: 0, p)
        labels = eqx.tree_at(lambda tree: tree.task_embed.weight, labels, 1)
        return labels

    param_labels = build_param_labels(params)

    transforms = {
        0: adam_atan2(
            beta1=config.beta1,
            beta2=config.beta2,
            weight_decay=config.weight_decay,
        ),
        1: sparse_sign_sgd(weight_decay=config.task_emb_weight_decay),
    }

    optimizer = optax.multi_transform(transforms, build_param_labels)
    opt_state = optimizer.init(params)
    return params, static, optimizer, opt_state, param_labels


def create_train_state(
    config: TrainConfig,
    train_metadata: DatasetMetadata,
    *,
    model_key: jnp.ndarray,
    train_key: jnp.ndarray,
):
    total_steps = int(
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        / config.global_batch_size
    )
    params, static, optimizer, opt_state, param_labels = create_model(
        config, train_metadata, key=model_key
    )
    return (
        TrainState(
            params=params,
            static=static,
            opt_state=opt_state,
            carry=None,
            step=0,
            total_steps=total_steps,
            rng=train_key,
        ),
        optimizer,
        param_labels,
    )


def create_dataloader(config: TrainConfig, split: str, **kwargs):
    dataset = Dataset(
        DatasetConfig(
            seed=config.seed,
            dataset_paths=[config.data_path],
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


def infinite_dataloader(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def batch_to_jnp(batch: Dict[str, torch.Tensor]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in batch.items()}


def compute_lr(base_lr: float, config: TrainConfig, train_state: TrainState):
    lr = cosine_warmup_schedule(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )
    return jnp.array(lr, dtype=jnp.float32)


def filter_carry(model: Model, carry: Carry, batch: Dict[str, jnp.ndarray]) -> Carry:
    new_states = model.reset_states(carry.halted, carry.states)
    new_steps = jnp.where(carry.halted, 0, carry.steps)
    halted = carry.halted
    data = {
        k: jnp.where(
            halted.reshape((-1,) + (1,) * (batch[k].ndim - 1)),
            batch[k],
            carry.data[k],
        )
        for k in batch
    }
    return Carry(
        states=new_states,
        steps=new_steps,
        halted=halted,
        data=data,
    )


def make_train_step(static_model, optimizer, param_labels, clipper):
    @eqx.filter_jit
    def train_step(
        params, opt_state, carry, batch_data, global_batch_size, rng, lr_main, lr_task
    ):
        gb = jnp.asarray(global_batch_size, dtype=jnp.float32)

        def loss_fn(p):
            model = eqx.combine(p, static_model)
            inp_carry = filter_carry(model, carry, batch_data)
            new_carry, loss, metrics, _ = act_loss(
                model, inp_carry, rng=rng, training=True
            )
            return loss / gb, (new_carry, metrics, loss)

        (loss, (new_carry, metrics, unscaled_loss)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(params)

        grads, _ = clipper.update(grads, optax.EmptyState())
        updates, opt_state = optimizer.update(grads, opt_state, params)

        def scale_update(update, label):
            if update is None:
                return None
            lr = jnp.where(jnp.equal(label, 1), lr_task, lr_main)
            return update * lr

        updates = jax.tree.map(scale_update, updates, param_labels)
        params = optax.apply_updates(params, updates)
        return params, opt_state, new_carry, unscaled_loss, metrics

    return train_step


def main(config: TrainConfig = TrainConfig()):
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

    rng = jax.random.PRNGKey(config.seed)
    rngs = jax.random.split(rng, 5)
    rng, model_key, train_key, eval_rng, logit_lens_rng = rngs

    train_state, optimizer, param_labels = create_train_state(
        config, train_metadata, model_key=model_key, train_key=train_key
    )

    progress_bar = tqdm.tqdm(total=train_state.total_steps)
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=config.__dict__,
        settings=wandb.Settings(_disable_stats=True),
    )

    ema_helper = EMAHelper(mu=config.ema_rate)
    ema_helper.register(eqx.combine(train_state.params, train_state.static))
    clipper = optax.clip_by_global_norm(config.grad_clip_norm)

    train_step = make_train_step(train_state.static, optimizer, param_labels, clipper)

    for _, batch, global_batch_size in train_loader:
        if train_state.step >= train_state.total_steps:
            break

        batch_jnp = batch_to_jnp(batch)

        if train_state.carry is None:
            model = eqx.combine(train_state.params, train_state.static)
            train_state.carry = model.initial_carry(batch_jnp)

        lr_main = compute_lr(config.lr, config, train_state)
        lr_task = compute_lr(config.task_emb_lr, config, train_state)
        rng, step_rng = jax.random.split(train_state.rng)

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
            batch_jnp,
            jnp.asarray(global_batch_size, dtype=jnp.float32),
            step_rng,
            lr_main,
            lr_task,
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
            logged["train/task_lr"] = float(lr_task)
            wandb.log(logged, step=train_state.step)
            progress_bar.update(train_state.step - progress_bar.n)

        if config.eval_every > 0 and train_state.step % config.eval_every == 0:
            eval_model = ema_helper.ema_copy()
            eval_rng, eval_step_rng = jax.random.split(eval_rng)
            eval_logs = evaluate_model(
                eval_model,
                test_loader,
                batch_converter=batch_to_jnp,
                filter_carry_fn=filter_carry,
                rng=eval_step_rng,
            )
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
                filter_carry_fn=filter_carry,
                step=train_state.step,
                rng=lens_step_rng,
            )

    wandb.finish()


if __name__ == "__main__":
    tyro.cli(main)()
