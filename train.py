import os
from dataclasses import dataclass
from typing import Any, Dict

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
from trm.losses import act_loss
from trm.model import Carry, Model, ModelConfig
from trm.optim import adam_atan2, sparse_sign_sgd, cosine_warmup_schedule
from trm.utils import EMAHelper


@dataclass
class TrainConfig:
    data_path: str = "data/arc1concept-aug-1000"
    global_batch_size: int = 768
    epochs: int = 100000
    eval_every: int = 200
    eval_batches: int = 1
    lr: float = 1e-4
    lr_min_ratio: float = 1.0
    lr_warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    task_emb_lr: float = 1e-2
    task_emb_weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    project_name: str = "Arc1concept-aug-1000-ACT-torch"
    run_name: str = "trm-v1"
    seed: int = 0
    ema_rate: float = 0.999
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
    per_device_batch_size: int,
):
    model_cfg = ModelConfig(
        batch_size=per_device_batch_size,
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
    per_device_batch_size: int,
):
    total_steps = int(
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        / config.global_batch_size
    )
    params, static, optimizer, opt_state, param_labels = create_model(
        config,
        train_metadata,
        key=model_key,
        per_device_batch_size=per_device_batch_size,
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


def create_dataloader(
    config: TrainConfig, split: str, rank: int = 0, world_size: int = 1, **kwargs
):
    dataset = Dataset(
        DatasetConfig(
            seed=config.seed,
            dataset_paths=[config.data_path],
            rank=rank,
            num_replicas=world_size,
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


def shard_batch(batch: Dict[str, jnp.ndarray], num_devices: int, per_device_batch: int):
    def _reshape(x):
        return x.reshape((num_devices, per_device_batch) + x.shape[1:])

    return {k: _reshape(v) for k, v in batch.items()}


def infinite_dataloader(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def batch_to_jnp(batch: Dict[str, torch.Tensor]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in batch.items()}


def unreplicate(tree):
    return jax.tree.map(lambda x: x[0], tree)


def log_metrics(
    metrics: Dict[str, jnp.ndarray],
    lr_main: jnp.ndarray,
    lr_task: jnp.ndarray,
    step: int,
    config: TrainConfig,
):
    mv = {k: float(v) for k, v in metrics.items()}
    count = mv.get("count", 0.0)
    if count <= 0:
        return
    logged = {
        f"train/{k}": (
            mv[k] / config.global_batch_size if k.endswith("loss") else mv[k] / count
        )
        for k in mv
    }
    logged["train/lr"] = float(lr_main)
    logged["train/task_lr"] = float(lr_task)
    wandb.log(logged, step=step)


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


def evaluate_model(
    model: Model,
    dataloader: Any,
    rng: jnp.ndarray,
    num_batches: int,
    devices,
    per_device_batch: int,
) -> Dict[str, float]:
    eval_step = make_eval_step(model.config.halt_max_steps)
    model_repl = jax.device_put_replicated(model, devices)
    num_devices = len(devices)
    totals = None

    for _ in range(num_batches):
        _, batch, _ = next(dataloader)
        b = batch_to_jnp(batch)
        b = shard_batch(b, num_devices, per_device_batch)
        rng, sr = jax.random.split(rng)
        sr = jnp.stack(jax.random.split(sr, num_devices))
        m = unreplicate(eval_step(model_repl, b, sr))
        totals = m if totals is None else jtu.tree_map(lambda a, b: a + b, totals, m)

    tf = {k: float(v) for k, v in totals.items()}
    count = tf.get("count", 0.0)
    if count <= 0:
        return {}

    tf["lm_loss"] /= float(model.config.halt_max_steps)
    return {f"test/{k}": tf[k] / count for k in tf if k != "count"}


def make_eval_step(max_steps: int):
    max_steps = int(max_steps) + 1

    @eqx.filter_jit
    def evaluate_rollout(
        model: Model, carry: Carry, rng: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        rngs = jax.random.split(rng, max_steps)

        def step_fn(state, rng_step):
            carry_in, finished_in = state
            warmed = model.warmup_carry(carry_in)
            carry_out, _loss, metrics_out, all_finish = act_loss(
                model, warmed, rng=rng_step, training=False
            )
            metrics_masked = jtu.tree_map(
                lambda x: jnp.where(finished_in, jnp.zeros_like(x), x), metrics_out
            )
            finished_out = jnp.logical_or(finished_in, all_finish)
            return (carry_out, finished_out), metrics_masked

        (_, _), metrics_seq = jax.lax.scan(step_fn, (carry, jnp.array(False)), rngs)
        return jtu.tree_map(lambda x: jnp.sum(x, axis=0), metrics_seq)

    def eval_step(model, batch_data, rng):
        carry = model.initial_carry(batch_data)
        carry = filter_carry(model, carry, batch_data)
        metrics = evaluate_rollout(model, carry, rng)
        return jax.tree.map(lambda x: jax.lax.psum(x, axis_name="devices"), metrics)

    return jax.pmap(eval_step, axis_name="devices")


def make_train_step(optimizer, param_labels, clipper):
    def train_step_fn(
        params,
        opt_state,
        carry,
        batch_data,
        local_batch_size,
        rng,
        lr_main,
        lr_task,
        static_repl,
    ):
        lb = jnp.asarray(local_batch_size, dtype=jnp.float32)

        model_for_warmup = eqx.combine(params, static_repl)
        filtered_carry = filter_carry(model_for_warmup, carry, batch_data)
        warmed_carry = model_for_warmup.warmup_carry(filtered_carry)

        def loss_fn(p):
            model = eqx.combine(p, static_repl)
            new_carry, loss, metrics, _ = act_loss(
                model, warmed_carry, rng=rng, training=True
            )
            return loss / lb, (new_carry, metrics, loss)

        (loss, (new_carry, metrics, unscaled_loss)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(params)

        grads = jax.lax.pmean(grads, axis_name="devices")
        metrics = jax.tree.map(lambda x: jax.lax.psum(x, axis_name="devices"), metrics)
        unscaled_loss = jax.lax.psum(unscaled_loss, axis_name="devices")

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

    return jax.pmap(train_step_fn, axis_name="devices")


def main(config: TrainConfig = TrainConfig()):
    torch.random.manual_seed(config.seed)

    num_devices = jax.local_device_count()
    per_device_batch = config.global_batch_size // num_devices
    global_batch_size = config.global_batch_size

    train_epochs_per_iter = config.eval_every
    total_iters = config.epochs // train_epochs_per_iter

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=global_batch_size,
        rank=0,
        world_size=1,
    )
    test_loader, test_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=global_batch_size,
        rank=0,
        world_size=1,
    )

    test_loader_iter = infinite_dataloader(test_loader)

    rng = jax.random.PRNGKey(config.seed)
    rngs = jax.random.split(rng, 4)
    rng, model_key, train_key, eval_rng = rngs

    train_state, optimizer, param_labels = create_train_state(
        config,
        train_metadata,
        model_key=model_key,
        train_key=train_key,
        per_device_batch_size=per_device_batch,
    )

    devices = jax.local_devices()
    static_host = train_state.static
    train_state = TrainState(
        params=jax.device_put_replicated(train_state.params, devices),
        static=jax.device_put_replicated(train_state.static, devices),
        opt_state=jax.device_put_replicated(train_state.opt_state, devices),
        carry=None,
        step=train_state.step,
        total_steps=train_state.total_steps,
        rng=train_state.rng,
    )

    progress_bar = tqdm.tqdm(total=train_state.total_steps)
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=config.__dict__,
        settings=wandb.Settings(_disable_stats=True),
    )

    params_host = unreplicate(train_state.params)
    ema_helper = EMAHelper(mu=config.ema_rate)
    ema_helper.register(eqx.combine(params_host, static_host))
    clipper = optax.clip_by_global_norm(config.grad_clip_norm)

    train_step = make_train_step(optimizer, param_labels, clipper)

    @jax.pmap
    def init_carry(params, static_model, batch):
        model = eqx.combine(params, static_model)
        return model.initial_carry(batch)

    for _ in range(total_iters):
        for _, batch, _ in train_loader:
            if train_state.step >= train_state.total_steps:
                break

            batch_jnp = batch_to_jnp(batch)
            batch_sharded = shard_batch(batch_jnp, num_devices, per_device_batch)

            if train_state.carry is None:
                train_state.carry = init_carry(
                    train_state.params, train_state.static, batch_sharded
                )

            lr_main = compute_lr(config.lr, config, train_state)
            lr_task = compute_lr(config.task_emb_lr, config, train_state)
            rngs = jax.random.split(train_state.rng, num_devices + 1)
            rng, step_rng = rngs[0], jnp.stack(rngs[1:])

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
                batch_sharded,
                jax.device_put_replicated(
                    jnp.asarray(per_device_batch, dtype=jnp.float32), devices
                ),
                step_rng,
                jax.device_put_replicated(lr_main, devices),
                jax.device_put_replicated(lr_task, devices),
                train_state.static,
            )
            train_state.params = new_params
            train_state.step += 1
            train_state.rng = rng

            params_host = unreplicate(train_state.params)
            ema_helper.update(eqx.combine(params_host, static_host))

            metrics_host = unreplicate(metrics)
            log_metrics(metrics_host, lr_main, lr_task, train_state.step, config)

            progress_bar.update(train_state.step - progress_bar.n)

        if train_state.step >= train_state.total_steps:
            break

        eval_model = ema_helper.ema_copy()
        eval_rng, eval_step_rng = jax.random.split(eval_rng)
        eval_logs = evaluate_model(
            eval_model,
            test_loader_iter,
            rng=eval_step_rng,
            num_batches=config.eval_batches,
            devices=devices,
            per_device_batch=per_device_batch,
        )
        wandb.log(eval_logs, step=train_state.step)

        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = os.path.join("checkpoints", f"{config.run_name}.eqx")
        eqx.tree_serialise_leaves(ckpt_path, eqx.combine(params_host, static_host))

    wandb.finish()


if __name__ == "__main__":
    tyro.cli(main)()
