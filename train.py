from dataclasses import dataclass
from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
import tyro
import tqdm
import wandb
from torch.utils.data import DataLoader

from dataset.dataset import DatasetConfig, DatasetMetadata, GroupDataset
from evaluate import evaluate_logit_lens, render_nearest_neighbors
from trm.encoder import DEFAULT_IGNORE_LABEL_ID, Encoder, EncoderConfig
from trm.losses import act_loss, info_nce_loss
from trm.model import Carry, Model, ModelConfig
from trm.optim import adam_atan2, cosine_warmup_schedule
from trm.utils import EMAHelper


@dataclass
class TrainConfig:
    data_path: str = "data/arc1concept-aug-1000"
    global_group_batch_size: int = 16
    examples_per_view: int = 4
    epochs: int = 50000
    lr: float = 1e-4
    encoder_lr: float = 3e-4
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 2000
    weight_decay: float = 1e-4
    encoder_weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0
    project_name: str = "trm-arc"
    run_name: str = "trm-encoder"
    seed: int = 0
    ema_rate: float = 0.999
    log_every: int = 10
    logit_lens_every: int = 200
    viz_every: int = 200
    viz_neighbors: int = 6
    temperature: float = 0.1
    halt_exploration_prob: float = 0.1
    halt_max_steps: int = 16
    y_cycles: int = 3
    z_cycles: int = 4
    num_layers: int = 2
    hidden_size: int = 512
    num_heads: int = 8
    expansion: int = 4
    forward_dtype: str = "bfloat16"
    task_emb_len: int = 1

    # Encoder
    encoder_hidden_size: int = 512
    encoder_num_layers: int = 4
    encoder_num_heads: int = 8
    encoder_expansion: float = 4.0
    encoder_proj_dim: int = 512
    encoder_rms_norm_eps: float = 1e-5


@dataclass
class TrainState:
    params: eqx.Module
    static: eqx.Module
    opt_state: optax.OptState
    encoder_params: eqx.Module
    encoder_static: eqx.Module
    encoder_opt_state: optax.OptState
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
    batch_size = config.global_group_batch_size * config.examples_per_view
    model_cfg = ModelConfig(
        batch_size=batch_size,
        seq_len=train_metadata.seq_len,
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

    optimizer = adam_atan2(
        beta1=config.beta1,
        beta2=config.beta2,
        weight_decay=config.weight_decay,
    )
    opt_state = optimizer.init(params)
    return params, static, optimizer, opt_state


def create_encoder(config: TrainConfig, metadata: DatasetMetadata, *, key: jnp.ndarray):
    enc_config = EncoderConfig(
        seq_len=metadata.seq_len,
        vocab_size=metadata.vocab_size,
        pad_id=metadata.pad_id,
        ignore_label_id=metadata.ignore_label_id or DEFAULT_IGNORE_LABEL_ID,
        hidden_size=config.encoder_hidden_size,
        num_layers=config.encoder_num_layers,
        num_heads=config.encoder_num_heads,
        expansion=config.encoder_expansion,
        proj_dim=config.encoder_proj_dim,
        forward_dtype=config.forward_dtype,
        rms_norm_eps=config.encoder_rms_norm_eps,
    )
    encoder = Encoder(enc_config, key=key)
    params, static = eqx.partition(encoder, eqx.is_array)
    optimizer = optax.adamw(learning_rate=1.0, weight_decay=config.encoder_weight_decay)
    opt_state = optimizer.init(params)
    return params, static, optimizer, opt_state


def create_train_state(
    config: TrainConfig,
    train_metadata: DatasetMetadata,
    *,
    model_key: jnp.ndarray,
    encoder_key: jnp.ndarray,
    train_key: jnp.ndarray,
):
    total_steps = int(
        config.epochs * train_metadata.total_groups / config.global_group_batch_size
    )
    (
        params,
        static,
        optimizer,
        opt_state,
    ) = create_model(config, train_metadata, key=model_key)
    (
        encoder_params,
        encoder_static,
        encoder_opt,
        encoder_opt_state,
    ) = create_encoder(config, train_metadata, key=encoder_key)

    return (
        TrainState(
            params=params,
            static=static,
            opt_state=opt_state,
            encoder_params=encoder_params,
            encoder_static=encoder_static,
            encoder_opt_state=encoder_opt_state,
            carry=None,
            step=0,
            total_steps=total_steps,
            rng=train_key,
        ),
        optimizer,
        encoder_opt,
    )


def create_dataloader(config: TrainConfig):
    dataset = GroupDataset(
        DatasetConfig(
            seed=config.seed,
            dataset_paths=[config.data_path],
            rank=0,
            num_replicas=1,
            global_batch_size=config.global_group_batch_size,
            epochs_per_iter=config.epochs,
            test_set_mode=False,
        ),
        split="train",
        examples_per_view=config.examples_per_view,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=True,
    )
    return dataloader, dataset.metadata


def batch_to_jnp(batch: Dict[str, torch.Tensor]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in batch.items()}


def batch_to_np(batch: Dict[str, torch.Tensor]) -> Dict[str, jnp.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in batch.items()}


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


def flatten_batch(
    batch: Dict[str, jnp.ndarray]
) -> Dict[str, jnp.ndarray]:
    b, k, s = batch["inputs_1"].shape
    flat_inputs1 = batch["inputs_1"].reshape(b * k, s)
    flat_labels1 = batch["labels_1"].reshape(b * k, s)
    flat_inputs2 = batch["inputs_2"].reshape(b * k, s)
    flat_labels2 = batch["labels_2"].reshape(b * k, s)

    flat_inputs = jnp.concatenate([flat_inputs1, flat_inputs2], axis=0)
    flat_labels = jnp.concatenate([flat_labels1, flat_labels2], axis=0)

    puzzle_ids = jnp.repeat(batch["group_ids"], 2 * k, axis=0)
    return {
        "inputs": flat_inputs,
        "labels": flat_labels,
        "puzzle_identifiers": puzzle_ids,
    }


def make_train_step(
    static_model,
    static_encoder,
    model_optimizer,
    encoder_optimizer,
    clipper,
    *,
    examples_per_view: int,
    temperature: float,
):
    @eqx.filter_jit
    def train_step(
        params,
        opt_state,
        encoder_params,
        encoder_opt_state,
        carry,
        batch,
        rng,
        lr_main,
        lr_encoder,
    ):
        def loss_fn(all_params):
            p, ep = all_params
            model = eqx.combine(p, static_model)
            encoder = eqx.combine(ep, static_encoder)

            trm_batch = flatten_batch(batch)
            effective_carry = filter_carry(model, carry, trm_batch)

            gb_examples = jnp.asarray(trm_batch["inputs"].shape[0], dtype=jnp.float32)

            flat_inputs = effective_carry.data["inputs"]
            flat_labels = effective_carry.data["labels"]
            z_trm = encoder.encode_examples(flat_inputs, flat_labels)
            task_emb = jax.lax.stop_gradient(z_trm)[:, None, :]

            new_carry, trm_loss, trm_metrics, _ = act_loss(
                model, effective_carry, rng=rng, training=True, task_emb=task_emb
            )

            _, g1 = encoder.encode_views(batch["inputs_1"], batch["labels_1"])
            _, g2 = encoder.encode_views(batch["inputs_2"], batch["labels_2"])

            info_loss, info_metrics = info_nce_loss(g1, g2, temperature=temperature)

            total_loss = trm_loss / gb_examples + info_loss

            metrics = {
                **trm_metrics,
                "trm_loss": trm_loss,
                **{f"encoder/{k}": v for k, v in info_metrics.items()},
                "count": gb_examples,
            }
            aux = {
                "carry": new_carry,
                "trm_loss": trm_loss,
                "info_loss": info_loss,
                "metrics": metrics,
            }
            return total_loss, aux

        (loss, aux), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )((params, encoder_params))

        grads, _ = clipper.update(grads, optax.EmptyState())
        grads_model, grads_encoder = grads

        updates, opt_state = model_optimizer.update(grads_model, opt_state, params)
        updates = jax.tree.map(
            lambda u: u * lr_main if u is not None else None, updates
        )
        params = optax.apply_updates(params, updates)

        enc_updates, encoder_opt_state = encoder_optimizer.update(
            grads_encoder, encoder_opt_state, encoder_params
        )
        enc_updates = jax.tree.map(
            lambda u: u * lr_encoder if u is not None else None, enc_updates
        )
        encoder_params = optax.apply_updates(encoder_params, enc_updates)

        return (
            params,
            opt_state,
            encoder_params,
            encoder_opt_state,
            aux["carry"],
            loss,
            aux["metrics"],
        )

    return train_step


def main(config: TrainConfig = TrainConfig()):
    torch.random.manual_seed(config.seed)
    train_loader, metadata = create_dataloader(config)

    rng = jax.random.PRNGKey(config.seed)
    rngs = jax.random.split(rng, 4)
    rng, model_key, encoder_key, train_key = rngs

    train_state, optimizer, encoder_opt = create_train_state(
        config, metadata, model_key=model_key, encoder_key=encoder_key, train_key=train_key
    )

    ema_helper = EMAHelper(mu=config.ema_rate)
    ema_helper.register(eqx.combine(train_state.params, train_state.static))

    clipper = optax.clip_by_global_norm(config.grad_clip_norm)
    train_step = make_train_step(
        train_state.static,
        train_state.encoder_static,
        optimizer,
        encoder_opt,
        clipper,
        examples_per_view=config.examples_per_view,
        temperature=config.temperature,
    )

    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=config.__dict__,
        settings=wandb.Settings(_disable_stats=True),
    )
    progress_bar = tqdm.tqdm(total=train_state.total_steps)

    for _, batch, _ in train_loader:
        if train_state.step >= train_state.total_steps:
            break

        batch_jnp = batch_to_jnp(batch)

        if train_state.carry is None:
            model_for_carry = eqx.combine(train_state.params, train_state.static)
            trm_batch = flatten_batch(batch_jnp)
            train_state.carry = model_for_carry.initial_carry(trm_batch)

        lr_main = compute_lr(config.lr, config, train_state)
        lr_encoder = compute_lr(config.encoder_lr, config, train_state)

        rng, step_rng = jax.random.split(train_state.rng)

        (
            train_state.params,
            train_state.opt_state,
            train_state.encoder_params,
            train_state.encoder_opt_state,
            new_carry,
            loss,
            metrics,
        ) = train_step(
            train_state.params,
            train_state.opt_state,
            train_state.encoder_params,
            train_state.encoder_opt_state,
            train_state.carry,
            batch_jnp,
            step_rng,
            lr_main,
            lr_encoder,
        )

        train_state.step += 1
        train_state.rng = rng
        train_state.carry = new_carry
        ema_helper.update(eqx.combine(train_state.params, train_state.static))
        progress_bar.update(train_state.step - progress_bar.n)

        if train_state.step % config.log_every == 0:
            metric_values = {k: float(v) for k, v in metrics.items()}
            count = metric_values.get("count", 1.0)
            logged = {
                "train/lr": float(lr_main),
                "train/encoder_lr": float(lr_encoder),
            }
            safe_count = max(count, 1.0)

            for k, v in metric_values.items():
                name = f"train/{k}"
                if k.endswith("loss") and not k.startswith("encoder/"):
                    logged[name] = v / safe_count
                elif k in ("accuracy", "exact_accuracy", "q_halt_accuracy", "steps"):
                    logged[name] = v / safe_count
                else:
                    logged[name] = v

            logged["train/loss"] = float(loss)
            wandb.log(logged, step=train_state.step)

        if config.viz_every > 0 and train_state.step % config.viz_every == 0:
            encoder_model = eqx.combine(
                train_state.encoder_params, train_state.encoder_static
            )
            viz_img = render_nearest_neighbors(
                encoder_model,
                batch_to_np(batch),
                metadata,
                top_k=config.viz_neighbors,
            )
            wandb.log(
                {"train/nearest_neighbors": wandb.Image(viz_img)},
                step=train_state.step,
            )

        if config.logit_lens_every > 0 and train_state.step % config.logit_lens_every == 0:
            lens_model = ema_helper.ema_copy()
            encoder_model = eqx.combine(
                train_state.encoder_params, train_state.encoder_static
            )
            _, g1 = encoder_model.encode_views(
                batch_jnp["inputs_1"], batch_jnp["labels_1"]
            )
            task_emb = jnp.repeat(g1, config.examples_per_view, axis=0)
            task_emb = task_emb.reshape(task_emb.shape[0], 1, -1)
            trm_batch = flatten_batch(batch_jnp)
            evaluate_logit_lens(
                lens_model,
                trm_batch,
                metadata,
                task_emb=task_emb,
                step=train_state.step,
                rng=train_state.rng,
            )

    wandb.finish()


if __name__ == "__main__":
    tyro.cli(main)()
