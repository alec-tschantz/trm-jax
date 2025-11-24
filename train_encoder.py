from dataclasses import dataclass
from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import tyro
import tqdm
import wandb
from torch.utils.data import DataLoader

from dataset.dataset import DatasetConfig, DatasetMetadata, GroupDataset
from evaluate import render_nearest_neighbors
from trm.encoder import DEFAULT_IGNORE_LABEL_ID, Encoder, EncoderConfig
from trm.losses import info_nce_loss
from trm.optim import cosine_warmup_schedule


@dataclass
class TrainEncoderConfig:
    data_path: str = "data/arc1concept-aug-1000"
    global_group_batch_size: int = 16
    epochs: int = 20000
    lr: float = 3e-4
    lr_warmup_steps: int = 1000
    lr_min_ratio: float = 0.1
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    seed: int = 0
    project_name: str = "trm-encoder"
    run_name: str = "encoder"
    examples_per_view: int = 4
    log_every: int = 10
    viz_every: int = 200
    viz_neighbors: int = 6
    temperature: float = 0.1
    forward_dtype: str = "bfloat16"
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 8
    expansion: float = 4.0
    proj_dim: int = 128
    rms_norm_eps: float = 1e-5


@dataclass
class TrainState:
    params: eqx.Module
    static: eqx.Module
    opt_state: optax.OptState
    step: int
    total_steps: int
    rng: jnp.ndarray


def create_dataloader(config: TrainEncoderConfig, split: str):
    dataset = GroupDataset(
        DatasetConfig(
            seed=config.seed,
            dataset_paths=[config.data_path],
            global_batch_size=config.global_group_batch_size,
            test_set_mode=split == "test",
            epochs_per_iter=config.epochs,
            rank=0,
            num_replicas=1,
        ),
        split=split,
        examples_per_view=config.examples_per_view,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=True,
    )
    return dataloader, dataset.metadata


def create_encoder_and_state(
    config: TrainEncoderConfig, metadata: DatasetMetadata, *, key: jnp.ndarray
):
    enc_config = EncoderConfig(
        seq_len=metadata.seq_len,
        vocab_size=metadata.vocab_size,
        pad_id=metadata.pad_id,
        ignore_label_id=DEFAULT_IGNORE_LABEL_ID,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        expansion=config.expansion,
        proj_dim=config.proj_dim,
        forward_dtype=config.forward_dtype,
        rms_norm_eps=config.rms_norm_eps,
    )
    model = Encoder(enc_config, key=key)
    params, static = eqx.partition(model, eqx.is_array)

    optimizer = optax.adamw(learning_rate=1.0, weight_decay=config.weight_decay)
    opt_state = optimizer.init(params)

    total_steps = int(
        config.epochs * metadata.total_groups / config.global_group_batch_size
    )
    train_state = TrainState(
        params=params,
        static=static,
        opt_state=opt_state,
        step=0,
        total_steps=total_steps,
        rng=key,
    )
    return model, train_state, optimizer


def make_train_step(static_encoder, optimizer, clipper):
    @eqx.filter_jit
    def train_step(params, opt_state, batch, temperature, lr):
        def loss_fn(p):
            model = eqx.combine(p, static_encoder)
            ex1, g1 = model.encode_views(batch["inputs_1"], batch["labels_1"])
            ex2, g2 = model.encode_views(batch["inputs_2"], batch["labels_2"])

            loss, contrastive_metrics = info_nce_loss(g1, g2, temperature=temperature)
            metrics = {
                **contrastive_metrics,
                "example_norm": jnp.mean(
                    jnp.linalg.norm(
                        jnp.concatenate(
                            [
                                ex1.reshape(-1, ex1.shape[-1]),
                                ex2.reshape(-1, ex2.shape[-1]),
                            ]
                        ),
                        axis=-1,
                    )
                ),
                "count": g1.shape[0],
            }
            return loss, metrics

        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            params
        )

        grads, _ = clipper.update(grads, optax.EmptyState())
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = jax.tree.map(lambda u: u * lr if u is not None else None, updates)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    return train_step


def batch_to_jnp(batch: Dict[str, torch.Tensor]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in batch.items()}


def batch_to_np(batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in batch.items()}


def compute_lr(config: TrainEncoderConfig, step: int, total_steps: int):
    lr = cosine_warmup_schedule(
        current_step=step,
        base_lr=config.lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=total_steps,
        min_ratio=config.lr_min_ratio,
    )
    return jnp.array(lr, dtype=jnp.float32)




def main(config: TrainEncoderConfig = TrainEncoderConfig()):
    torch.random.manual_seed(config.seed)
    train_loader, metadata = create_dataloader(config, "train")

    rng = jax.random.PRNGKey(config.seed)
    rng, model_key, train_key = jax.random.split(rng, 3)
    model, train_state, optimizer = create_encoder_and_state(
        config, metadata, key=model_key
    )
    train_state.rng = train_key
    clipper = optax.clip_by_global_norm(config.grad_clip_norm)
    train_step = make_train_step(train_state.static, optimizer, clipper)

    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=config.__dict__,
        settings=wandb.Settings(_disable_stats=True),
    )
    progress_bar = tqdm.tqdm(total=train_state.total_steps)

    for _, batch, _global_group_batch in train_loader:
        if train_state.step >= train_state.total_steps:
            break

        batch_jnp = batch_to_jnp(batch)
        lr = compute_lr(config, train_state.step, train_state.total_steps)
        temperature = jnp.asarray(config.temperature, dtype=jnp.float32)
        (
            train_state.params,
            train_state.opt_state,
            loss,
            metrics,
        ) = train_step(
            train_state.params,
            train_state.opt_state,
            batch_jnp,
            temperature,
            lr,
        )

        train_state.step += 1
        train_state.rng, _ = jax.random.split(train_state.rng)
        progress_bar.update(train_state.step - progress_bar.n)

        if train_state.step % config.log_every == 0:
            metric_values = {k: float(v) for k, v in metrics.items()}
            metric_values["lr"] = float(lr)
            metric_values["temperature"] = float(config.temperature)
            metric_values["loss"] = float(loss)
            wandb.log(
                {f"train/{k}": v for k, v in metric_values.items()},
                step=train_state.step,
            )

        if config.viz_every > 0 and train_state.step % config.viz_every == 0:
            model_for_viz = eqx.combine(train_state.params, train_state.static)
            viz_img = render_nearest_neighbors(
                model_for_viz,
                batch_to_np(batch),
                metadata,
                top_k=config.viz_neighbors,
            )
            wandb.log(
                {"train/nearest_neighbors": wandb.Image(viz_img)},
                step=train_state.step,
            )

    wandb.finish()


if __name__ == "__main__":
    tyro.cli(main)()
