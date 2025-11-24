import math
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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from dataset.dataset import (
    DatasetConfig,
    DatasetMetadata,
    GroupDataset,
)
from trm.encoder import DEFAULT_IGNORE_LABEL_ID, Encoder, EncoderConfig, info_nce_loss
from trm.optim import cosine_warmup_schedule

LOGIT_LENS_COLORS = [
    (25, 25, 25),
    (242, 242, 242),
    (52, 168, 83),
    (234, 67, 53),
    (30, 136, 229),
    (249, 168, 37),
    (171, 71, 188),
    (0, 172, 193),
    (255, 109, 132),
    (123, 31, 162),
    (255, 214, 0),
    (0, 137, 123),
]

FONT = ImageFont.load_default()
PANEL_SIZE = 160
PANEL_TITLE_HEIGHT = 20
ROW_BANNER_HEIGHT = 26
ROW_BG = (245, 245, 245)


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


def render_nearest_neighbors(
    model: Encoder,
    batch_np: Dict[str, np.ndarray],
    metadata: DatasetMetadata,
    *,
    top_k: int,
) -> Image.Image:
    flat_inputs, flat_labels, embeddings = _encode_examples_np(model, batch_np)
    if embeddings.shape[0] <= 1:
        return Image.new("RGB", (PANEL_SIZE * 2, PANEL_SIZE), ROW_BG)

    norms = np.linalg.norm(embeddings, axis=-1)
    norms = np.maximum(norms, 1e-8)
    query = embeddings[0]
    query_norm = max(np.linalg.norm(query), 1e-8)
    sims = (embeddings @ query) / (norms * query_norm)
    order = np.argsort(-sims)
    neighbors = [idx for idx in order if idx != 0][:top_k]
    selected = [0] + neighbors

    palette = np.asarray(LOGIT_LENS_COLORS, dtype=np.uint8)
    grid_size = int(round(math.sqrt(metadata.seq_len)))

    rows = []
    for rank, idx in enumerate(selected):
        header = "query" if rank == 0 else f"nn {rank}  sim={sims[idx]:.3f}"
        row = _render_pair_row(
            flat_inputs[idx],
            flat_labels[idx],
            header,
            palette,
            grid_size,
        )
        rows.append(row)

    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + 8 * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), ROW_BG)
    y = 0
    for i, row in enumerate(rows):
        canvas.paste(row, (0, y))
        y += row.height + (8 if i + 1 < len(rows) else 0)
    return canvas


def _encode_examples_np(
    model: Encoder, batch_np: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_len = batch_np["inputs_1"].shape[-1]
    flat_inputs = np.concatenate(
        [batch_np["inputs_1"], batch_np["inputs_2"]], axis=0
    ).reshape(-1, seq_len)
    flat_labels = np.concatenate(
        [batch_np["labels_1"], batch_np["labels_2"]], axis=0
    ).reshape(-1, seq_len)

    embeddings = model.encode_examples(
        jnp.asarray(flat_inputs), jnp.asarray(flat_labels)
    )
    embeddings = np.asarray(jax.device_get(embeddings))
    return flat_inputs, flat_labels, embeddings


def _render_pair_row(
    inputs: np.ndarray,
    labels: np.ndarray,
    header: str,
    palette: np.ndarray,
    grid_size: int,
) -> Image.Image:
    input_panel = _render_panel(inputs, palette, grid_size, "Input")
    label_panel = _render_panel(labels, palette, grid_size, "Output")

    width = input_panel.width + label_panel.width
    height = ROW_BANNER_HEIGHT + input_panel.height
    row = Image.new("RGB", (width, height), ROW_BG)
    row.paste(input_panel, (0, ROW_BANNER_HEIGHT))
    row.paste(label_panel, (input_panel.width, ROW_BANNER_HEIGHT))

    draw = ImageDraw.Draw(row)
    draw.text(
        (width // 2, ROW_BANNER_HEIGHT // 2),
        header,
        font=FONT,
        fill=(0, 0, 0),
        anchor="mm",
    )
    return row


def _render_panel(
    tokens: np.ndarray, palette: np.ndarray, grid_size: int, title: str
) -> Image.Image:
    safe = np.where(tokens < 0, 0, tokens)
    flat = safe[: grid_size * grid_size]
    if flat.size < grid_size * grid_size:
        flat = np.pad(flat, (0, grid_size * grid_size - flat.size), constant_values=0)

    colors = palette[np.mod(flat.reshape(grid_size, grid_size), palette.shape[0])]
    img = Image.fromarray(colors.astype(np.uint8), mode="RGB")
    img = img.resize((PANEL_SIZE, PANEL_SIZE), Image.NEAREST)

    panel = Image.new(
        "RGB", (PANEL_SIZE, PANEL_SIZE + PANEL_TITLE_HEIGHT), (255, 255, 255)
    )
    panel.paste(img, (0, PANEL_TITLE_HEIGHT))
    draw = ImageDraw.Draw(panel)
    draw.text(
        (PANEL_SIZE // 2, PANEL_TITLE_HEIGHT // 2),
        title,
        font=FONT,
        fill=(0, 0, 0),
        anchor="mm",
    )
    return panel


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
