import math
from typing import Any, Callable, Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import wandb
from torch.utils.data import DataLoader

from dataset import PuzzleDatasetMetadata
from trm.model import Model

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
LOGIT_LENS_PANEL_SIZE = 256
LOGIT_LENS_TITLE_HEIGHT = 24
LOGIT_LENS_BANNER_HEIGHT = 28
LOGIT_LENS_TITLE_BG = (242, 242, 242)
LOGIT_LENS_TEXT_COLOR = (0, 0, 0)

LOGIT_LENS_FONT = ImageFont.load_default()


def evaluate_model(
    model: Model,
    dataloader: DataLoader,
    *,
    batch_converter: Callable[[Dict[str, Any]], Dict[str, jnp.ndarray]],
    num_outer_steps: int,
    rng: jnp.ndarray | None = None,
) -> Dict[str, float]:
    eval_rng = rng if rng is not None else jax.random.PRNGKey(0)
    totals = {
        "loss_sum": 0.0,
        "token_correct": 0.0,
        "token_count": 0.0,
        "seq_correct": 0.0,
        "seq_count": 0.0,
    }

    def sample_step_size(key: jnp.ndarray) -> jnp.ndarray:
        cfg = model.config
        return jax.random.uniform(
            key,
            (),
            minval=cfg.energy_step_size_min,
            maxval=cfg.energy_step_size_max,
            dtype=jnp.float32,
        )

    for _, batch, _ in dataloader:
        batch_jnp = batch_converter(batch)
        eval_rng, step_rng = jax.random.split(eval_rng)
        step_rng, loss_rng = jax.random.split(step_rng)
        step_size = sample_step_size(step_rng)
        loss, metrics = model.loss(
            batch_jnp["inputs"],
            batch_jnp["labels"],
            rng=loss_rng,
            num_outer_steps=num_outer_steps,
            step_size=step_size,
            training=False,
        )
        metrics = jtu.tree_map(lambda x: float(x), metrics)
        token_count = metrics["token_count"]
        seq_count = metrics["seq_count"]
        totals["loss_sum"] += metrics["loss"] * token_count
        totals["token_correct"] += metrics["token_correct"]
        totals["token_count"] += token_count
        totals["seq_correct"] += metrics["seq_correct"]
        totals["seq_count"] += seq_count

    results = {}
    if totals["token_count"] > 0:
        results["test/loss"] = totals["loss_sum"] / totals["token_count"]
        results["test/token_accuracy"] = totals["token_correct"] / totals["token_count"]
    if totals["seq_count"] > 0:
        results["test/seq_accuracy"] = totals["seq_correct"] / totals["seq_count"]
    results["test/outer_steps"] = float(num_outer_steps)
    return results


def _build_logit_lens_palette() -> np.ndarray:
    return np.asarray(LOGIT_LENS_COLORS, dtype=np.uint8)


def _tokens_to_color_grid(tokens: np.ndarray, palette: np.ndarray, grid_size: int):
    flat = tokens.reshape(grid_size, grid_size)
    indices = np.mod(flat, palette.shape[0])
    colors = palette[np.clip(indices, 0, palette.shape[0] - 1)]
    return colors.astype(np.uint8)


def _logits_to_tokens(logits: jnp.ndarray) -> jnp.ndarray:
    logits = logits.astype(jnp.float32)
    return jnp.argmax(logits, axis=-1)


def _render_panel(
    tokens: np.ndarray, palette: np.ndarray, grid_size: int, title: str
) -> Image.Image:
    colors = _tokens_to_color_grid(tokens, palette, grid_size)
    img = Image.fromarray(colors, mode="RGB")
    img = img.resize((LOGIT_LENS_PANEL_SIZE, LOGIT_LENS_PANEL_SIZE), Image.NEAREST)

    canvas = Image.new(
        "RGB",
        (LOGIT_LENS_PANEL_SIZE, LOGIT_LENS_PANEL_SIZE + LOGIT_LENS_TITLE_HEIGHT),
        LOGIT_LENS_TITLE_BG,
    )
    canvas.paste(img, (0, LOGIT_LENS_TITLE_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (LOGIT_LENS_PANEL_SIZE // 2, LOGIT_LENS_TITLE_HEIGHT // 2),
        title,
        font=LOGIT_LENS_FONT,
        fill=LOGIT_LENS_TEXT_COLOR,
        anchor="mm",
    )
    return canvas


def _pair_panels(
    left: Image.Image,
    right: Image.Image,
    banner: Optional[str],
) -> Image.Image:
    row_width = left.width + right.width
    row_height = left.height
    total_height = LOGIT_LENS_BANNER_HEIGHT + row_height
    frame = Image.new("RGB", (row_width, total_height), LOGIT_LENS_TITLE_BG)

    frame.paste(left, (0, LOGIT_LENS_BANNER_HEIGHT))
    frame.paste(right, (left.width, LOGIT_LENS_BANNER_HEIGHT))

    if banner:
        draw = ImageDraw.Draw(frame)
        draw.text(
            (row_width // 2, LOGIT_LENS_BANNER_HEIGHT // 2),
            banner,
            font=LOGIT_LENS_FONT,
            fill=LOGIT_LENS_TEXT_COLOR,
            anchor="mm",
        )
    return frame


def _pil_to_chw(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    return np.transpose(arr, (2, 0, 1))


def _safe_tokens(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.where(arr < 0, 0, arr)
    return arr


def _render_logit_lens_frames(
    y_tokens: np.ndarray,
    target_tokens: np.ndarray,
    palette: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    frames: List[np.ndarray] = []
    if y_tokens.ndim < 4:
        return np.zeros((0, 3, LOGIT_LENS_PANEL_SIZE, LOGIT_LENS_PANEL_SIZE), dtype=np.uint8)

    num_steps = y_tokens.shape[0]
    num_z = y_tokens.shape[1]
    num_y = y_tokens.shape[2]

    target_panel = _render_panel(
        _safe_tokens(target_tokens), palette, grid_size, "Target"
    )

    for step_idx in range(num_steps):
        for z_idx in range(num_z):
            for y_idx in range(num_y):
                pred_panel = _render_panel(
                    _safe_tokens(y_tokens[step_idx, z_idx, y_idx]),
                    palette,
                    grid_size,
                    f"z={z_idx}, y={y_idx}",
                )
                frame = _pair_panels(
                    pred_panel,
                    target_panel,
                    banner=f"iter={step_idx}, z={z_idx}, y={y_idx}",
                )
                frames.append(_pil_to_chw(frame))

    if not frames:
        return np.zeros((0, 3, LOGIT_LENS_PANEL_SIZE, LOGIT_LENS_PANEL_SIZE), dtype=np.uint8)

    return np.stack(frames, axis=0).astype(np.uint8)


def evaluate_logit_lens(
    model: Model,
    batch: Dict[str, jnp.ndarray],
    metadata: PuzzleDatasetMetadata,
    *,
    step: int,
    rng: jnp.ndarray,
):
    if batch["inputs"].shape[0] == 0:
        return

    rng, step_rng = jax.random.split(rng)
    rng, lens_rng = jax.random.split(rng)
    step_size = jax.random.uniform(
        step_rng,
        (),
        minval=model.config.energy_step_size_min,
        maxval=model.config.energy_step_size_max,
        dtype=jnp.float32,
    )
    y_hidden = model.logit_lens_states(
        batch,
        rng=lens_rng,
        step_size=step_size,
    )

    palette = _build_logit_lens_palette()
    grid_size = int(round(math.sqrt(metadata.seq_len)))

    y_tokens = _logits_to_tokens(y_hidden)

    y_tokens_np = np.asarray(jax.device_get(y_tokens))
    if y_tokens_np.shape[2] == 0:
        return

    sample_y_tokens = y_tokens_np[:, :, 0]
    sample_y_tokens = np.expand_dims(sample_y_tokens, axis=0)
    sample_labels = np.asarray(jax.device_get(batch["labels"][0]))

    frames = _render_logit_lens_frames(
        sample_y_tokens,
        sample_labels,
        palette,
        grid_size,
    )
    if frames.size == 0:
        return

    wandb.log(
        {"logit_lens/video": wandb.Video(frames, format="mp4", fps=4)},
        step=step,
    )
