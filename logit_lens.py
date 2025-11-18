import math
from typing import Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import wandb

from dataset import PuzzleDatasetMetadata
from trm.model import Carry, Model


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


def _build_logit_lens_palette() -> np.ndarray:
    return np.asarray(LOGIT_LENS_COLORS, dtype=np.uint8)


def _tokens_to_color_grid(tokens: np.ndarray, palette: np.ndarray, grid_size: int):
    flat = tokens.reshape(grid_size, grid_size)
    indices = np.mod(flat, palette.shape[0])
    colors = palette[np.clip(indices, 0, palette.shape[0] - 1)]
    return colors.astype(np.uint8)


def _hidden_to_tokens(
    hidden_states: jnp.ndarray, lm_head, task_emb_len: int
) -> jnp.ndarray:
    logits = lm_head(hidden_states).astype(jnp.float32)
    logits = logits[..., task_emb_len:, :]
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


def _frame_from_panels(
    left: Image.Image, right: Image.Image, banner: Optional[str]
) -> Image.Image:
    width = left.width + right.width
    height = LOGIT_LENS_BANNER_HEIGHT + max(left.height, right.height)
    frame = Image.new("RGB", (width, height), LOGIT_LENS_TITLE_BG)
    frame.paste(left, (0, LOGIT_LENS_BANNER_HEIGHT))
    frame.paste(right, (left.width, LOGIT_LENS_BANNER_HEIGHT))

    if banner:
        draw = ImageDraw.Draw(frame)
        draw.text(
            (width // 2, LOGIT_LENS_BANNER_HEIGHT // 2),
            banner,
            font=LOGIT_LENS_FONT,
            fill=LOGIT_LENS_TEXT_COLOR,
            anchor="mm",
        )
    return frame


def _grid_from_panels(
    top_left: Image.Image,
    top_right: Image.Image,
    bottom_left: Image.Image,
    bottom_right: Image.Image,
    banner: Optional[str],
) -> Image.Image:
    row_width = top_left.width + top_right.width
    row_height = top_left.height
    total_height = LOGIT_LENS_BANNER_HEIGHT + row_height * 2
    frame = Image.new("RGB", (row_width, total_height), LOGIT_LENS_TITLE_BG)

    frame.paste(top_left, (0, LOGIT_LENS_BANNER_HEIGHT))
    frame.paste(top_right, (top_left.width, LOGIT_LENS_BANNER_HEIGHT))
    frame.paste(bottom_left, (0, LOGIT_LENS_BANNER_HEIGHT + row_height))
    frame.paste(
        bottom_right,
        (top_left.width, LOGIT_LENS_BANNER_HEIGHT + row_height),
    )

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
    z_tokens: np.ndarray,
    sample_inputs: np.ndarray,
    sample_outputs: np.ndarray | None,
    palette: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    frames: List[np.ndarray] = []

    num_iters = y_tokens.shape[0]
    z_cycles = z_tokens.shape[1] if z_tokens.ndim >= 2 else 0
    input_panel = _render_panel(
        _safe_tokens(sample_inputs), palette, grid_size, "Input"
    )
    output_tokens = (
        _safe_tokens(sample_outputs)
        if sample_outputs is not None
        else _safe_tokens(sample_inputs)
    )
    output_title = "Output" if sample_outputs is not None else "Input"
    output_panel = _render_panel(output_tokens, palette, grid_size, output_title)
    blank_tokens = np.zeros_like(sample_inputs)

    for iter_idx in range(num_iters):
        y_panel = _render_panel(
            _safe_tokens(y_tokens[iter_idx]),
            palette,
            grid_size,
            f"y={iter_idx}",
        )
        if z_cycles == 0:
            z_panel = _render_panel(blank_tokens, palette, grid_size, "z")
            frame = _grid_from_panels(
                y_panel,
                z_panel,
                input_panel,
                output_panel,
                banner=f"iter={iter_idx}",
            )
            frames.append(_pil_to_chw(frame))
            continue

        for z_idx in range(z_cycles):
            z_panel = _render_panel(
                _safe_tokens(z_tokens[iter_idx, z_idx]),
                palette,
                grid_size,
                f"z={z_idx}",
            )
            frame = _grid_from_panels(
                y_panel,
                z_panel,
                input_panel,
                output_panel,
                banner=f"iter={iter_idx}",
            )
            frames.append(_pil_to_chw(frame))

    return np.stack(frames, axis=0).astype(np.uint8)


@eqx.filter_jit
def _collect_state_histories(
    model: Model,
    carry: Carry,
    rng: jnp.ndarray,
) -> tuple[Carry, jnp.ndarray, jnp.ndarray]:
    new_carry, outputs = model(
        carry,
        rng=rng,
        training=False,
        record=True,
    )
    return new_carry, outputs["y_states"], outputs["z_states"]


def evaluate_logit_lens(
    model: Model,
    batch: Dict[str, jnp.ndarray],
    metadata: PuzzleDatasetMetadata,
    filter_carry_fn,
    *,
    step: int,
    rng: jnp.ndarray,
):
    if batch["inputs"].shape[0] == 0:
        return

    single = {
        k: v[:1]
        for k, v in batch.items()
        if k in ("inputs", "labels", "puzzle_identifiers")
    }
    carry = model.initial_carry(single)
    carry = filter_carry_fn(model, carry, single)
    _, y_hidden, z_hidden = _collect_state_histories(model, carry, rng)

    palette = _build_logit_lens_palette()
    grid_size = int(round(math.sqrt(metadata.seq_len)))

    y_tokens = _hidden_to_tokens(y_hidden, model.lm_head, model.task_emb_len)
    z_tokens = _hidden_to_tokens(z_hidden, model.lm_head, model.task_emb_len)

    y_tokens_np = np.asarray(jax.device_get(y_tokens))[:, 0, :]
    z_tokens_np = np.asarray(jax.device_get(z_tokens))
    z_tokens_np = z_tokens_np[:, :, 0, :]
   
        
    sample_inputs = np.asarray(jax.device_get(single["inputs"][0]))
    sample_labels = (
        np.asarray(jax.device_get(single["labels"][0])) if "labels" in single else None
    )

    frames = _render_logit_lens_frames(
        y_tokens_np,
        z_tokens_np,
        sample_inputs,
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
