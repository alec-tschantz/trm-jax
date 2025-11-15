import math
from typing import Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import wandb

from dataset import PuzzleDatasetMetadata
from trm.model import Carry, InnerCarry, Model


LOGIT_LENS_CHARSET = "# SGo"
LOGIT_LENS_COLOR_MAP = {
    "#": (25, 25, 25),
    " ": (242, 242, 242),
    "S": (52, 168, 83),
    "G": (234, 67, 53),
    "o": (30, 136, 229),
}
LOGIT_LENS_PANEL_SIZE = 256
LOGIT_LENS_TITLE_HEIGHT = 24
LOGIT_LENS_BANNER_HEIGHT = 28
LOGIT_LENS_TITLE_BG = (242, 242, 242)
LOGIT_LENS_TEXT_COLOR = (0, 0, 0)

LOGIT_LENS_FONT = ImageFont.load_default()


def _build_logit_lens_palette(vocab_size: int) -> np.ndarray:
    palette = np.zeros((vocab_size, 3), dtype=np.uint8)
    palette[0] = np.array([120, 120, 120], dtype=np.uint8)
    for idx, token in enumerate(LOGIT_LENS_CHARSET, start=1):
        palette[idx] = np.array(
            LOGIT_LENS_COLOR_MAP.get(token, (200, 200, 200)), dtype=np.uint8
        )
    return palette


def _tokens_to_color_grid(tokens: np.ndarray, palette: np.ndarray, grid_size: int):
    flat = tokens.reshape(grid_size, grid_size)
    colors = palette[np.clip(flat, 0, palette.shape[0] - 1)]
    return colors.astype(np.uint8)


def _hidden_to_tokens(
    hidden_states: jnp.ndarray, lm_head, puzzle_emb_len: int
) -> jnp.ndarray:
    logits = lm_head(hidden_states).astype(jnp.float32)
    logits = logits[..., puzzle_emb_len:, :]
    return jnp.argmax(logits, axis=-1)


def _render_panel(
    tokens: np.ndarray, palette: np.ndarray, grid_size: int, title: str
) -> Image.Image:
    colors = _tokens_to_color_grid(tokens, palette, grid_size)
    img = Image.fromarray(colors, mode="RGB")
    img = img.resize(
        (LOGIT_LENS_PANEL_SIZE, LOGIT_LENS_PANEL_SIZE), Image.NEAREST
    )

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


def _pil_to_chw(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    return np.transpose(arr, (2, 0, 1))


def _safe_tokens(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.where(arr < 0, 0, arr)
    return arr


def _render_logit_lens_frames(
    zh_tokens: np.ndarray,
    zl_tokens: np.ndarray,
    sample_inputs: np.ndarray,
    sample_labels: np.ndarray | None,
    palette: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    frames: List[np.ndarray] = []

    intro_right_tokens = (
        _safe_tokens(sample_labels)
        if sample_labels is not None
        else _safe_tokens(sample_inputs)
    )
    intro_left = _render_panel(
        _safe_tokens(sample_inputs), palette, grid_size, "Input"
    )
    intro_right = _render_panel(
        intro_right_tokens, palette, grid_size, "Label" if sample_labels is not None else "Input"
    )
    intro_frame = _frame_from_panels(intro_left, intro_right, banner=None)
    frames.append(_pil_to_chw(intro_frame))

    steps, h_cycles = zh_tokens.shape[:2]
    l_cycles = zl_tokens.shape[2] if zl_tokens.ndim >= 3 else 0

    for step_idx in range(steps):
        for h_idx in range(h_cycles):
            zh_panel = _render_panel(
                _safe_tokens(zh_tokens[step_idx, h_idx]), palette, grid_size, f"ZH={h_idx}"
            )
            for l_idx in range(l_cycles):
                zl_panel = _render_panel(
                    _safe_tokens(zl_tokens[step_idx, h_idx, l_idx]),
                    palette,
                    grid_size,
                    f"ZL={l_idx}",
                )
                combined = _frame_from_panels(zh_panel, zl_panel, banner=f"R={step_idx}")
                frames.append(_pil_to_chw(combined))

    return np.stack(frames, axis=0).astype(np.uint8)


def _inner_forward_with_states(
    model: Model, carry: InnerCarry, batch: Dict[str, jnp.ndarray]
) -> tuple[InnerCarry, jnp.ndarray, jnp.ndarray]:
    inner = model.inner
    cos_sin = inner.rotary_emb()
    inp = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    z_H, z_L = carry.z_H, carry.z_L

    z_H_traces = []
    z_L_traces = []

    for _ in range(inner.config.H_cycles):
        inj = (z_H + inp).astype(inner.forward_dtype)

        l_cycle_states = []
        for _ in range(inner.config.L_cycles):
            z_L = inner.L_level(z_L, inj, cos_sin).astype(inner.forward_dtype)
            l_cycle_states.append(jax.lax.stop_gradient(z_L))

        z_H = inner.L_level(z_H, z_L, cos_sin).astype(inner.forward_dtype)
        z_H = jax.lax.stop_gradient(z_H)
        z_L = jax.lax.stop_gradient(z_L)

        z_H_traces.append(z_H)
        z_L_traces.append(jnp.stack(l_cycle_states))

    new_carry = InnerCarry(z_H=z_H, z_L=z_L)
    return (
        new_carry,
        jnp.stack(z_H_traces),
        jnp.stack(z_L_traces),
    )


@eqx.filter_jit
def forward_with_logits(
    model: Model, carry: Carry, *, max_steps: int | None = None
) -> tuple[Carry, jnp.ndarray, jnp.ndarray]:
    steps_to_run = model.config.halt_max_steps if max_steps is None else int(max_steps)
    inner_carry = carry.inner_carry
    current_data = carry.current_data
    steps = carry.steps
    halted = carry.halted

    zh_states = []
    zl_states = []

    for _ in range(steps_to_run):
        inner_carry, zh_cycle, zl_cycle = _inner_forward_with_states(
            model, inner_carry, current_data
        )
        zh_states.append(zh_cycle)
        zl_states.append(zl_cycle)
        steps = steps + 1
        halted = jnp.logical_or(halted, steps >= model.config.halt_max_steps)

    new_carry = Carry(
        inner_carry=inner_carry,
        steps=steps,
        halted=halted,
        current_data=current_data,
    )
    return new_carry, jnp.stack(zh_states), jnp.stack(zl_states)


def evaluate_logit_lens(
    model: Model,
    batch: Dict[str, jnp.ndarray],
    metadata: PuzzleDatasetMetadata,
    prepare_carry_fn,
    *,
    step: int,
):
    if batch["inputs"].shape[0] == 0:
        return

    single = {
        k: v[:1]
        for k, v in batch.items()
        if k in ("inputs", "labels", "puzzle_identifiers")
    }
    carry = model.initial_carry(single)
    carry = prepare_carry_fn(model, carry, single)
    _, zh_hidden, zl_hidden = forward_with_logits(model, carry)

    palette = _build_logit_lens_palette(metadata.vocab_size)
    grid_size = int(round(math.sqrt(metadata.seq_len)))
    if grid_size * grid_size != metadata.seq_len:
        return

    zh_tokens = _hidden_to_tokens(
        zh_hidden, model.inner.lm_head, model.inner.puzzle_emb_len
    )
    zl_tokens = _hidden_to_tokens(
        zl_hidden, model.inner.lm_head, model.inner.puzzle_emb_len
    )

    zh_tokens_np = np.asarray(jax.device_get(zh_tokens))[:, :, 0, :]
    zl_tokens_np = np.asarray(jax.device_get(zl_tokens))[:, :, :, 0, :]
    sample_inputs = np.asarray(jax.device_get(single["inputs"][0]))
    sample_labels = (
        np.asarray(jax.device_get(single["labels"][0])) if "labels" in single else None
    )

    frames = _render_logit_lens_frames(
        zh_tokens_np,
        zl_tokens_np,
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
