import colorsys
import math
from typing import Dict, List

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import wandb

from dataset import DatasetMetadata
from trm.model import Carry, InnerCarry, Model


LOGIT_LENS_PANEL_SIZE = 256
LOGIT_LENS_TITLE_HEIGHT = 24
LOGIT_LENS_COLUMN_LABEL_HEIGHT = 28
LOGIT_LENS_TITLE_BG = (242, 242, 242)
LOGIT_LENS_TEXT_COLOR = (0, 0, 0)
LOGIT_LENS_COLUMNS = 4
LOGIT_LENS_ROWS = 4

LOGIT_LENS_FONT = ImageFont.load_default()


def _build_logit_lens_palette(vocab_size: int) -> np.ndarray:
    if vocab_size <= 0:
        return np.zeros((1, 3), dtype=np.uint8)
    palette = np.zeros((vocab_size, 3), dtype=np.uint8)
    palette[0] = np.array([120, 120, 120], dtype=np.uint8)
    golden_ratio = 0.6180339887498949
    for idx in range(1, vocab_size):
        hue = (idx * golden_ratio) % 1.0
        sat = 0.4 + 0.5 * ((idx * 31) % 100) / 100.0
        val = 0.65 + 0.35 * ((idx * 17) % 100) / 100.0
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        palette[idx] = np.array([int(c * 255) for c in rgb], dtype=np.uint8)
    return palette


def _tokens_to_color_grid(tokens: np.ndarray, palette: np.ndarray, grid_size: int):
    flat = tokens.reshape(grid_size, grid_size)
    colors = palette[np.clip(flat, 0, palette.shape[0] - 1)]
    return colors.astype(np.uint8)


def _hidden_to_tokens(
    hidden_states: jnp.ndarray, lm_head, prefix_len: int
) -> jnp.ndarray:
    logits = lm_head(hidden_states).astype(jnp.float32)
    logits = logits[..., prefix_len:, :]
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


def _stack_column(panels: List[Image.Image], label: str) -> Image.Image:
    width = panels[0].width
    total_height = LOGIT_LENS_COLUMN_LABEL_HEIGHT + sum(panel.height for panel in panels)
    column = Image.new("RGB", (width, total_height), LOGIT_LENS_TITLE_BG)
    draw = ImageDraw.Draw(column)
    if label:
        draw.text(
            (width // 2, LOGIT_LENS_COLUMN_LABEL_HEIGHT // 2),
            label,
            font=LOGIT_LENS_FONT,
            fill=LOGIT_LENS_TEXT_COLOR,
            anchor="mm",
        )
    offset = LOGIT_LENS_COLUMN_LABEL_HEIGHT
    for panel in panels:
        column.paste(panel, (0, offset))
        offset += panel.height
    return column


def _combine_columns(columns: List[Image.Image]) -> Image.Image:
    width = sum(column.width for column in columns)
    height = max(column.height for column in columns)
    canvas = Image.new("RGB", (width, height), LOGIT_LENS_TITLE_BG)
    offset = 0
    for column in columns:
        canvas.paste(column, (offset, 0))
        offset += column.width
    return canvas


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
    actions: np.ndarray | None,
    palette: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    frames: List[np.ndarray] = []
    total_steps = zh_tokens.shape[0]
    label_tokens = (
        _safe_tokens(sample_labels)
        if sample_labels is not None
        else _safe_tokens(sample_inputs)
    )
    label_title = "Label" if sample_labels is not None else "Input"
    input_panel = _render_panel(_safe_tokens(sample_inputs), palette, grid_size, "Input")
    label_panel = _render_panel(label_tokens, palette, grid_size, label_title)

    def _format_action(actions_array: np.ndarray | None) -> str:
        if actions_array is None or actions_array.size == 0:
            return ""
        flat = actions_array.reshape(-1)
        if flat.size == 1:
            return f" A={int(flat[0])}"
        values = ",".join(str(int(v)) for v in flat.tolist())
        return f" A=[{values}]"

    action_suffix = _format_action(actions)

    blank_tokens = np.zeros_like(sample_inputs)
    blank_panel = _render_panel(blank_tokens, palette, grid_size, "")

    for start in range(0, total_steps, LOGIT_LENS_COLUMNS):
        columns: List[Image.Image] = []
        for offset in range(LOGIT_LENS_COLUMNS):
            step_idx = start + offset
            if step_idx < total_steps:
                zh_panel = _render_panel(
                    _safe_tokens(zh_tokens[step_idx, -1]),
                    palette,
                    grid_size,
                    "ZH",
                )
                zl_panel = _render_panel(
                    _safe_tokens(zl_tokens[step_idx, -1, -1]),
                    palette,
                    grid_size,
                    "ZL",
                )
                panels = [zh_panel, zl_panel, input_panel, label_panel]
                label = f"R={step_idx}{action_suffix}"
            else:
                panels = [blank_panel] * LOGIT_LENS_ROWS
                label = ""
            columns.append(_stack_column(panels, label))

        frame = _combine_columns(columns)
        frames.append(_pil_to_chw(frame))

    if not frames:
        return np.zeros((0, 3, 0, 0), dtype=np.uint8)

    return np.stack(frames, axis=0).astype(np.uint8)


def _inner_forward_with_states(
    model: Model,
    carry: InnerCarry,
    batch: Dict[str, jnp.ndarray],
) -> tuple[InnerCarry, jnp.ndarray, jnp.ndarray]:
    inner = model.inner
    cos_sin = inner.rotary_emb()
    inp = inner._input_embeddings(
        batch["inputs"], batch["puzzle_identifiers"], batch.get("actions")
    )
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
    model: Model,
    carry: Carry,
    *,
    max_steps: int | None = None,
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
            model,
            inner_carry,
            current_data,
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
    metadata: DatasetMetadata,
    prepare_carry_fn,
    *,
    step: int,
):
    if batch["inputs"].shape[0] == 0:
        return

    single = {
        k: v[:1]
        for k, v in batch.items()
        if k in ("inputs", "labels", "puzzle_identifiers", "actions")
    }
    carry = model.initial_carry(single)
    carry = prepare_carry_fn(model, carry, single)
    _, zh_hidden, zl_hidden = forward_with_logits(
        model,
        carry,
    )

    palette = _build_logit_lens_palette(metadata.vocab_size)
    grid_size = int(round(math.sqrt(metadata.seq_len)))
    if grid_size * grid_size != metadata.seq_len:
        return

    zh_tokens = _hidden_to_tokens(
        zh_hidden, model.inner.lm_head, model.inner.sequence_prefix_len
    )
    zl_tokens = _hidden_to_tokens(
        zl_hidden, model.inner.lm_head, model.inner.sequence_prefix_len
    )

    zh_tokens_np = np.asarray(jax.device_get(zh_tokens))[:, :, 0, :]
    zl_tokens_np = np.asarray(jax.device_get(zl_tokens))[:, :, :, 0, :]
    sample_inputs = np.asarray(jax.device_get(single["inputs"][0]))
    sample_labels = (
        np.asarray(jax.device_get(single["labels"][0])) if "labels" in single else None
    )
    sample_actions = (
        np.asarray(jax.device_get(single.get("actions")))[0]
        if "actions" in single
        else None
    )

    frames = _render_logit_lens_frames(
        zh_tokens_np,
        zl_tokens_np,
        sample_inputs,
        sample_labels,
        sample_actions,
        palette,
        grid_size,
    )
    if frames.size == 0:
        return

    wandb.log(
        {"logit_lens/video": wandb.Video(frames, format="mp4", fps=4)},
        step=step,
    )
