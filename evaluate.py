import math
from typing import Dict, List, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import wandb

from dataset import DatasetMetadata
from trm.losses import act_loss
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
NN_PANEL_SIZE = 160
NN_TITLE_HEIGHT = 20
NN_ROW_BANNER_HEIGHT = 26
NN_ROW_BG = (245, 245, 245)


def render_nearest_neighbors(
    encoder,
    batch_np: Dict[str, np.ndarray],
    metadata: DatasetMetadata,
    *,
    top_k: int,
) -> Image.Image:
    flat_inputs, flat_labels, embeddings = _encode_examples_np(encoder, batch_np)
    if embeddings.shape[0] <= 1:
        return Image.new("RGB", (NN_PANEL_SIZE * 2, NN_PANEL_SIZE), NN_ROW_BG)

    norms = np.linalg.norm(embeddings, axis=-1)
    norms = np.maximum(norms, 1e-8)
    query = embeddings[0]
    sims = (embeddings @ query) / (norms * max(np.linalg.norm(query), 1e-8))
    order = np.argsort(-sims)
    neighbors = [idx for idx in order if idx != 0][:top_k]
    selected = [0] + neighbors

    palette = _build_logit_lens_palette()
    grid_size = int(round(math.sqrt(metadata.seq_len)))

    rows = []
    for rank, idx in enumerate(selected):
        header = "query" if rank == 0 else f"nn {rank}  sim={sims[idx]:.3f}"
        row = _render_nn_pair_row(
            flat_inputs[idx],
            flat_labels[idx],
            header,
            palette,
            grid_size,
        )
        rows.append(row)

    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + 8 * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), NN_ROW_BG)
    y = 0
    for i, row in enumerate(rows):
        canvas.paste(row, (0, y))
        y += row.height + (8 if i + 1 < len(rows) else 0)
    return canvas


def _encode_examples_np(
    encoder, batch_np: Dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_len = batch_np["inputs_1"].shape[-1]
    flat_inputs = np.concatenate(
        [batch_np["inputs_1"], batch_np["inputs_2"]], axis=0
    ).reshape(-1, seq_len)
    flat_labels = np.concatenate(
        [batch_np["labels_1"], batch_np["labels_2"]], axis=0
    ).reshape(-1, seq_len)

    embeddings = encoder.encode_examples(
        jnp.asarray(flat_inputs), jnp.asarray(flat_labels)
    )
    embeddings = np.asarray(jax.device_get(embeddings))
    return flat_inputs, flat_labels, embeddings


def _render_nn_pair_row(
    inputs: np.ndarray,
    labels: np.ndarray,
    header: str,
    palette: np.ndarray,
    grid_size: int,
) -> Image.Image:
    input_panel = _render_nn_panel(inputs, palette, grid_size, "Input")
    label_panel = _render_nn_panel(labels, palette, grid_size, "Output")

    width = input_panel.width + label_panel.width
    height = NN_ROW_BANNER_HEIGHT + input_panel.height
    row = Image.new("RGB", (width, height), NN_ROW_BG)
    row.paste(input_panel, (0, NN_ROW_BANNER_HEIGHT))
    row.paste(label_panel, (input_panel.width, NN_ROW_BANNER_HEIGHT))

    draw = ImageDraw.Draw(row)
    draw.text(
        (width // 2, NN_ROW_BANNER_HEIGHT // 2),
        header,
        font=LOGIT_LENS_FONT,
        fill=(0, 0, 0),
        anchor="mm",
    )
    return row


def _render_nn_panel(
    tokens: np.ndarray, palette: np.ndarray, grid_size: int, title: str
) -> Image.Image:
    safe = np.where(tokens < 0, 0, tokens)
    flat = safe[: grid_size * grid_size]
    if flat.size < grid_size * grid_size:
        flat = np.pad(flat, (0, grid_size * grid_size - flat.size), constant_values=0)

    colors = palette[np.mod(flat.reshape(grid_size, grid_size), palette.shape[0])]
    img = Image.fromarray(colors.astype(np.uint8), mode="RGB")
    img = img.resize((NN_PANEL_SIZE, NN_PANEL_SIZE), Image.NEAREST)

    panel = Image.new(
        "RGB", (NN_PANEL_SIZE, NN_PANEL_SIZE + NN_TITLE_HEIGHT), (255, 255, 255)
    )
    panel.paste(img, (0, NN_TITLE_HEIGHT))
    draw = ImageDraw.Draw(panel)
    draw.text(
        (NN_PANEL_SIZE // 2, NN_TITLE_HEIGHT // 2),
        title,
        font=LOGIT_LENS_FONT,
        fill=(0, 0, 0),
        anchor="mm",
    )
    return panel


class EvalState(NamedTuple):
    accuracy: jnp.ndarray
    exact_accuracy: jnp.ndarray
    q_halt_accuracy: jnp.ndarray
    count: jnp.ndarray
    lm_loss: jnp.ndarray


def evaluate_logit_lens(
    model: Model,
    batch: Dict[str, jnp.ndarray],
    metadata: DatasetMetadata,
    *,
    step: int,
    rng: jnp.ndarray,
    task_emb: jnp.ndarray | None = None,
):
    single = {
        k: v[:1]
        for k, v in batch.items()
        if k in ("inputs", "labels", "puzzle_identifiers")
    }
    task_emb_single = task_emb[:1] if task_emb is not None else None
    carry = model.initial_carry(single)
    _, y_hidden, z_hidden = _rollout_state_histories(
        model, carry, rng, task_emb_single
    )

    palette = _build_logit_lens_palette()
    grid_size = int(round(math.sqrt(metadata.seq_len)))

    y_tokens = _hidden_to_tokens(y_hidden, model.lm_head, model.task_emb_len)
    z_tokens = _hidden_to_tokens(z_hidden, model.lm_head, model.task_emb_len)

    y_tokens_np = np.asarray(jax.device_get(y_tokens))[:, :, 0]  
    z_tokens_np = np.asarray(jax.device_get(z_tokens))[:, :, :, 0]  

    sample_inputs = np.asarray(jax.device_get(single["inputs"][0]))
    sample_labels = np.asarray(jax.device_get(single["labels"][0]))

    frames = _render_logit_lens_frames(
        y_tokens_np,
        z_tokens_np,
        sample_inputs,
        sample_labels,
        palette,
        grid_size,
    )

    wandb.log(
        {"logit_lens/video": wandb.Video(frames, format="mp4", fps=4)},
        step=step,
    )


@eqx.filter_jit
def _eval_rollout(
    model: Model,
    carry: Carry,
    max_steps: int,
    rng: jnp.ndarray,
    task_emb: jnp.ndarray,
) -> EvalState:
    max_steps = jnp.asarray(max_steps, dtype=jnp.int32)

    def cond_fn(state):
        _, _, steps, finished, _ = state
        return jnp.logical_and(steps < max_steps, jnp.logical_not(finished))

    def body_fn(state):
        carry, agg, steps, finished, rng = state
        rng, step_rng = jax.random.split(rng)
        carry, _loss, metrics, all_finish = act_loss(
            model, carry, rng=step_rng, training=False, task_emb=task_emb
        )
        new_agg = EvalState(
            accuracy=agg.accuracy + metrics["accuracy"],
            exact_accuracy=agg.exact_accuracy + metrics["exact_accuracy"],
            q_halt_accuracy=agg.q_halt_accuracy + metrics["q_halt_accuracy"],
            count=agg.count + metrics["count"],
            lm_loss=metrics["lm_loss"],
        )
        new_finished = jnp.logical_or(finished, all_finish)
        return carry, new_agg, steps + 1, new_finished, rng

    init_state = (
        carry,
        _zero_eval_state(),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        rng,
    )

    _, aggregates, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return aggregates


@eqx.filter_jit
def _rollout_state_histories(
    model: Model,
    carry: Carry,
    rng: jnp.ndarray,
    task_emb: jnp.ndarray | None = None,
) -> tuple[Carry, jnp.ndarray, jnp.ndarray]:
    num_steps = int(model.config.halt_max_steps)

    def step_fn(state, rng_step):
        cur_carry = state
        new_carry, outputs = model(
            cur_carry,
            rng=rng_step,
            training=False,
            record=True,
            task_emb=task_emb,
        )
        return new_carry, (
            outputs["y_hist"],
            outputs["z_hist"],
        )

    rngs = jax.random.split(rng, num_steps)
    final_carry, (y_histories, z_histories) = jax.lax.scan(step_fn, carry, xs=rngs)
    return final_carry, y_histories, z_histories


def _render_logit_lens_frames(
    y_tokens: np.ndarray,
    z_tokens: np.ndarray,
    sample_inputs: np.ndarray,
    sample_outputs: np.ndarray,
    palette: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    frames: List[np.ndarray] = []

    num_iters = y_tokens.shape[0]
    y_cycles = y_tokens.shape[1]
    z_cycles = z_tokens.shape[2] if z_tokens.ndim >= 3 else 0

    input_panel = _render_panel(
        _safe_tokens(sample_inputs), palette, grid_size, "Input"
    )
    output_tokens = _safe_tokens(sample_outputs)
    output_panel = _render_panel(output_tokens, palette, grid_size, "Output")
    blank_tokens = np.zeros_like(sample_inputs)

    for iter_idx in range(num_iters):
        for y_idx in range(y_cycles):
            y_panel = _render_panel(
                _safe_tokens(y_tokens[iter_idx, y_idx]),
                palette,
                grid_size,
                f"y={y_idx}",
            )
            inner_cycles = max(z_cycles, 1)

            for z_idx in range(inner_cycles):
                if z_cycles == 0:
                    z_panel = _render_panel(
                        blank_tokens,
                        palette,
                        grid_size,
                        "z",
                    )
                else:
                    z_panel = _render_panel(
                        _safe_tokens(z_tokens[iter_idx, y_idx, z_idx]),
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


def _zero_eval_state(dtype=jnp.float32) -> EvalState:
    zero = jnp.array(0.0, dtype=dtype)
    return EvalState(zero, zero, zero, zero, zero)


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
