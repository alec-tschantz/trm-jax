import math
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import wandb
from torch.utils.data import DataLoader

from dataset import PuzzleDatasetMetadata
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


class EvalState(NamedTuple):
    accuracy: jnp.ndarray
    exact_accuracy: jnp.ndarray
    q_halt_accuracy: jnp.ndarray
    count: jnp.ndarray
    lm_loss: jnp.ndarray


def _zero_eval_state(dtype=jnp.float32) -> EvalState:
    zero = jnp.array(0.0, dtype=dtype)
    return EvalState(zero, zero, zero, zero, zero)


@eqx.filter_jit
def _eval_rollout(
    model: Model,
    carry: Carry,
    rng: jnp.ndarray,
    max_steps: int,
) -> EvalState:
    max_steps = jnp.asarray(max_steps, dtype=jnp.int32)

    def cond_fn(state):
        _, _, _, steps, finished = state
        return jnp.logical_and(steps < max_steps, jnp.logical_not(finished))

    def body_fn(state):
        carry, rng, agg, steps, finished = state
        rng, step_rng = jax.random.split(rng)
        carry, _loss, metrics, _, all_finish = act_loss(
            model,
            carry,
            rng=step_rng,
            training=False,
        )
        new_agg = EvalState(
            accuracy=agg.accuracy + metrics["accuracy"],
            exact_accuracy=agg.exact_accuracy + metrics["exact_accuracy"],
            q_halt_accuracy=agg.q_halt_accuracy + metrics["q_halt_accuracy"],
            count=agg.count + metrics["count"],
            lm_loss=metrics["lm_loss"],
        )
        new_finished = jnp.logical_or(finished, all_finish)
        return carry, rng, new_agg, steps + 1, new_finished

    init_state = (
        carry,
        rng,
        _zero_eval_state(),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
    )

    _, _, aggregates, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return aggregates


def evaluate_model(
    model: Model,
    dataloader: DataLoader,
    *,
    batch_converter: Callable[[Dict[str, Any]], Dict[str, jnp.ndarray]],
    prepare_carry_fn: Callable[[Model, Carry, Dict[str, jnp.ndarray]], Carry],
    rng: jnp.ndarray | None = None,
) -> Dict[str, float]:
    eval_rng = rng if rng is not None else jax.random.PRNGKey(0)
    totals = {
        "lm_loss": 0.0,
        "accuracy": 0.0,
        "exact_accuracy": 0.0,
        "q_halt_accuracy": 0.0,
        "count": 0.0,
        "loss_denominator": 0.0,
    }
    max_steps = getattr(model.config, "halt_max_steps", 1)

    for _, batch, global_batch_size in dataloader:
        batch_jnp = batch_converter(batch)
        carry = model.initial_carry(batch_jnp)
        carry = prepare_carry_fn(model, carry, batch_jnp)
        eval_rng, batch_rng = jax.random.split(eval_rng)
        aggregates = _eval_rollout(
            model,
            carry,
            batch_rng,
            max_steps,
        )
        aggregates = jtu.tree_map(lambda x: float(x), aggregates)
        totals["accuracy"] += aggregates.accuracy
        totals["exact_accuracy"] += aggregates.exact_accuracy
        totals["q_halt_accuracy"] += aggregates.q_halt_accuracy
        totals["count"] += aggregates.count
        totals["lm_loss"] += aggregates.lm_loss
        totals["loss_denominator"] += float(global_batch_size)

    results = {}
    if totals["loss_denominator"] > 0:
        results["test/lm_loss"] = totals["lm_loss"] / totals["loss_denominator"]
    if totals["count"] > 0:
        denom = max(totals["count"], 1e-8)
        results["test/accuracy"] = totals["accuracy"] / denom
        results["test/exact_accuracy"] = totals["exact_accuracy"] / denom
        results["test/q_halt_accuracy"] = totals["q_halt_accuracy"] / denom
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


@eqx.filter_jit
def _rollout_state_histories(
    model: Model,
    carry: Carry,
    rng: jnp.ndarray,
) -> tuple[Carry, jnp.ndarray]:
    num_steps = max(int(model.config.halt_max_steps), 1)

    def step_fn(state, _):
        cur_carry, cur_rng = state
        cur_rng, step_rng = jax.random.split(cur_rng)
        new_carry, outputs = model(
            cur_carry,
            rng=step_rng,
            training=False,
            record=True,
        )
        return (new_carry, cur_rng), outputs["y_states"]

    (final_carry, _), y_histories = jax.lax.scan(
        step_fn,
        (carry, rng),
        xs=None,
        length=num_steps,
    )
    return final_carry, y_histories


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

    carry = model.initial_carry(batch)
    carry = filter_carry_fn(model, carry, batch)
    _, y_hidden = _rollout_state_histories(model, carry, rng)

    palette = _build_logit_lens_palette()
    grid_size = int(round(math.sqrt(metadata.seq_len)))

    y_tokens = _logits_to_tokens(y_hidden)

    y_tokens_np = np.asarray(jax.device_get(y_tokens))
    if y_tokens_np.shape[3] == 0:
        return

    sample_y_tokens = np.take(y_tokens_np, 0, axis=3)

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
