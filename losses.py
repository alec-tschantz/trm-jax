# losses.py
# JAX/Optax-compatible rewrite of your PyTorch ACTLossHead + losses.
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp


# -----------------------
# Stablemax utilities
# -----------------------
def s(x: jnp.ndarray, epsilon: float = 1e-30) -> jnp.ndarray:
    # piecewise definition; matches PyTorch version semantics
    return jnp.where(x < 0.0, 1.0 / (1.0 - x + epsilon), x + 1.0)


def log_stablemax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    sx = s(x)
    denom = jnp.sum(sx, axis=axis, keepdims=True)
    return jnp.log(sx) - jnp.log(denom)


def stablemax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Returns per-token loss (same shape as `labels`), like the PyTorch version.
    """
    logprobs = log_stablemax(logits.astype(jnp.float32), axis=-1)  # [B,S,V]
    labels_clamped = jnp.where(valid_mask, labels, 0)
    picked = jnp.take_along_axis(
        logprobs, labels_clamped[..., None].astype(jnp.int32), axis=-1
    ).squeeze(-1)
    return -jnp.where(valid_mask, picked, 0.0)


def softmax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Per-token standard softmax CE (no reduction), aligned with PyTorch call.
    """
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    labels_z = jnp.where(valid_mask, labels, 0)
    picked = jnp.take_along_axis(logp, labels_z[..., None], axis=-1).squeeze(-1)
    return -jnp.where(valid_mask, picked, 0.0)


# -----------------------
# BCE-with-logits (sum reduction)
# -----------------------
def bce_with_logits_sum(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Numerically-stable BCE with logits; reduction='sum' to match the PyTorch code.
    """
    x = logits.astype(jnp.float32)
    t = targets.astype(jnp.float32)
    # relu(x) - x*t + log(1 + exp(-|x|))
    return jnp.sum(jnp.maximum(x, 0.0) - x * t + jnp.log1p(jnp.exp(-jnp.abs(x))))


# -----------------------
# ACT loss head (functional)
# -----------------------
def act_loss(
    model,  # Equinox module with .initial_carry and __call__(carry, batch, key=..., is_training=True)
    *,
    batch: Dict[str, jnp.ndarray],
    key: jax.random.PRNGKey,
    loss_type: str = "stablemax_cross_entropy",
) -> Tuple[
    Any, jnp.ndarray, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray
]:
    """
    Functional equivalent of the PyTorch ACTLossHead.forward.

    Returns:
      new_carry,
      total_loss (scalar),
      metrics (dict of scalars),
      detached_outputs (model outputs with stop_gradient applied),
      all_halted (bool array reduced to scalar via .all()).
    """
    # Forward
    carry = model.initial_carry(batch)
    new_carry, outputs = model(carry, batch, key=key, is_training=True)
    labels = new_carry.current_data["labels"]
    output_mask = new_carry.current_data["output_mask"].astype(jnp.bool_)

    # Predictions (for metrics)
    preds = jnp.argmax(outputs["logits"], axis=-1)

    # Mask & per-sequence divisor
    loss_counts = output_mask.sum(axis=-1).astype(jnp.int32)  # [B]
    loss_divisor = jnp.maximum(loss_counts, 1)[:, None].astype(jnp.float32)  # [B,1]

    # Token-level correctness and exact sequence correctness
    is_correct = (preds == labels) & output_mask  # [B,S]
    seq_is_correct = is_correct.sum(axis=-1) == loss_counts  # [B], bool

    # Metrics: only count sequences with at least one valid token and have halted
    valid_metrics = new_carry.halted & (loss_counts > 0)
    metrics: Dict[str, jnp.ndarray] = {}

    # Language loss
    if loss_type == "stablemax_cross_entropy":
        token_losses = stablemax_cross_entropy(
            outputs["logits"], labels, valid_mask=output_mask
        )
    elif loss_type == "softmax_cross_entropy":
        token_losses = softmax_cross_entropy(
            outputs["logits"], labels, valid_mask=output_mask
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    lm_loss = jnp.sum(token_losses / loss_divisor)  # sum over batch (per-seq averaged)
    q_halt_logits = jnp.clip(outputs["q_halt_logits"], -30.0, 30.0)
    q_halt_loss = bce_with_logits_sum(q_halt_logits, seq_is_correct)

    total_loss = lm_loss + 0.5 * q_halt_loss

    # Metrics
    # accuracy over valid tokens, averaged per-seq then summed for valid_metrics
    per_seq_acc = (
        is_correct.astype(jnp.float32) / loss_divisor
    ).sum(axis=-1)
    metrics["count"] = valid_metrics.sum()
    metrics["accuracy"] = jnp.where(valid_metrics, per_seq_acc, 0.0).sum()
    metrics["exact_accuracy"] = (valid_metrics & seq_is_correct).sum()
    metrics["q_halt_accuracy"] = (
        valid_metrics & ((q_halt_logits >= 0.0) == seq_is_correct)
    ).sum()
    metrics["steps"] = jnp.where(valid_metrics, new_carry.steps, 0).sum()

    metrics["lm_loss"] = lm_loss
    metrics["q_halt_loss"] = q_halt_loss

    # Detached outputs: stop gradients
    detached_outputs = {k: jax.lax.stop_gradient(v) for k, v in outputs.items()}

    all_halted = new_carry.halted.all()
    return new_carry, total_loss, metrics, detached_outputs, all_halted
