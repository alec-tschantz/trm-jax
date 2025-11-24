from typing import Dict, Tuple

import jax.nn as jnn
import jax.numpy as jnp
import optax

from trm.model import Carry, Model

IGNORE_LABEL_ID = -100


def act_loss(
    model: Model,
    carry: Carry,
    rng: jnp.ndarray,
    *,
    training: bool,
    task_emb: jnp.ndarray,
) -> Tuple[
    Carry, jnp.ndarray, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray
]:
    new_carry, outputs = model(
        carry,
        rng=rng,
        training=training,
        record=False,
        task_emb=task_emb,
    )

    labels = new_carry.data["labels"]
    y_logits = outputs["y_logits"]
    q_logits = outputs["q_logits"]
    preds = jnp.argmax(y_logits, axis=-1)

    metrics, mask, seq_is_correct, loss_divisor = compute_act_metrics(
        labels,
        preds,
        q_logits,
        new_carry,
    )
    q_targets = seq_is_correct.astype(y_logits.dtype)

    lm_loss = jnp.sum(
        stablemax_cross_entropy(y_logits, labels, valid_mask=mask) / loss_divisor
    )
    q_halt_loss = jnp.sum(
        optax.sigmoid_binary_cross_entropy(logits=q_logits, labels=q_targets)
    )
    total_loss = lm_loss + 0.5 * q_halt_loss

    metrics = {**metrics, "lm_loss": lm_loss, "q_halt_loss": q_halt_loss}
    return new_carry, total_loss, metrics, jnp.all(new_carry.halted)


def compute_act_metrics(
    labels: jnp.ndarray,
    preds: jnp.ndarray,
    q_halt_logits: jnp.ndarray,
    carry: Carry,
) -> tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mask = labels != IGNORE_LABEL_ID
    loss_counts = jnp.sum(mask, axis=-1)
    loss_divisor = jnp.maximum(loss_counts, 1)[..., None]
    is_correct = jnp.logical_and(preds == labels, mask)
    seq_is_correct = jnp.sum(is_correct, axis=-1) == loss_counts
    valid_metrics = jnp.logical_and(carry.halted, loss_counts > 0)

    def reduce_metric(values):
        return jnp.sum(jnp.where(valid_metrics, values, 0.0))

    metrics: Dict[str, jnp.ndarray] = {
        "count": jnp.sum(valid_metrics.astype(jnp.float32)),
        "accuracy": reduce_metric(
            jnp.sum(is_correct.astype(jnp.float32) / loss_divisor, axis=-1)
        ),
        "exact_accuracy": reduce_metric(seq_is_correct.astype(jnp.float32)),
        "q_halt_accuracy": reduce_metric(
            (jnp.where(q_halt_logits >= 0, 1, 0) == seq_is_correct).astype(jnp.float32)
        ),
        "steps": reduce_metric(carry.steps.astype(jnp.float32)),
    }
    return metrics, mask, seq_is_correct, loss_divisor


def stablemax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    ignore_index: int = IGNORE_LABEL_ID,
    valid_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    logprobs = jnn.log_softmax(logits.astype(jnp.float32), axis=-1)
    if valid_mask is None:
        valid_mask = labels != ignore_index
    labels = jnp.where(valid_mask, labels, 0)
    pred_logprobs = jnp.take_along_axis(logprobs, labels[..., None], axis=-1)
    return -jnp.where(valid_mask, pred_logprobs.squeeze(-1), 0.0)


def info_nce_loss(
    z1: jnp.ndarray,
    z2: jnp.ndarray,
    *,
    temperature: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Symmetric InfoNCE loss for two batches of embeddings."""
    z1_norm = _l2_normalize(z1)
    z2_norm = _l2_normalize(z2)

    logits = jnp.matmul(z1_norm, jnp.transpose(z2_norm, (1, 0))) / temperature
    labels = jnp.arange(logits.shape[0], dtype=jnp.int32)

    loss12 = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss21 = optax.softmax_cross_entropy_with_integer_labels(
        jnp.transpose(logits, (1, 0)), labels
    )
    loss = 0.5 * (jnp.mean(loss12) + jnp.mean(loss21))

    diag = jnp.sum(z1_norm * z2_norm, axis=-1)
    off_diag = jnp.where(jnp.eye(logits.shape[0], dtype=bool), -1e9, logits)

    metrics: Dict[str, jnp.ndarray] = {
        "info_nce_loss": loss,
        "pos_sim": jnp.mean(diag),
        "logits_std": jnp.std(logits),
        "top1_i2t": jnp.mean(
            (jnp.argmax(logits, axis=1) == labels).astype(jnp.float32)
        ),
        "top1_t2i": jnp.mean(
            (jnp.argmax(logits, axis=0) == labels).astype(jnp.float32)
        ),
        "max_neg_logit": jnp.max(off_diag),
        "mean_norm": jnp.mean(
            jnp.concatenate(
                [jnp.linalg.norm(z1, axis=-1), jnp.linalg.norm(z2, axis=-1)]
            )
        ),
    }
    return loss, metrics


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    denom = jnp.linalg.norm(x, axis=-1, keepdims=True)
    denom = jnp.maximum(denom, eps)
    return x / denom
