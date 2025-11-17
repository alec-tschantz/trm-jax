from typing import Dict, Sequence, Tuple

import jax.nn as jnn
import jax.numpy as jnp
import optax

from trm.model import Carry, Model

IGNORE_LABEL_ID = -100


def stablemax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    ignore_index: int = IGNORE_LABEL_ID,
    valid_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    logprobs = jnn.log_softmax(logits.astype(jnp.float32), axis=-1)
    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = jnp.where(valid_mask, labels, 0)
    prediction_logprobs = jnp.take_along_axis(
        logprobs, transformed_labels[..., None], axis=-1
    ).squeeze(-1)
    return -jnp.where(valid_mask, prediction_logprobs, 0.0)


def act_loss(
    model: Model,
    carry: Carry,
    rng: jnp.ndarray,
    return_keys: Sequence[str],
    *,
    training: bool,
) -> Tuple[
    Carry, jnp.ndarray, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray
]:
    new_carry, outputs = model(
        carry,
        rng=rng,
        training=training,
    )
    labels = new_carry.current_data["labels"]
    logits = outputs["logits"]
    q_halt_logits = outputs["q_halt_logits"]
    preds = jnp.argmax(logits, axis=-1)
    outputs = dict(outputs)
    outputs["preds"] = preds

    mask = labels != IGNORE_LABEL_ID
    loss_counts = jnp.sum(mask, axis=-1)
    loss_divisor = jnp.maximum(loss_counts, 1)[..., None]
    is_correct = jnp.logical_and(preds == labels, mask)
    seq_is_correct = jnp.sum(is_correct, axis=-1) == loss_counts
    valid_metrics = jnp.logical_and(new_carry.halted, loss_counts > 0)

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
        "steps": reduce_metric(new_carry.steps.astype(jnp.float32)),
    }

    lm_loss = jnp.sum(
        stablemax_cross_entropy(logits, labels, valid_mask=mask) / loss_divisor
    )
    targets = seq_is_correct.astype(logits.dtype)
    q_halt_loss = jnp.sum(
        optax.sigmoid_binary_cross_entropy(logits=q_halt_logits, labels=targets)
    )
    metrics.update(
        {
            "lm_loss": lm_loss,
            "q_halt_loss": q_halt_loss,
        }
    )

    detached_outputs = {k: outputs[k] for k in return_keys if k in outputs}
    total_loss = lm_loss + 0.5 * q_halt_loss
    all_finish = jnp.all(new_carry.halted)
    return new_carry, total_loss, metrics, detached_outputs, all_finish
