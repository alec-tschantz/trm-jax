from typing import Dict, NamedTuple, Sequence, Tuple

import equinox as eqx
import jax
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


class _LossStats(NamedTuple):
    mask: jnp.ndarray
    loss_counts: jnp.ndarray
    loss_divisor: jnp.ndarray
    preds: jnp.ndarray
    is_correct: jnp.ndarray
    seq_is_correct: jnp.ndarray
    lm_loss: jnp.ndarray
    q_halt_loss: jnp.ndarray
    total_loss: jnp.ndarray


def _gather_loss_stats(
    logits: jnp.ndarray, labels: jnp.ndarray, q_halt_logits: jnp.ndarray
) -> _LossStats:
    mask = labels != IGNORE_LABEL_ID
    loss_counts = jnp.sum(mask, axis=-1)
    loss_divisor = jnp.maximum(loss_counts, 1)[..., None]
    preds = jnp.argmax(logits, axis=-1)
    is_correct = jnp.logical_and(preds == labels, mask)
    seq_is_correct = jnp.sum(is_correct, axis=-1) == loss_counts
    lm_loss = jnp.sum(
        stablemax_cross_entropy(logits, labels, valid_mask=mask) / loss_divisor
    )
    targets = seq_is_correct.astype(jnp.float32)
    q_halt_loss = jnp.sum(
        optax.sigmoid_binary_cross_entropy(logits=q_halt_logits, labels=targets)
    )
    total_loss = lm_loss + 0.5 * q_halt_loss
    return _LossStats(
        mask=mask,
        loss_counts=loss_counts,
        loss_divisor=loss_divisor,
        preds=preds,
        is_correct=is_correct,
        seq_is_correct=seq_is_correct,
        lm_loss=lm_loss,
        q_halt_loss=q_halt_loss,
        total_loss=total_loss,
    )


def adapt_task_latent(
    model: Model,
    carry: Carry,
    rng: jnp.ndarray,
) -> Carry:
    if model.config.task_adaptation_lr == 0.0:
        return carry
    steps = int(getattr(model.config, "task_adaptation_steps", 1))
    if steps <= 0:
        return carry
    z_task = carry.inner_carry.z_task
    if z_task.size == 0:
        return carry

    labels = carry.current_data.get("labels")
    if labels is None or labels.size == 0:
        return carry

    step = model.config.task_adaptation_lr
    step_rngs = jax.random.split(rng, steps)

    def single_step(task_latent: jnp.ndarray, step_rng: jnp.ndarray):
        def loss_fn(current_task: jnp.ndarray):
            patched = eqx.tree_at(
                lambda c: c.inner_carry.z_task,
                carry,
                current_task,
            )
            _, outputs = model(
                patched,
                rng=step_rng,
                training=False,
            )
            stats = _gather_loss_stats(
                outputs["logits"],
                labels,
                outputs["q_halt_logits"],
            )
            return stats.total_loss

        _, grads = jax.value_and_grad(loss_fn)(task_latent)
        updated = jax.lax.stop_gradient(task_latent - step * grads)
        return updated, None

    new_task, _ = jax.lax.scan(single_step, z_task, step_rngs)
    return eqx.tree_at(lambda c: c.inner_carry.z_task, carry, new_task)


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
    adapt_rng, forward_rng = jax.random.split(rng)
    carry = adapt_task_latent(model, carry, adapt_rng)
    new_carry, outputs = model(
        carry,
        rng=forward_rng,
        training=training,
    )
    labels = new_carry.current_data["labels"]
    logits = outputs["logits"]
    q_halt_logits = outputs["q_halt_logits"]
    stats = _gather_loss_stats(logits, labels, q_halt_logits)
    preds = stats.preds
    outputs = dict(outputs)
    outputs["preds"] = preds

    valid_metrics = jnp.logical_and(new_carry.halted, stats.loss_counts > 0)

    def reduce_metric(values):
        return jnp.sum(jnp.where(valid_metrics, values, 0.0))

    metrics: Dict[str, jnp.ndarray] = {
        "count": jnp.sum(valid_metrics.astype(jnp.float32)),
        "accuracy": reduce_metric(
            jnp.sum(stats.is_correct.astype(jnp.float32) / stats.loss_divisor, axis=-1)
        ),
        "exact_accuracy": reduce_metric(stats.seq_is_correct.astype(jnp.float32)),
        "q_halt_accuracy": reduce_metric(
            (jnp.where(q_halt_logits >= 0, 1, 0) == stats.seq_is_correct).astype(
                jnp.float32
            )
        ),
        "steps": reduce_metric(new_carry.steps.astype(jnp.float32)),
    }

    metrics.update(
        {
            "lm_loss": stats.lm_loss,
            "q_halt_loss": stats.q_halt_loss,
        }
    )

    detached_outputs = {k: outputs[k] for k in return_keys if k in outputs}
    total_loss = stats.total_loss
    all_finish = jnp.all(new_carry.halted)
    return new_carry, total_loss, metrics, detached_outputs, all_finish
