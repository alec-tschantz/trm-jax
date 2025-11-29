from typing import Any, Callable, Dict, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from trm.losses import act_loss
from trm.model import Carry, Model


class EvalState(NamedTuple):
    accuracy: jnp.ndarray
    exact_accuracy: jnp.ndarray
    q_halt_accuracy: jnp.ndarray
    count: jnp.ndarray
    lm_loss: jnp.ndarray
    steps: jnp.ndarray


def evaluate_model(
    model: Model,
    dataloader: Any,
    *,
    batch_converter: Callable[[Dict[str, Any]], Dict[str, jnp.ndarray]],
    filter_carry_fn: Callable[[Model, Carry, Dict[str, jnp.ndarray]], Carry],
    rng: jnp.ndarray,
) -> Dict[str, float]:
    totals = {
        "lm_loss": 0.0,
        "accuracy": 0.0,
        "exact_accuracy": 0.0,
        "q_halt_accuracy": 0.0,
        "steps": 0.0,
        "count": 0.0,
    }
    max_steps = model.config.halt_max_steps

    for _, batch, _ in dataloader:
        batch_jnp = batch_converter(batch)
        carry = model.initial_carry(batch_jnp)
        carry = filter_carry_fn(model, carry, batch_jnp)

        rng, rollout_rng = jax.random.split(rng)
        aggregates = _eval_rollout(model, carry, max_steps, rollout_rng)
        aggregates = jtu.tree_map(lambda x: float(x), aggregates)
        totals["accuracy"] += aggregates.accuracy
        totals["exact_accuracy"] += aggregates.exact_accuracy
        totals["q_halt_accuracy"] += aggregates.q_halt_accuracy
        totals["count"] += aggregates.count
        totals["lm_loss"] += aggregates.lm_loss
        totals["steps"] += aggregates.steps

    results = {}
    if totals["count"] > 0:
        denom = max(totals["count"], 1e-8)
        results["test/lm_loss"] = totals["lm_loss"] / denom
        results["test/accuracy"] = totals["accuracy"] / denom
        results["test/exact_accuracy"] = totals["exact_accuracy"] / denom
        results["test/q_halt_accuracy"] = totals["q_halt_accuracy"] / denom
        results["test/steps"] = totals["steps"] / denom
    return results


@eqx.filter_jit
def _eval_rollout(
    model: Model, carry: Carry, max_steps: int, rng: jnp.ndarray
) -> EvalState:
    max_steps = jnp.asarray(max_steps, dtype=jnp.int32)

    def cond_fn(state):
        _, _, steps, finished, _ = state
        return jnp.logical_and(steps < max_steps, jnp.logical_not(finished))

    def body_fn(state):
        carry, agg, steps, finished, rng = state
        rng, step_rng = jax.random.split(rng)
        warmed_carry = model.warmup_carry(carry)
        carry, _loss, metrics, all_finish = act_loss(
            model, warmed_carry, rng=step_rng, training=False
        )
        new_agg = EvalState(
            accuracy=agg.accuracy + metrics["accuracy"],
            exact_accuracy=agg.exact_accuracy + metrics["exact_accuracy"],
            q_halt_accuracy=agg.q_halt_accuracy + metrics["q_halt_accuracy"],
            count=agg.count + metrics["count"],
            lm_loss=agg.lm_loss + metrics["lm_loss"],
            steps=agg.steps + metrics["steps"],
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


def _zero_eval_state(dtype=jnp.float32) -> EvalState:
    zero = jnp.array(0.0, dtype=dtype)
    return EvalState(zero, zero, zero, zero, zero, zero)
