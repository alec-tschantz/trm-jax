from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


class AdamAtan2State(NamedTuple):
    count: jnp.ndarray
    exp_avg: Any
    exp_avg_sq: Any


class SparseSignSGDState(NamedTuple):
    pass


def adam_atan2(
    *,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
    a: float = 1.27,
    b: float = 1.0,
) -> optax.GradientTransformation:
    beta1 = jnp.asarray(beta1)
    beta2 = jnp.asarray(beta2)
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    weight_decay = jnp.asarray(weight_decay)

    def init_fn(params):
        zeros = jax.tree.map(jnp.zeros_like, params)
        return AdamAtan2State(
            count=jnp.zeros([], dtype=jnp.int32),
            exp_avg=zeros,
            exp_avg_sq=zeros,
        )

    def update_fn(grads, state, params):
        if params is None:
            raise ValueError("Parameters must be provided to adam_atan2 update.")

        count = state.count + jnp.array(1, dtype=state.count.dtype)

        def update_moment(moment, grad, beta):
            if grad is None:
                return moment
            return beta * moment + (1.0 - beta) * grad

        exp_avg = jax.tree.map(
            lambda m, g: update_moment(m, g, beta1), state.exp_avg, grads
        )
        exp_avg_sq = jax.tree.map(
            lambda v, g: update_moment(v, jnp.square(g) if g is not None else g, beta2),
            state.exp_avg_sq,
            grads,
        )

        bias_c1 = 1.0 - beta1**count
        bias_c2 = 1.0 - beta2**count

        def compute_update(m, v, g, p):
            if (m is None) or (v is None) or (p is None):
                return None
            m_hat = m / bias_c1
            v_hat = v / bias_c2
            v_hat = jnp.maximum(v_hat, 0.0)
            denom = jnp.sqrt(v_hat * (b * b) + 1e-12)
            atan_val = jnp.arctan2(m_hat, denom)
            return -(a * atan_val + weight_decay * p)

        updates = jax.tree.map(compute_update, exp_avg, exp_avg_sq, grads, params)
        new_state = AdamAtan2State(count=count, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def sparse_sign_sgd(*, weight_decay: float = 0.0) -> optax.GradientTransformation:
    weight_decay = jnp.asarray(weight_decay)

    def init_fn(params):
        return SparseSignSGDState()

    def update_fn(grads, state, params):
        def compute_update(g, p):
            if (g is None) or (p is None):
                return None
            finite_grad = jnp.where(jnp.isfinite(g), g, 0.0)
            mask = jnp.any(finite_grad != 0.0, axis=-1, keepdims=True)
            mask = mask.astype(p.dtype)
            signed = jnp.sign(finite_grad)
            return -mask * (signed + weight_decay * p)

        updates = jax.tree.map(compute_update, grads, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)
