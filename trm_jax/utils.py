from __future__ import annotations

import jax
import jax.numpy as jnp


def trunc_normal(
    key: jnp.ndarray, shape, std: float = 1.0, lower: float = -2.0, upper: float = 2.0
) -> jnp.ndarray:
    lower_scaled = lower / std
    upper_scaled = upper / std
    samples = jax.random.truncated_normal(key, lower_scaled, upper_scaled, shape)
    return samples * std
