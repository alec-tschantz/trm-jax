import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import equinox as eqx


class EMAHelper:
    def __init__(self, mu: float):
        self.mu = mu
        self.shadow = None

    def register(self, params):
        self.shadow = jtu.tree_map(lambda x: x, params)

    def update(self, params):
        if self.shadow is None:
            self.register(params)
            return

        def _update(ema_value, new_value):
            if eqx.is_array(ema_value):
                return self.mu * ema_value + (1.0 - self.mu) * new_value
            return new_value

        self.shadow = jtu.tree_map(_update, self.shadow, params)

    def ema_copy(self):
        return self.shadow


def trunc_normal(
    key: jnp.ndarray, shape, std: float = 1.0, lower: float = -2.0, upper: float = 2.0
) -> jnp.ndarray:
    lower_scaled = lower / std
    upper_scaled = upper / std
    samples = jax.random.truncated_normal(key, lower_scaled, upper_scaled, shape)
    return samples * std

