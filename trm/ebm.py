from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx

from trm.model import Model
from trm.nn import Linear


@dataclass
class EnergyConfig:
    lr: float = 0.1
    noise_std: float = 1e-2


class EnergyModel(Model):
    energy_cfg: EnergyConfig = eqx.field(static=True)
    energy_head: Linear

    def __init__(self, model_cfg: dict, energy_cfg: EnergyConfig, *, key):
        base_key, head_key = jax.random.split(key)
        self.energy_cfg = energy_cfg

        super().__init__(model_cfg, key=base_key)
        self.energy_head = Linear(self.config.hidden_size, 1, bias=True, key=head_key)

    def update_state(
        self,
        state: jnp.ndarray,
        context: jnp.ndarray,
        cos_sin,
        key: jnp.ndarray,
    ):
        def energy_batch(s):
            h = self.network(s, context, cos_sin).astype(self.forward_dtype)
            energy_map = self.energy_head(h).astype(jnp.float32)
            return jnp.sum(jnp.mean(energy_map, axis=(1, 2)))

        grad = jax.grad(energy_batch)(state)
        step = jnp.asarray(self.energy_cfg.lr, dtype=grad.dtype)
        noise = (
            jax.random.normal(key, state.shape).astype(grad.dtype)
            * self.energy_cfg.noise_std
        )
        updated = (state - step * grad + noise).astype(self.forward_dtype)
        return updated
