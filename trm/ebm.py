from dataclasses import dataclass

import jax
import jax.numpy as jnp

from trm.model import Model
from trm.nn import Linear


@dataclass
class EnergyConfig:
    lr: float = 0.1


class EnergyModel(Model):
    def __init__(self, model_cfg: dict, energy_cfg: EnergyConfig, *, key):
        base_key, head_key = jax.random.split(key)
        super().__init__(model_cfg, key=base_key)
        self.energy_cfg = energy_cfg
        self.energy_head = Linear(
            self.config.hidden_size,
            1,
            bias=True,
            key=head_key,
        )

    def update_state(self, state, context, cos_sin):
        def energy_batch(s):
            h = self.network(s, context, cos_sin).astype(self.forward_dtype)
            energy_map = self.energy_head(h).astype(jnp.float32)
            return jnp.sum(jnp.mean(energy_map, axis=(1, 2)))

        grad = jax.grad(energy_batch)(state)
        step = jnp.asarray(self.energy_cfg.lr, dtype=grad.dtype)
        updated = (state - step * grad).astype(self.forward_dtype)
        return updated
