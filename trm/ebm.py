from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from trm.model import Model
from trm.nn import Linear


@dataclass
class EnergyConfig:
    lr: float = 5.0
    noise_scale: float = 3.0


class EnergyModel(Model):
    energy_cfg: EnergyConfig = eqx.field(static=True)
    energy_head: Linear

    def __init__(self, model_cfg: dict, energy_cfg: EnergyConfig, *, key):
        base_key, head_key = jax.random.split(key)
        super().__init__(model_cfg, key=base_key)

        self.energy_cfg = energy_cfg
        self.energy_head = Linear(self.config.hidden_size, 1, bias=True, key=head_key)

    def _energy(
        self,
        state: jnp.ndarray,
        context: jnp.ndarray,
        cos_sin,
    ) -> jnp.ndarray:

        ctx = jax.lax.stop_gradient(context)
        cs = jax.lax.stop_gradient(cos_sin)

        s_cast = state.astype(self.forward_dtype)
        h = self.network(s_cast, ctx, cs).astype(self.forward_dtype)

        energy_map = self.energy_head(h).astype(jnp.float32)
        return jnp.mean(energy_map)

    def update_state(
        self,
        state: jnp.ndarray,
        context: jnp.ndarray,
        cos_sin,
        key: jnp.ndarray,
    ) -> jnp.ndarray:

        state = state.astype(self.forward_dtype)
        lr = jnp.asarray(self.energy_cfg.lr, dtype=state.dtype)

        def energy_fn(s):
            return self._energy(s, context, cos_sin)

        grad_E = jax.grad(energy_fn)(state)

        noise = self.energy_cfg.noise_scale * jax.random.normal(
            key, state.shape, dtype=state.dtype
        )
        new_state = state - lr * grad_E + noise

        return new_state.astype(self.forward_dtype)
