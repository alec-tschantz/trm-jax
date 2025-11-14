import equinox as eqx
import jax.numpy as jnp

from trm.utils import trunc_normal


class SparseEmbedding(eqx.Module):
    weight: jnp.ndarray
    cast_to: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_std: float,
        key,
        cast_to=jnp.float32,
    ):
        if init_std == 0.0:
            weight = jnp.zeros((num_embeddings, embedding_dim), dtype=jnp.float32)
        else:
            weight = trunc_normal(key, (num_embeddings, embedding_dim), std=init_std)
        self.weight = weight.astype(jnp.float32)
        self.cast_to = cast_to

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        idx = inputs.astype(jnp.int32)
        embedded = self.weight[idx]
        return embedded.astype(self.cast_to)
