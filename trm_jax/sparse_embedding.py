from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class CastedSparseEmbedding(eqx.Module):
    embedding: eqx.nn.Embedding
    cast_to: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        key,
        cast_to=jnp.float32,
    ):
        self.embedding = eqx.nn.Embedding(num_embeddings, embedding_dim, key=key)
        self.cast_to = cast_to

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        flat_inputs = inputs.reshape(-1)
        embedded = jax.vmap(self.embedding)(flat_inputs)
        output_dim = embedded.shape[-1]
        return embedded.reshape(*inputs.shape, output_dim).astype(self.cast_to)
