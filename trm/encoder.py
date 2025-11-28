import jax
import jax.numpy as jnp
import equinox as eqx

from trm.nn import SparseEmbedding


class Encoder(eqx.Module):
    task_embed: SparseEmbedding
    task_emb_len: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    forward_dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_task_identifiers: int,
        task_emb_ndim: int,
        task_emb_len: int,
        hidden_size: int,
        forward_dtype: str,
        key: jnp.ndarray,
    ):
        self.task_emb_len = task_emb_len
        self.hidden_size = hidden_size
        self.forward_dtype = getattr(jnp, forward_dtype)
        self.task_embed = SparseEmbedding(
            num_task_identifiers,
            task_emb_ndim,
            init_std=0.0,
            cast_to=self.forward_dtype,
            key=key,
        )

    def __call__(self, inputs: jnp.ndarray, task_ids: jnp.ndarray) -> jnp.ndarray:
        del inputs  
        task = self.task_embed(task_ids)
        need = self.task_emb_len * self.hidden_size - task.shape[-1]
        if need > 0:
            task = jnp.pad(task, ((0, 0), (0, need)))

        task = task.reshape(-1, self.task_emb_len, self.hidden_size)
        return task.astype(self.forward_dtype)
