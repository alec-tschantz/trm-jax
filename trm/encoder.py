import equinox as eqx
import jax.numpy as jnp

from trm.nn import SparseEmbedding


class TaskEncoder(eqx.Module):
    task_embed: SparseEmbedding
    task_emb_len: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        num_task_identifiers: int,
        task_emb_ndim: int,
        *,
        task_emb_len: int,
        hidden_size: int,
        cast_to: jnp.dtype,
        key,
    ):
        self.task_embed = SparseEmbedding(
            num_task_identifiers,
            task_emb_ndim,
            init_std=0.0,
            cast_to=cast_to,
            key=key,
        )
        self.task_emb_len = task_emb_len
        self.hidden_size = hidden_size

    def __call__(self, task_ids: jnp.ndarray) -> jnp.ndarray:
        task = self.task_embed(task_ids)
        need = self.task_emb_len * self.hidden_size - task.shape[-1]
        if need > 0:
            task = jnp.pad(task, ((0, 0), (0, need)))
        return task.reshape(-1, self.task_emb_len, self.hidden_size)
