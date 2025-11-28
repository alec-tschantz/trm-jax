import math
from typing import Dict
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from trm.utils import trunc_normal
from trm.nn import (
    Block,
    CosSin,
    Embedding,
    Linear,
    RotaryEmbedding,
    SparseEmbedding,
    Transformer,
)


@dataclass
class ModelConfig:
    batch_size: int
    seq_len: int
    task_emb_ndim: int
    num_task_identifiers: int
    vocab_size: int

    y_cycles: int
    z_cycles: int
    num_layers: int

    hidden_size: int
    expansion: float
    num_heads: int

    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str
    task_emb_len: int

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0


class State(eqx.Module):
    y: jnp.ndarray
    z: jnp.ndarray


class Carry(eqx.Module):
    states: State
    steps: jnp.ndarray
    halted: jnp.ndarray
    data: Dict[str, jnp.ndarray]


class Model(eqx.Module):
    config: ModelConfig = eqx.field(static=True)
    forward_dtype: jnp.dtype = eqx.field(static=True)
    embed_scale: float = eqx.field(static=True)
    task_emb_len: int = eqx.field(static=True)
    H_init: jnp.ndarray = eqx.field(static=True)
    L_init: jnp.ndarray = eqx.field(static=True)

    embed_tokens: Embedding
    lm_head: Linear
    q_head: Linear
    task_embed: SparseEmbedding
    rotary_emb: RotaryEmbedding
    network: Transformer

    def __init__(self, config: ModelConfig, *, key):
        self.config = config

        dtype = getattr(jnp, config.forward_dtype)
        self.forward_dtype = dtype
        self.embed_scale = math.sqrt(config.hidden_size)
        self.task_emb_len = config.task_emb_len
        embed_init_std = 1.0 / self.embed_scale

        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            init_std=embed_init_std,
            key=k1,
            cast_to=dtype,
        )

        self.task_embed = SparseEmbedding(
            config.num_task_identifiers,
            config.task_emb_ndim,
            init_std=0.0,
            cast_to=dtype,
            key=k4,
        )

        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=config.seq_len + config.task_emb_len,
            base=config.rope_theta,
        )

        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False, key=k2)

        q_head = Linear(config.hidden_size, 1, bias=True, key=k3)
        q_head = eqx.tree_at(lambda m: m.weight, q_head, jnp.zeros_like(q_head.weight))
        bias_val = jnp.full_like(q_head.bias, -5.0)
        q_head = eqx.tree_at(lambda m: m.bias, q_head, bias_val)
        self.q_head = q_head

        layer_keys = jax.random.split(k5, config.num_layers)
        self.network = Transformer(
            tuple(
                Block(
                    hidden_size=config.hidden_size,
                    expansion=config.expansion,
                    num_heads=config.num_heads,
                    rms_norm_eps=config.rms_norm_eps,
                    key=kk,
                )
                for kk in layer_keys
            )
        )

        self.H_init = trunc_normal(k6, (config.hidden_size,), std=1.0).astype(dtype)
        self.L_init = trunc_normal(k7, (config.hidden_size,), std=1.0).astype(dtype)

    def __call__(
        self,
        carry: Carry,
        rng: jnp.ndarray,
        training: bool,
        *,
        record: bool = False,
    ):
        batch = carry.data
        cos_sin = self.rotary_emb()
        x_embed = self.embed_inputs(batch["inputs"], batch["puzzle_identifiers"])

        y, z, iters_aux = self.run_iters(
            x_embed,
            carry.states.y,
            carry.states.z,
            cos_sin,
            record=record,
        )

        y_logits = self.lm_head(y).astype(jnp.float32)[:, self.task_emb_len :, :]
        q_logits = self.q_head(y[:, 0]).astype(jnp.float32).squeeze(-1)

        q_rng, rng = jax.random.split(rng)
        new_steps, halted = self.update_halt_state(
            carry.steps, q_logits, q_rng, training
        )

        carry = Carry(
            states=State(
                y=jax.lax.stop_gradient(y),
                z=jax.lax.stop_gradient(z),
            ),
            steps=new_steps,
            halted=halted,
            data=carry.data,
        )
        aux = {**iters_aux, "y_logits": y_logits, "q_logits": q_logits}
        return carry, aux

    def run_iters(self, x_embed, y, z, cos_sin, *, record=False):
        x_embed = x_embed.astype(self.forward_dtype)
        y0, z0 = y.astype(self.forward_dtype), z.astype(self.forward_dtype)
        is_last = jnp.arange(self.config.y_cycles) == (self.config.y_cycles - 1)

        def body(carry, last):
            y_curr, z_curr = carry
            z_next, z_hist = self.run_z_iters(
                x_embed, y_curr, z_curr, cos_sin, record=record
            )
            y_next = self.network(y_curr, z_next, cos_sin).astype(self.forward_dtype)
            y_c = jax.lax.cond(last, lambda x: x, jax.lax.stop_gradient, y_next)
            z_c = jax.lax.cond(last, lambda x: x, jax.lax.stop_gradient, z_next)
            hist = (jax.lax.stop_gradient(y_next), z_hist) if record else None
            return (y_c, z_c), hist

        (y_final, z_final), hist = jax.lax.scan(body, (y0, z0), is_last)
        y_hist, z_hist = hist if record else (None, None)

        return y_final, z_final, {"y_hist": y_hist, "z_hist": z_hist}

    def run_z_iters(
        self,
        x_embed: jnp.ndarray,
        y_state: jnp.ndarray,
        z_state: jnp.ndarray,
        cos_sin: CosSin,
        *,
        record: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def body(z_curr, _):
            context = (y_state + x_embed).astype(self.forward_dtype)
            z_next = self.network(z_curr, context, cos_sin).astype(self.forward_dtype)
            hist = jax.lax.stop_gradient(z_next) if record else None
            return z_next, hist

        z_final, z_hist = jax.lax.scan(
            body, z_state, xs=None, length=self.config.z_cycles
        )
        return z_final, z_hist

    def embed_inputs(self, inputs: jnp.ndarray, task_ids: jnp.ndarray) -> jnp.ndarray:
        tok = self.embed_tokens(inputs.astype(jnp.int32))

        task = self.task_embed(task_ids)
        need = self.task_emb_len * self.config.hidden_size - task.shape[-1]
        if need > 0:
            task = jnp.pad(task, ((0, 0), (0, need)))

        task = task.reshape(-1, self.task_emb_len, self.config.hidden_size)
        emb = jnp.concatenate([task, tok], axis=1)

        emb = (emb * self.embed_scale).astype(self.forward_dtype)
        return emb

    def update_halt_state(
        self,
        steps: jnp.ndarray,
        q_logits: jnp.ndarray,
        rng: jnp.ndarray,
        training: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        new_steps = steps + 1
        halted = new_steps >= self.config.halt_max_steps

        if training and self.config.halt_max_steps > 1:
            halted = jnp.logical_or(halted, q_logits > 0)

            rng, explore_rng = jax.random.split(rng)
            rng, min_rng = jax.random.split(rng)
            r = jax.random.uniform(explore_rng, q_logits.shape)
            sample = r < self.config.halt_exploration_prob

            sampled_min = jax.random.randint(
                min_rng,
                new_steps.shape,
                minval=2,
                maxval=self.config.halt_max_steps + 1,
                dtype=jnp.int32,
            )
            min_steps = jnp.where(sample, sampled_min, jnp.zeros_like(sampled_min))
            halted = jnp.logical_and(halted, new_steps >= min_steps)

        return new_steps, halted

    def reset_states(self, reset: jnp.ndarray, states: State) -> State:
        flag = reset.reshape((-1, 1, 1))
        y0 = self.H_init[None, None, :]
        z0 = self.L_init[None, None, :]
        y = jnp.where(flag, y0, states.y)
        z = jnp.where(flag, z0, states.z)
        return State(y=y, z=z)

    def initial_carry(self, batch: Dict[str, jnp.ndarray]) -> Carry:
        bs = batch["inputs"].shape[0]
        return Carry(
            states=self.initial_state(bs),
            steps=jnp.zeros((bs,), dtype=jnp.int32),
            halted=jnp.ones((bs,), dtype=jnp.bool_),
            data={k: jnp.zeros_like(v) for k, v in batch.items()},
        )

    def initial_state(self, batch_size: int) -> State:
        zeros = jnp.zeros(
            (
                batch_size,
                self.config.seq_len + self.task_emb_len,
                self.config.hidden_size,
            ),
            dtype=self.forward_dtype,
        )
        return State(y=zeros, z=zeros)
