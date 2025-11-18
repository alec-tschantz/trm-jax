import math
from dataclasses import dataclass
from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel

from trm.utils import trunc_normal
from trm.nn import (
    Attention,
    Embedding,
    Linear,
    CosSin,
    RotaryEmbedding,
    SparseEmbedding,
    SwiGLU,
    rms_norm,
)


class ModelConfig(BaseModel):
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

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"
    task_emb_len: int = 16


class State(eqx.Module):
    y: jnp.ndarray
    z: jnp.ndarray


class Carry(eqx.Module):
    states: State
    steps: jnp.ndarray
    halted: jnp.ndarray
    data: Dict[str, jnp.ndarray]


class Block(eqx.Module):
    self_attn: Attention
    mlp: SwiGLU
    norm_eps: float = eqx.field(static=True)

    def __init__(self, config: ModelConfig, *, key):
        k1, k2 = jax.random.split(key)
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
            key=k1,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
            key=k2,
        )
        self.norm_eps = config.rms_norm_eps

    def __call__(self, cos_sin: CosSin, h: jnp.ndarray) -> jnp.ndarray:
        dtype = h.dtype

        attn_out = self.self_attn(cos_sin, h)
        h2 = rms_norm(h + attn_out.astype(dtype), eps=self.norm_eps)

        mlp_out = self.mlp(h2)
        h3 = rms_norm(h2 + mlp_out.astype(dtype), eps=self.norm_eps)

        return h3


class Transformer(eqx.Module):
    layers: Tuple[Block, ...]

    def __call__(self, h: jnp.ndarray, x: jnp.ndarray, cos_sin: CosSin) -> jnp.ndarray:
        h = h + x
        for layer in self.layers:
            h = layer(cos_sin, h)
        return h


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

    def __init__(self, cfg: dict, *, key):
        config = ModelConfig(**cfg)
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

        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False, key=k2)

        q_head = Linear(config.hidden_size, 1, bias=True, key=k3)
        q_head = eqx.tree_at(lambda m: m.weight, q_head, jnp.zeros_like(q_head.weight))
        bias_val = jnp.full_like(q_head.bias, -5.0)
        q_head = eqx.tree_at(lambda m: m.bias, q_head, bias_val)
        self.q_head = q_head

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

        layer_keys = jax.random.split(k5, config.num_layers)
        self.network = Transformer(tuple(Block(config, key=kk) for kk in layer_keys))

        self.H_init = trunc_normal(k6, (config.hidden_size,), std=1.0).astype(dtype)
        self.L_init = trunc_normal(k7, (config.hidden_size,), std=1.0).astype(dtype)

    def __call__(
        self,
        carry: Carry,
        rng,
        training: bool,
        *,
        record: bool = False,
    ):
        batch = carry.data
        cos_sin = self.rotary_emb()
        embeddings = self.embed(batch["inputs"], batch["puzzle_identifiers"])

        state_rng, rng = jax.random.split(rng)

        y, z, y_hist, z_hist = self.inference(
            carry.states.y,
            carry.states.z,
            embeddings,
            cos_sin,
            record=record,
            key=state_rng,
        )

        logits = self.lm_head(y).astype(jnp.float32)[:, self.task_emb_len :, :]
        qh = self.q_head(y[:, 0]).astype(jnp.float32).squeeze(-1)
        new_steps, halted, rng = self._update_halt_state(carry.steps, qh, rng, training)

        return Carry(
            states=State(
                y=jax.lax.stop_gradient(y),
                z=jax.lax.stop_gradient(z),
            ),
            steps=new_steps,
            halted=halted,
            data=carry.data,
        ), {
            "logits": logits,
            "q_halt_logits": qh,
            "y_states": y_hist,
            "z_states": z_hist,
        }

    def update_state(
        self,
        state: jnp.ndarray,
        context: jnp.ndarray,
        cos_sin: CosSin,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.network(state, context, cos_sin).astype(self.forward_dtype)

    def inference(
        self,
        y: jnp.ndarray,
        z: jnp.ndarray,
        embeddings: jnp.ndarray,
        cos_sin: CosSin,
        *,
        record: bool = False,
        key: jnp.ndarray,
    ):
        y_state = y.astype(self.forward_dtype)
        z_state = z.astype(self.forward_dtype)
        y_records = [] if record else None
        z_records = [] if record else None

        for idx in range(self.config.y_cycles):
            x_inj = (y_state + embeddings).astype(self.forward_dtype)
            key, z_rng = jax.random.split(key)

            z_state, z_hist = self._run_z_cycles(
                z_state, x_inj, cos_sin, record=record, key=z_rng
            )

            key, y_rng = jax.random.split(key)
            y_state = self.update_state(y_state, z_state, cos_sin, key=y_rng)

            if record:
                y_records.append(jax.lax.stop_gradient(y_state))
                z_records.append(z_hist)

            if idx < self.config.y_cycles - 1:
                y_state = jax.lax.stop_gradient(y_state)
                z_state = jax.lax.stop_gradient(z_state)

        y_records = jnp.stack(y_records) if record else None
        z_records = jnp.stack(z_records) if record else None
        return y_state, z_state, y_records, z_records

    def _run_z_cycles(
        self,
        z_state: jnp.ndarray,
        x_inj: jnp.ndarray,
        cos_sin: CosSin,
        *,
        record: bool,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        keys = jax.random.split(key, self.config.z_cycles)

        def body(z_curr, rng):
            z_next = self.update_state(z_curr, x_inj, cos_sin, key=rng)
            hist = jax.lax.stop_gradient(z_next) if record else None
            return z_next, hist

        body = eqx.filter_checkpoint(body)
        z_final, z_hist = jax.lax.scan(body, z_state, keys)
        return z_final, z_hist

    def _update_halt_state(
        self,
        steps: jnp.ndarray,
        q_logits: jnp.ndarray,
        rng: jnp.ndarray,
        training: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

        return new_steps, halted, rng

    def empty_state(self, batch_size: int) -> State:
        zeros = jnp.zeros(
            (
                batch_size,
                self.config.seq_len + self.task_emb_len,
                self.config.hidden_size,
            ),
            dtype=self.forward_dtype,
        )
        return State(y=zeros, z=zeros)

    def reset_states(self, reset: jnp.ndarray, states: State) -> State:
        flag = reset.reshape((-1, 1, 1))
        y0 = self.H_init[None, None, :]
        z0 = self.L_init[None, None, :]
        y = jnp.where(flag, y0, states.y)
        z = jnp.where(flag, z0, states.z)
        return State(y=y, z=z)

    def embed(self, inputs: jnp.ndarray, task_ids: jnp.ndarray) -> jnp.ndarray:
        tok = self.embed_tokens(inputs.astype(jnp.int32))

        task = self.task_embed(task_ids)
        need = self.task_emb_len * self.config.hidden_size - task.shape[-1]
        if need > 0:
            task = jnp.pad(task, ((0, 0), (0, need)))

        task = task.reshape(-1, self.task_emb_len, self.config.hidden_size)
        emb = jnp.concatenate([task, tok], axis=1)

        emb = (emb * self.embed_scale).astype(self.forward_dtype)
        return emb

    def initial_carry(self, batch: Dict[str, jnp.ndarray]) -> Carry:
        bs = batch["inputs"].shape[0]
        return Carry(
            states=self.empty_state(bs),
            steps=jnp.zeros((bs,), dtype=jnp.int32),
            halted=jnp.ones((bs,), dtype=jnp.bool_),
            data={k: jnp.zeros_like(v) for k, v in batch.items()},
        )
