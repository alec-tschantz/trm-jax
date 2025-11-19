import math
from dataclasses import dataclass
from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from pydantic import BaseModel

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
    energy_step_size: float = 0.1
    energy_noise_scale: float = 0.1


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

    embed_tokens: Embedding
    q_head: Linear
    energy_head: Linear
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

        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            init_std=embed_init_std,
            key=k1,
            cast_to=dtype,
        )

        q_head = Linear(config.hidden_size, 1, bias=True, key=k2)
        q_head = eqx.tree_at(lambda m: m.weight, q_head, jnp.zeros_like(q_head.weight))
        bias_val = jnp.full_like(q_head.bias, -5.0)
        q_head = eqx.tree_at(lambda m: m.bias, q_head, bias_val)
        self.q_head = q_head

        self.task_embed = SparseEmbedding(
            config.num_task_identifiers,
            config.task_emb_ndim,
            init_std=0.0,
            cast_to=dtype,
            key=k3,
        )

        self.energy_head = Linear(config.hidden_size, 1, bias=True, key=k4)

        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=config.seq_len + config.task_emb_len,
            base=config.rope_theta,
        )

        layer_keys = jax.random.split(k5, config.num_layers)
        self.network = Transformer(tuple(Block(config, key=kk) for kk in layer_keys))

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

        logits = y.astype(jnp.float32)[:, self.task_emb_len :, :]
        y_embed = self._logits_to_embeddings(y, self.embed_tokens.weight)
        qh = self.q_head(y_embed[:, 0]).astype(jnp.float32).squeeze(-1)
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

    def _logits_to_embeddings(
        self,
        logits: jnp.ndarray,
        weight: jnp.ndarray,
    ) -> jnp.ndarray:
        probs = jnn.softmax(logits.astype(jnp.float32), axis=-1)
        embeds = jnp.einsum("bsv,vd->bsd", probs, weight.astype(jnp.float32))
        return embeds.astype(self.forward_dtype)

    def update_state(
        self,
        logits: jnp.ndarray,
        context: jnp.ndarray,
        cos_sin: CosSin,
        key: jnp.ndarray,
        *,
        weight: jnp.ndarray,
    ) -> jnp.ndarray:
        context = context.astype(self.forward_dtype)
        step = jnp.asarray(self.config.energy_step_size, dtype=jnp.float32)
        noise_scale = jnp.asarray(self.config.energy_noise_scale, dtype=jnp.float32)
        logits32 = logits.astype(jnp.float32)

        def energy_fn(cur_logits):
            energy_map = self._energy_from_logits(
                cur_logits,
                context,
                cos_sin,
                weight=weight,
            )
            return jnp.sum(energy_map)

        grad_E = jax.grad(energy_fn)(logits32)
        new_logits = logits32 - step * grad_E

        if self.config.energy_noise_scale > 0:
            noise = noise_scale * jax.random.normal(
                key, new_logits.shape, dtype=new_logits.dtype
            )
            new_logits = new_logits + noise
        return new_logits.astype(self.forward_dtype)

    def _energy_from_logits(
        self,
        logits: jnp.ndarray,
        context: jnp.ndarray,
        cos_sin: CosSin,
        *,
        weight: jnp.ndarray,
    ) -> jnp.ndarray:
        state_embeddings = self._logits_to_embeddings(logits, weight)
        energy_map = self.network(state_embeddings, context, cos_sin)
        return self.energy_head(energy_map).astype(jnp.float32)

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
            y_embed = self._logits_to_embeddings(y_state, self.embed_tokens.weight)
            context = (y_embed + embeddings).astype(self.forward_dtype)
            key, z_rng = jax.random.split(key)

            z_state, z_hist = self._run_z_cycles(
                z_state,
                context,
                cos_sin,
                record=record,
                key=z_rng,
            )

            z_embed = self._logits_to_embeddings(z_state, self.embed_tokens.weight)
            key, y_rng = jax.random.split(key)
            y_state = self.update_state(
                y_state,
                z_embed,
                cos_sin,
                key=y_rng,
                weight=self.embed_tokens.weight,
            )

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
        context: jnp.ndarray,
        cos_sin: CosSin,
        *,
        record: bool,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        keys = jax.random.split(key, self.config.z_cycles)

        def body(z_curr, rng):
            z_next = self.update_state(
                z_curr,
                context,
                cos_sin,
                key=rng,
                weight=self.embed_tokens.weight,
            )
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
        total_len = self.config.seq_len + self.task_emb_len
        y_zeros = jnp.zeros(
            (batch_size, total_len, self.config.vocab_size), dtype=self.forward_dtype
        )
        z_zeros = jnp.zeros(
            (batch_size, total_len, self.config.vocab_size), dtype=self.forward_dtype
        )
        return State(y=y_zeros, z=z_zeros)

    def reset_states(self, reset: jnp.ndarray, states: State) -> State:
        flag = reset.reshape((-1, 1, 1))
        y = jnp.where(flag, jnp.zeros_like(states.y), states.y)
        z = jnp.where(flag, jnp.zeros_like(states.z), states.z)
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
