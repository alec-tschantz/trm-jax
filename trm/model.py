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
    SwiGLU,
    rms_norm,
)
from trm.embedding import SparseEmbedding


class ModelConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int
    num_puzzle_identifiers: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "float32"
    puzzle_emb_len: int = 16


class InnerCarry(eqx.Module):
    z_H: jnp.ndarray
    z_L: jnp.ndarray


class Carry(eqx.Module):
    inner_carry: InnerCarry
    steps: jnp.ndarray
    halted: jnp.ndarray
    current_data: Dict[str, jnp.ndarray]


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
        @jax.checkpoint
        def _block(h):
            dtype = h.dtype
            attn_out = self.self_attn(cos_sin, h)
            h2 = rms_norm(h + attn_out.astype(dtype), variance_epsilon=self.norm_eps)
            mlp_out = self.mlp(h2)
            return rms_norm(h2 + mlp_out.astype(dtype), variance_epsilon=self.norm_eps)

        return _block(h)


class ReasoningModule(eqx.Module):
    layers: Tuple[Block, ...]

    def __call__(
        self, h: jnp.ndarray, inj: jnp.ndarray, cos_sin: CosSin
    ) -> jnp.ndarray:
        h = h + inj
        for layer in self.layers:
            h = layer(cos_sin, h)
        return h


class Inner(eqx.Module):
    config: ModelConfig = eqx.field(static=True)
    forward_dtype: jnp.dtype = eqx.field(static=True)
    embed_scale: float = eqx.field(static=True)
    puzzle_emb_len: int = eqx.field(static=True)

    embed_tokens: Embedding
    lm_head: Linear
    q_head: Linear
    puzzle_emb: SparseEmbedding
    rotary_emb: RotaryEmbedding
    L_level: ReasoningModule
    H_init: jnp.ndarray
    L_init: jnp.ndarray

    def __init__(self, config: ModelConfig, *, key):
        self.config = config
        dtype = getattr(jnp, config.forward_dtype)
        self.forward_dtype = dtype
        self.embed_scale = math.sqrt(config.hidden_size)
        self.puzzle_emb_len = config.puzzle_emb_len
        embed_init_std = 1.0 / self.embed_scale

        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            init_std=embed_init_std,
            key=k1,
            cast_to=dtype,
        )
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            key=k2,
        )
        q_head = Linear(config.hidden_size, 1, bias=True, key=k3)
        q_head = eqx.tree_at(lambda m: m.weight, q_head, jnp.zeros_like(q_head.weight))
        bias_val = jnp.full_like(q_head.bias, -5.0)
        q_head = eqx.tree_at(lambda m: m.bias, q_head, bias_val)
        self.q_head = q_head

        self.puzzle_emb = SparseEmbedding(
            config.num_puzzle_identifiers,
            config.puzzle_emb_ndim,
            init_std=0.0,
            cast_to=dtype,
            key=k4,
        )

        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=config.seq_len + config.puzzle_emb_len,
            base=config.rope_theta,
        )

        layer_keys = jax.random.split(k5, config.L_layers)
        self.L_level = ReasoningModule(
            tuple(Block(config, key=kk) for kk in layer_keys)
        )

        self.H_init = trunc_normal(k6, (config.hidden_size,), std=1.0).astype(dtype)
        self.L_init = trunc_normal(k7, (config.hidden_size,), std=1.0).astype(dtype)

    def _input_embeddings(self, inputs, puzzle_ids):
        tok = self.embed_tokens(inputs.astype(jnp.int32))
        puzz = self.puzzle_emb(puzzle_ids)

        need = self.puzzle_emb_len * self.config.hidden_size - puzz.shape[-1]
        if need > 0:
            puzz = jnp.pad(puzz, ((0, 0), (0, need)))

        puzz = puzz.reshape(-1, self.puzzle_emb_len, self.config.hidden_size)
        emb = jnp.concatenate([puzz, tok], axis=1)
        return (emb * self.embed_scale).astype(self.forward_dtype)

    def empty_carry(self, bs):
        z = jnp.zeros(
            (bs, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size),
            dtype=self.forward_dtype,
        )
        return InnerCarry(z_H=z, z_L=z)

    def reset_carry(self, reset, c):
        flag = reset.reshape((-1, 1, 1))
        h0 = self.H_init[None, None, :]
        l0 = self.L_init[None, None, :]
        zH = jnp.where(flag, h0, c.z_H)
        zL = jnp.where(flag, l0, c.z_L)
        return InnerCarry(z_H=zH, z_L=zL)

    def _run_L(self, z_L, inj, cos_sin):
        z_L = z_L.astype(self.forward_dtype)
        inj = inj.astype(self.forward_dtype)

        def step(h, _):
            out = self.L_level(h, inj, cos_sin).astype(self.forward_dtype)
            return out, None

        z_L, _ = jax.lax.scan(step, z_L, xs=None, length=self.config.L_cycles)
        return z_L

    def _run_H(self, z_H, z_L, inp, cos_sin):
        z_H = z_H.astype(self.forward_dtype)
        z_L = z_L.astype(self.forward_dtype)
        inp = inp.astype(self.forward_dtype)

        def step(carry, _):
            zH, zL = carry
            inj = (zH + inp).astype(self.forward_dtype)

            zL_new = self._run_L(zL, inj, cos_sin)
            zH_new = self.L_level(zH, zL_new, cos_sin).astype(self.forward_dtype)

            zH_new = jax.lax.stop_gradient(zH_new)
            zL_new = jax.lax.stop_gradient(zL_new)

            return (zH_new, zL_new), None

        if self.config.H_cycles > 1:
            (z_H, z_L), _ = jax.lax.scan(
                step, (z_H, z_L), None, length=self.config.H_cycles - 1
            )

        return z_H, z_L

    def __call__(self, carry, batch):
        cos_sin = self.rotary_emb()
        inp = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        z_H, z_L = carry.z_H, carry.z_L

        z_H, z_L = self._run_H(z_H, z_L, inp, cos_sin)

        inj = (z_H + inp).astype(self.forward_dtype)
        z_L = self._run_L(z_L, inj, cos_sin)
        z_H = self.L_level(z_H, z_L, cos_sin).astype(self.forward_dtype)

        new_carry = InnerCarry(
            z_H=jax.lax.stop_gradient(z_H),
            z_L=jax.lax.stop_gradient(z_L),
        )

        logits = self.lm_head(z_H).astype(jnp.float32)[:, self.puzzle_emb_len :, :]
        qh = self.q_head(z_H[:, 0]).astype(jnp.float32).squeeze(-1)

        return new_carry, logits, qh


class Model(eqx.Module):
    config: ModelConfig = eqx.field(static=True)
    inner: Inner

    def __init__(self, cfg, *, key):
        config = ModelConfig(**cfg)
        self.config = config
        self.inner = Inner(config, key=key)

    def initial_carry(self, batch):
        bs = batch["inputs"].shape[0]
        return Carry(
            inner_carry=self.inner.empty_carry(bs),
            steps=jnp.zeros((bs,), dtype=jnp.int32),
            halted=jnp.ones((bs,), dtype=jnp.bool_),
            current_data={k: jnp.zeros_like(v) for k, v in batch.items()},
        )

    def __call__(self, carry, rng, training: bool):
        new_inner, logits, qh = self.inner(carry.inner_carry, carry.current_data)

        new_steps = carry.steps + 1
        is_last = new_steps >= self.config.halt_max_steps
        halted = is_last

        if training and self.config.halt_max_steps > 1:
            halted = jnp.logical_or(halted, qh > 0)
            rng_h, rng_m = jax.random.split(rng)
            r = jax.random.uniform(rng_h, qh.shape)
            sample = r < self.config.halt_exploration_prob
            sampled_min = jax.random.randint(
                rng_m, new_steps.shape, 2, self.config.halt_max_steps + 1, jnp.int32
            )
            min_steps = jnp.where(sample, sampled_min, jnp.zeros_like(sampled_min))
            halted = jnp.logical_and(halted, new_steps >= min_steps)

        return (
            Carry(new_inner, new_steps, halted, carry.current_data),
            {"logits": logits, "q_halt_logits": qh},
        )
