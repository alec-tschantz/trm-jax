from __future__ import annotations

import math
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from trm_jax.utils import trunc_normal


CosSin = Tuple[jnp.ndarray, jnp.ndarray]


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CastedLinear(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, in_features: int, out_features: int, *, bias: bool, key):
        self.linear = eqx.nn.Linear(
            in_features,
            out_features,
            use_bias=bias,
            key=key,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        out = jax.vmap(self.linear)(x_flat)
        return out.reshape(*orig_shape, out.shape[-1]).astype(x.dtype)


class CastedEmbedding(eqx.Module):
    embedding: eqx.nn.Embedding
    cast_to: jnp.dtype = eqx.field(static=True)

    def __init__(self, num_embeddings: int, embedding_dim: int, *, key, cast_to=jnp.float32):
        self.embedding = eqx.nn.Embedding(num_embeddings, embedding_dim, key=key)
        self.cast_to = cast_to

    def __call__(self, input: jnp.ndarray) -> jnp.ndarray:
        flat = input.reshape(-1)
        embeds = jax.vmap(self.embedding)(flat)
        out_dim = embeds.shape[-1]
        return embeds.reshape(*input.shape, out_dim).astype(self.cast_to)


class RotaryEmbedding(eqx.Module):
    cos_cached: jnp.ndarray
    sin_cached: jnp.ndarray

    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        inv_freq = 1.0 / (
            base**(jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = jnp.cos(emb)
        self.sin_cached = jnp.sin(emb)

    def __call__(self) -> CosSin:
        return self.cos_cached, self.sin_cached


class Attention(eqx.Module):
    hidden_size: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_key_value_heads: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)

    qkv_proj: CastedLinear
    o_proj: CastedLinear

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        causal: bool,
        *,
        key,
    ):
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        q_key, o_key = jax.random.split(key)
        self.qkv_proj = CastedLinear(
            hidden_size,
            (num_heads + 2 * num_key_value_heads) * head_dim,
            bias=False,
            key=q_key,
        )
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False, key=o_key)

    def __call__(self, cos_sin: CosSin, hidden_states: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(
            batch_size,
            seq_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )
        query = qkv[:, :, : self.num_heads]
        key = qkv[
            :, :, self.num_heads : self.num_heads + self.num_key_value_heads
        ]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            cos = cos[:seq_len]
            sin = sin[:seq_len]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = jnp.einsum("bthd, bshd -> bhts", query, key) * scale
        if self.causal:
            mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            attn_scores = jnp.where(mask[None, None, :, :], attn_scores, -1e9)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        attn_output = jnp.einsum("bhts, bshd -> bthd", attn_weights, value)
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class SwiGLU(eqx.Module):
    gate_up_proj: CastedLinear
    down_proj: CastedLinear
    hidden_size: int = eqx.field(static=True)
    expansion: float = eqx.field(static=True)

    def __init__(self, hidden_size: int, expansion: float, *, key):
        self.hidden_size = hidden_size
        self.expansion = expansion
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        gate_key, down_key = jax.random.split(key)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False, key=gate_key)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False, key=down_key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate, up = jnp.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(jax.nn.silu(gate) * up)


def rms_norm(hidden_states: jnp.ndarray, variance_epsilon: float) -> jnp.ndarray:
    orig_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(variance + variance_epsilon)
    return hidden_states.astype(orig_dtype)
