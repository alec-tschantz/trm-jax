# nn.py
# Equinox/JAX utilities and layers used by TRM.
from __future__ import annotations

import math
from typing import Tuple, Optional
import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import erfinv


CosSin = Tuple[jnp.ndarray, jnp.ndarray]


# -----------------------
# Inits and misc utils
# -----------------------
def trunc_normal_init_(
    key: jax.random.PRNGKey,
    shape,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Correct truncated normal init (mirrors provided PyTorch version)."""
    if std == 0.0:
        return jnp.zeros(shape, dtype=dtype)

    sqrt2 = math.sqrt(2.0)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    z = (b - a) / 2.0

    c = (2.0 * math.pi) ** -0.5
    pdf_u = c * math.exp(-0.5 * (lower**2))
    pdf_l = c * math.exp(-0.5 * (upper**2))
    comp_std = std / math.sqrt(
        1.0 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
    )

    u = jax.random.uniform(key, shape=shape, dtype=dtype, minval=a, maxval=b)
    x = erfinv(u) * (sqrt2 * comp_std)
    x = jnp.clip(x, lower * comp_std, upper * comp_std)
    return x.astype(dtype)


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # q, k: [B, S, H, D], cos/sin: [S, D]
    orig_dtype = q.dtype
    q = q.astype(cos.dtype)
    k = k.astype(cos.dtype)
    cos_e = cos[None, :, None, :]
    sin_e = sin[None, :, None, :]
    q_embed = (q * cos_e) + (rotate_half(q) * sin_e)
    k_embed = (k * cos_e) + (rotate_half(k) * sin_e)
    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


# -----------------------
# Core layers
# -----------------------
class CastedLinear(eqx.Module):
    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        key: jax.random.PRNGKey,
    ):
        kw, kb = jax.random.split(key)
        std = 1.0 / math.sqrt(in_features)  # LeCun truncated normal
        self.weight = trunc_normal_init_(
            kw, (out_features, in_features), std=std, dtype=jnp.float32
        )
        self.bias = (
            trunc_normal_init_(kb, (out_features,), std=0.0, dtype=jnp.float32)
            if bias
            else None
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        w = self.weight.astype(x.dtype)
        y = x @ w.T
        if self.bias is not None:
            y = y + self.bias.astype(x.dtype)
        return y


class CastedEmbedding(eqx.Module):
    embedding_weight: jnp.ndarray
    cast_to: jnp.dtype = eqx.static_field()

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float,
        cast_to: jnp.dtype,
        *,
        key: jax.random.PRNGKey,
    ):
        self.cast_to = cast_to
        self.embedding_weight = trunc_normal_init_(
            key, (num_embeddings, embedding_dim), std=init_std, dtype=jnp.float32
        )

    def __call__(self, indices: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(self.embedding_weight.astype(self.cast_to), indices, axis=0)


class RotaryEmbedding(eqx.Module):
    inv_freq: np.ndarray = eqx.static_field()
    max_position_embeddings: int = eqx.static_field()

    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        inv = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        self.inv_freq = inv.astype(np.float32)
        self.max_position_embeddings = int(max_position_embeddings)

    def __call__(self, seq_len: int) -> CosSin:
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, jnp.asarray(self.inv_freq))  # [S, dim/2]
        emb = jnp.concatenate([freqs, freqs], axis=-1)  # [S, dim]
        return jnp.cos(emb), jnp.sin(emb)


def _scaled_dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    *,
    attention_mask: Optional[jnp.ndarray],
) -> jnp.ndarray:
    # q,k,v: [B, H, S, D]
    d = query.shape[-1]
    qf = query.astype(jnp.float32)
    kf = key.astype(jnp.float32)
    vf = value.astype(jnp.float32)
    scores = jnp.einsum("bhqd,bhkd->bhqk", qf, kf) * (1.0 / math.sqrt(d))
    neg_inf = jnp.array(-1e9, dtype=scores.dtype)
    if attention_mask is not None:
        key_mask = attention_mask[:, None, None, :].astype(bool)
        scores = jnp.where(key_mask, scores, neg_inf)
    S = scores.shape[-1]
    causal_mask = jnp.triu(jnp.ones((S, S), dtype=bool), k=1)
    scores = jnp.where(
        causal_mask[None, None, :, :],
        jnp.full_like(scores, neg_inf),
        scores,
    )
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", attn, vf)
    if attention_mask is not None:
        query_mask = attention_mask[:, None, :, None].astype(out.dtype)
        out = out * query_mask
    return out.astype(query.dtype)


class Attention(eqx.Module):
    hidden_size: int = eqx.static_field()
    head_dim: int = eqx.static_field()
    output_size: int = eqx.static_field()
    num_heads: int = eqx.static_field()
    num_key_value_heads: int = eqx.static_field()

    qkv_proj: CastedLinear
    o_proj: CastedLinear

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        *,
        key: jax.random.PRNGKey,
    ):
        self.hidden_size = int(hidden_size)
        self.head_dim = int(head_dim)
        self.output_size = int(head_dim * num_heads)
        self.num_heads = int(num_heads)
        self.num_key_value_heads = int(num_key_value_heads)

        kqkv, ko = jax.random.split(key)
        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
            key=kqkv,
        )
        self.o_proj = CastedLinear(
            self.output_size, self.hidden_size, bias=False, key=ko
        )

    def __call__(
        self,
        cos_sin: Optional[CosSin],
        hidden_states: jnp.ndarray,
        *,
        attention_mask: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        B, S, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)  # [B, S, (H + 2*KV)*Dh]

        qkv = qkv.reshape(
            B, S, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )
        query = qkv[:, :, : self.num_heads, :]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads, :]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :, :]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # [B, S, H, D] -> [B, H, S, D]
        query = jnp.swapaxes(query, 1, 2)
        key = jnp.swapaxes(key, 1, 2)
        value = jnp.swapaxes(value, 1, 2)

        out = _scaled_dot_product_attention(
            query, key, value, attention_mask=attention_mask
        )

        # [B, H, S, D] -> [B, S, H*D]
        out = jnp.swapaxes(out, 1, 2).reshape(B, S, self.output_size)
        return self.o_proj(out)


class LinearSwish(eqx.Module):
    linear: CastedLinear

    def __init__(self, hidden_size: int, *, key: jax.random.PRNGKey):
        self.linear = CastedLinear(hidden_size, hidden_size, bias=False, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(jax.nn.silu(x))


class SwiGLU(eqx.Module):
    gate_up_proj: CastedLinear
    down_proj: CastedLinear

    def __init__(self, hidden_size: int, expansion: float, *, key: jax.random.PRNGKey):
        inter_approx = round(expansion * hidden_size * 2.0 / 3.0)
        inter = _find_multiple(inter_approx, 256)
        k1, k2 = jax.random.split(key)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False, key=k1)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gu = self.gate_up_proj(x)
        gate, up = jnp.split(gu, 2, axis=-1)
        return self.down_proj(jax.nn.silu(gate) * up)


def rms_norm(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    inp_dtype = x.dtype
    xf = x.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    out = xf * lax.rsqrt(var + eps)
    return out.astype(inp_dtype)
