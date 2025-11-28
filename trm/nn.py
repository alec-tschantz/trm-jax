import math
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from trm.utils import trunc_normal


CosSin = Tuple[jnp.ndarray, jnp.ndarray]


class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray | None
    use_bias: bool = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, *, bias: bool, key):
        std = 1.0 / math.sqrt(in_features)
        self.weight = trunc_normal(key, (out_features, in_features), std=std).astype(
            jnp.float32
        )
        self.use_bias = bias
        if bias:
            self.bias = jnp.zeros((out_features,), dtype=jnp.float32)
        else:
            self.bias = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out_dtype = x.dtype
        x32 = x.astype(jnp.float32)
        weight_t = jnp.transpose(self.weight, (1, 0))
        out = jnp.matmul(x32, weight_t)
        if self.bias is not None:
            out = out + self.bias
        return out.astype(out_dtype)


class Embedding(eqx.Module):
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

    def __call__(self, input: jnp.ndarray) -> jnp.ndarray:
        idx = input.astype(jnp.int32)
        embeds = self.weight[idx]
        return embeds.astype(self.cast_to)


class RotaryEmbedding(eqx.Module):
    cos_cached: jnp.ndarray = eqx.field(static=True)
    sin_cached: jnp.ndarray = eqx.field(static=True)

    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = jnp.cos(emb)
        self.sin_cached = jnp.sin(emb)

    def __call__(self) -> CosSin:
        return self.cos_cached, self.sin_cached


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


class Attention(eqx.Module):
    hidden_size: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_key_value_heads: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)

    qkv_proj: Linear
    o_proj: Linear

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
        self.output_size = num_heads * head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        k_qkv, k_o = jax.random.split(key)
        self.qkv_proj = Linear(
            hidden_size,
            (num_heads + 2 * num_key_value_heads) * head_dim,
            bias=False,
            key=k_qkv,
        )
        self.o_proj = Linear(self.output_size, hidden_size, bias=False, key=k_o)

    def __call__(self, cos_sin: CosSin, hidden_states: jnp.ndarray) -> jnp.ndarray:
        out_dtype = hidden_states.dtype
        b, s, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(
            b, s, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )

        q = qkv[:, :, : self.num_heads]  # (B, S, H, D)
        k = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        v = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            cos = cos[:s]
            sin = sin[:s]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.astype(jnp.bfloat16)
        k = k.astype(jnp.bfloat16)
        v = v.astype(jnp.bfloat16)

        out_heads = _flash_attention(q, k, v, causal=self.causal)  # (B, S, H, D)

        out = out_heads.reshape(b, s, self.output_size)
        out = out.astype(out_dtype)
        return self.o_proj(out)


class SwiGLU(eqx.Module):
    gate_up_proj: Linear
    down_proj: Linear
    hidden_size: int = eqx.field(static=True)
    expansion: float = eqx.field(static=True)

    def __init__(self, hidden_size: int, expansion: float, *, key):
        self.hidden_size = hidden_size
        self.expansion = expansion
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        gate_key, down_key = jax.random.split(key)
        self.gate_up_proj = Linear(hidden_size, inter * 2, bias=False, key=gate_key)
        self.down_proj = Linear(inter, hidden_size, bias=False, key=down_key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate, up = jnp.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(jax.nn.silu(gate) * up)


class Block(eqx.Module):
    self_attn: Attention
    mlp: SwiGLU
    norm_eps: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        hidden_size: int,
        expansion: float,
        num_heads: int,
        rms_norm_eps: float,
        key,
    ):
        k1, k2 = jax.random.split(key)
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
            key=k1,
        )
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            expansion=expansion,
            key=k2,
        )
        self.norm_eps = rms_norm_eps

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
            # h = layer(cos_sin, h)
            h = eqx.filter_checkpoint(layer)(cos_sin, h)

        return h


def _flash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    *,
    causal: bool,
) -> jnp.ndarray:

    B, T, H, D = q.shape
    S = k.shape[1]

    def _pad(x: jnp.ndarray, pad_len: int) -> jnp.ndarray:
        return jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)))

    Q = ((T + 3) // 4) * 4
    K = ((S + 3) // 4) * 4
    pad_T = Q - T
    pad_S = K - S

    q_p = _pad(q, pad_T)
    k_p = _pad(k, pad_S)
    v_p = _pad(v, pad_S)

    attention_mask = jnp.ones((Q, K), dtype=jnp.bool_)
    attention_mask = attention_mask.at[T:, :].set(False)
    attention_mask = attention_mask.at[:, S:].set(False)

    mask = attention_mask[None, None, :, :]

    out = jax.nn.dot_product_attention(
        query=q_p,
        key=k_p,
        value=v_p,
        mask=mask,
        bias=None,
        implementation="cudnn",
        is_causal=causal,
    )

    return out[:, :T, :, :]


def rms_norm(hidden_states: jnp.ndarray, eps: float) -> jnp.ndarray:
    orig_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(variance + eps)
    return hidden_states.astype(orig_dtype)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    orig_dtype = q.dtype
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q = q.astype(cos.dtype)
    k = k.astype(cos.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b
