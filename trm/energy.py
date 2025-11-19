import math
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from trm.nn import Attention, CosSin, Embedding, Linear, RotaryEmbedding, SwiGLU, rms_norm


class Block(eqx.Module):
    self_attn: Attention
    mlp: SwiGLU
    norm_eps: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        rms_norm_eps: float,
        key: jax.Array,
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
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion, key=k2)
        self.norm_eps = rms_norm_eps

    def __call__(self, cos_sin: CosSin, hidden_states: jnp.ndarray) -> jnp.ndarray:
        dtype = hidden_states.dtype
        attn_out = self.self_attn(cos_sin, hidden_states)
        hidden_states = rms_norm(hidden_states + attn_out.astype(dtype), eps=self.norm_eps)
        mlp_out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_out.astype(dtype), eps=self.norm_eps)
        return hidden_states


class Transformer(eqx.Module):
    layers: Tuple[Block, ...]

    def __call__(self, h: jnp.ndarray, cos_sin: CosSin) -> jnp.ndarray:
        for layer in self.layers:
            h = layer(cos_sin, h)
        return h


class Energy(eqx.Module):
    embed_tokens: Embedding
    z_embed_tokens: Embedding
    rotary_emb: RotaryEmbedding
    network: Transformer
    energy_head: Linear
    embed_scale: float = eqx.field(static=True)
    forward_dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        *,
        vocab_size: int,
        z_vocab_size: int,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        num_layers: int,
        rms_norm_eps: float,
        seq_len: int,
        rope_theta: float,
        forward_dtype: str,
        key: jax.Array,
    ):
        dtype = getattr(jnp, forward_dtype)
        self.forward_dtype = dtype
        self.embed_scale = math.sqrt(hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.embed_tokens = Embedding(
            vocab_size,
            hidden_size,
            init_std=embed_init_std,
            key=k1,
            cast_to=dtype,
        )
        self.z_embed_tokens = Embedding(
            z_vocab_size,
            hidden_size,
            init_std=embed_init_std,
            key=k2,
            cast_to=dtype,
        )
        self.energy_head = Linear(hidden_size, 1, bias=True, key=k3)
        layer_keys = jax.random.split(k4, num_layers)
        self.network = Transformer(
            tuple(
                Block(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    expansion=expansion,
                    rms_norm_eps=rms_norm_eps,
                    key=kk,
                )
                for kk in layer_keys
            )
        )
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            max_position_embeddings=seq_len,
            base=rope_theta,
        )

    def embed_inputs(self, inputs: jnp.ndarray) -> jnp.ndarray:
        tokens = self.embed_tokens(inputs.astype(jnp.int32))
        return (tokens * self.embed_scale).astype(self.forward_dtype)

    def logits_to_embeddings(self, logits: jnp.ndarray, *, weight: jnp.ndarray) -> jnp.ndarray:
        logits32 = logits.astype(jnp.float32)
        orig_shape = logits32.shape[:-1]
        flat = logits32.reshape(-1, logits32.shape[-1])
        probs = jax.nn.softmax(flat, axis=-1)
        weight32 = weight.astype(jnp.float32)
        embeds = jnp.matmul(probs, weight32)
        embed_shape = orig_shape + (weight32.shape[-1],)
        return embeds.reshape(embed_shape).astype(self.forward_dtype)

    def energy_map(
        self,
        x_embed: jnp.ndarray,
        y_embed: jnp.ndarray,
        z_embed: jnp.ndarray,
        cos_sin: CosSin,
    ) -> jnp.ndarray:
        if z_embed.ndim == 2:
            z_embed = z_embed[None, ...]
        h = (x_embed + y_embed + z_embed).astype(self.forward_dtype)
        energy = self.network(h, cos_sin)
        return self.energy_head(energy).astype(jnp.float32)
