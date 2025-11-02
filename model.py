# model.py
# Tiny Recursive Model (TRM) in JAX/Equinox with simplified names and only the no_ACT_continue path.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from nn import (
    CosSin,
    trunc_normal_init_,
    CastedEmbedding,
    CastedLinear,
    RotaryEmbedding,
    Attention,
    SwiGLU,
    rms_norm,
)


# -------------------------
# Public single-word names
# -------------------------
@dataclass
class Config:
    batch_size: int
    seq_len: int
    task_embedding_dim: int
    num_task_embeddings: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    L_layers: int

    hidden_size: int
    expansion: float
    num_heads: int

    # Only RoPE in this implementation
    rope_theta: float = 10000.0

    # ACT settings
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0

    # Numerics
    forward_dtype: str = "bfloat16"

    # Always > 0
    task_token_length: int = 16


class InnerCarry(eqx.Module):
    z_H: jnp.ndarray
    z_L: jnp.ndarray


class Carry(eqx.Module):
    inner_carry: InnerCarry
    steps: jnp.ndarray
    halted: jnp.ndarray
    current_data: Dict[str, jnp.ndarray]


class Block(eqx.Module):
    attn: Attention
    mlp: SwiGLU
    norm_eps: float

    def __init__(self, cfg: Config, *, key: jax.random.PRNGKey):
        ka, km = jax.random.split(key)
        self.attn = Attention(
            hidden_size=cfg.hidden_size,
            head_dim=cfg.hidden_size // cfg.num_heads,
            num_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_heads,
            key=ka,
        )
        self.mlp = SwiGLU(hidden_size=cfg.hidden_size, expansion=cfg.expansion, key=km)
        self.norm_eps = 1e-5

    def __call__(
        self,
        *,
        cos_sin: Optional[CosSin],
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        attn_out = self.attn(
            cos_sin=cos_sin,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        h = rms_norm(hidden_states + attn_out, self.norm_eps)
        out = self.mlp(h)
        h = rms_norm(h + out, self.norm_eps)
        return h


class Reasoning(eqx.Module):
    layers: List[Block]

    def __init__(self, cfg: Config, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, cfg.L_layers)
        self.layers = [Block(cfg, key=k) for k in keys]

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_injection: jnp.ndarray,
        *,
        cos_sin: Optional[CosSin],
        attention_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        h = hidden_states + input_injection
        for layer in self.layers:
            h = layer(cos_sin=cos_sin, hidden_states=h, attention_mask=attention_mask)
        return h


class _Inner(eqx.Module):
    cfg: Config = eqx.static_field()

    # numerics
    forward_dtype: jnp.dtype = eqx.static_field()

    # embeddings/heads
    embed_scale: float
    embed_tokens: CastedEmbedding
    task_embedding: CastedEmbedding
    lm_head: CastedLinear
    q_head: CastedLinear

    # position
    rotary_emb: RotaryEmbedding

    # reasoning
    L_level: Reasoning

    # initial H/L
    H_init: jnp.ndarray
    L_init: jnp.ndarray

    def __init__(self, cfg: Config, *, key: jax.random.PRNGKey):
        self.cfg = cfg
        self.forward_dtype = (
            jnp.bfloat16
            if cfg.forward_dtype.lower() in ("bfloat16", "bf16")
            else jnp.float32
        )

        self.embed_scale = cfg.hidden_size**0.5
        embed_init_std = 1.0 / self.embed_scale

        k_tok, k_puz, k_lm, k_q, k_rope, k_layers, kH, kL = jax.random.split(key, 8)

        self.embed_tokens = CastedEmbedding(
            cfg.vocab_size,
            cfg.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
            key=k_tok,
        )
        # Always present task embedding; assume > 0 dims and > 0 len
        self.task_embedding = CastedEmbedding(
            cfg.num_task_embeddings,
            cfg.task_embedding_dim,
            init_std=0.0,
            cast_to=self.forward_dtype,
            key=k_puz,
        )

        self.lm_head = CastedLinear(
            cfg.hidden_size, cfg.vocab_size, bias=False, key=k_lm
        )
        self.q_head = CastedLinear(cfg.hidden_size, 1, bias=True, key=k_q)
        self.q_head = eqx.tree_at(
            lambda m: m.weight, self.q_head, jnp.zeros_like(self.q_head.weight)
        )
        self.q_head = eqx.tree_at(
            lambda m: m.bias, self.q_head, jnp.full_like(self.q_head.bias, -5.0)
        )

        total_len = cfg.seq_len + cfg.task_token_length
        self.rotary_emb = RotaryEmbedding(
            dim=cfg.hidden_size // cfg.num_heads,
            max_position_embeddings=total_len,
            base=cfg.rope_theta,
        )

        self.L_level = Reasoning(cfg, key=k_layers)

        self.H_init = trunc_normal_init_(
            kH, (cfg.hidden_size,), std=1.0, dtype=self.forward_dtype
        )
        self.L_init = trunc_normal_init_(
            kL, (cfg.hidden_size,), std=1.0, dtype=self.forward_dtype
        )

    # ---- helpers ----
    def _input_embeddings(
        self, batch: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        input_ids = batch["inputs"]
        task_tokens = batch["task_tokens"]
        input_mask = batch["input_mask"].astype(jnp.bool_)
        output_mask = batch["output_mask"].astype(jnp.bool_)

        B, S = input_ids.shape
        tok = self.embed_tokens(input_ids.astype(jnp.int32))  # [B,S,D]

        # task embedding -> pad -> reshape to [B, task_token_length, D] -> concat
        task_vec = self.task_embedding(
            task_tokens.astype(jnp.int32)
        )  # [B, task_embedding_dim]
        need = self.cfg.task_token_length * self.cfg.hidden_size - task_vec.shape[-1]
        pad = jnp.zeros((B, need), dtype=task_vec.dtype)
        task_vec = jnp.concatenate([task_vec, pad], axis=-1).reshape(
            B, self.cfg.task_token_length, self.cfg.hidden_size
        )

        embeddings = jnp.concatenate([task_vec, tok], axis=1)
        embeddings = (self.embed_scale * embeddings).astype(self.forward_dtype)

        token_mask = jnp.logical_and(input_mask, output_mask)
        task_mask = jnp.ones((B, self.cfg.task_token_length), dtype=token_mask.dtype)
        attention_mask = jnp.concatenate([task_mask, token_mask], axis=1)
        embeddings = embeddings * attention_mask[..., None].astype(embeddings.dtype)
        return embeddings, attention_mask

    def empty_carry(self, batch_size: int) -> InnerCarry:
        L = self.cfg.seq_len + self.cfg.task_token_length
        shape = (batch_size, L, self.cfg.hidden_size)
        return InnerCarry(
            z_H=jnp.empty(shape, dtype=self.forward_dtype),
            z_L=jnp.empty(shape, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: jnp.ndarray, carry: InnerCarry) -> InnerCarry:
        return InnerCarry(
            z_H=jnp.where(reset_flag[:, None, None], self.H_init, carry.z_H),
            z_L=jnp.where(reset_flag[:, None, None], self.L_init, carry.z_L),
        )

    # ---- forward ----
    def __call__(
        self, carry: InnerCarry, batch: Dict[str, jnp.ndarray]
    ) -> Tuple[InnerCarry, jnp.ndarray, jnp.ndarray]:
        x, attention_mask = self._input_embeddings(batch)
        seq_len = x.shape[1]
        cos_sin = self.rotary_emb(seq_len)
        mask_f = attention_mask[..., None].astype(self.forward_dtype)
        z_H, z_L = carry.z_H, carry.z_L
        z_H = z_H * mask_f
        z_L = z_L * mask_f

        # H_cycles - 1 with no grad
        for _ in range(max(0, self.cfg.H_cycles - 1)):
            for __ in range(self.cfg.L_cycles):
                z_L = lax.stop_gradient(
                    self.L_level(
                        z_L,
                        z_H + x,
                        cos_sin=cos_sin,
                        attention_mask=attention_mask,
                    )
                )
            z_H = lax.stop_gradient(
                self.L_level(z_H, z_L, cos_sin=cos_sin, attention_mask=attention_mask)
            )

        # Final with grad
        for __ in range(self.cfg.L_cycles):
            z_L = self.L_level(
                z_L, z_H + x, cos_sin=cos_sin, attention_mask=attention_mask
            )
        z_H = self.L_level(z_H, z_L, cos_sin=cos_sin, attention_mask=attention_mask)

        z_H = z_H * mask_f
        z_L = z_L * mask_f

        new_carry = InnerCarry(z_H=lax.stop_gradient(z_H), z_L=lax.stop_gradient(z_L))
        logits = jax.vmap(jax.vmap(self.lm_head))(
            z_H[:, self.cfg.task_token_length :, :]
        ).astype(
            jnp.float32
        )  # [B,S,V]
        q_logits = self.q_head(z_H[:, 0, :]).astype(jnp.float32).squeeze(-1)  # [B]
        return new_carry, logits, q_logits


class Model(eqx.Module):
    cfg: Config = eqx.static_field()
    inner: _Inner

    def __init__(self, cfg: Config, *, key: jax.random.PRNGKey):
        self.cfg = cfg
        self.inner = _Inner(cfg, key=key)

    def initial_carry(self, batch: Dict[str, jnp.ndarray]) -> Carry:
        B = batch["inputs"].shape[0]
        return Carry(
            inner_carry=self.inner.empty_carry(B),
            steps=jnp.zeros((B,), dtype=jnp.int32),
            halted=jnp.ones((B,), dtype=jnp.bool_),
            current_data={k: jnp.empty_like(v) for k, v in batch.items()},
        )

    def __call__(
        self,
        carry: Carry,
        batch: Dict[str, jnp.ndarray],
        *,
        key: Optional[jax.random.PRNGKey] = None,
        is_training: bool = False,
    ):
        # refresh on halted
        inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        steps = jnp.where(carry.halted, 0, carry.steps)

        def _bcast(mask, x_new, x_old):
            return jnp.where(mask[(...,) + (None,) * (x_new.ndim - 1)], x_new, x_old)

        current = {
            k: _bcast(carry.halted, batch[k], v) for k, v in carry.current_data.items()
        }

        # inner forward
        inner_carry, logits, q_halt = self.inner(inner_carry, current)
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
        }

        # step & halt
        steps = steps + 1
        is_last = steps >= self.cfg.halt_max_steps
        halted = is_last

        if is_training and (self.cfg.halt_max_steps > 1):
            halted = jnp.logical_or(halted, q_halt > 0.0)

            if key is not None and self.cfg.halt_exploration_prob > 0.0:
                k1, k2 = jax.random.split(key)
                explore = (
                    jax.random.uniform(k1, q_halt.shape)
                    < self.cfg.halt_exploration_prob
                )
                rand_steps = jax.random.randint(
                    k2, steps.shape, minval=2, maxval=self.cfg.halt_max_steps + 1
                )
                min_halt = jnp.where(explore, rand_steps, 0)
                halted = jnp.logical_and(halted, steps >= min_halt)

        new_carry = Carry(
            inner_carry=inner_carry, steps=steps, halted=halted, current_data=current
        )
        return new_carry, outputs
