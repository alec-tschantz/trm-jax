# dataset.py
# Deterministic addition dataset for the TRM model.
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class AdditionDatasetConfig:
    batch_size: int
    seq_len: int
    max_addend: int = 9
    vocab_size: int = 11  # tokens 0-10 inclusive
    plus_token: int = 10
    pad_token: int = 0


class AdditionDataset:
    def __init__(self, config: AdditionDatasetConfig):
        self.cfg = config
        core_len = 5  # a, +, b, tens(sum), ones(sum)
        if self.cfg.seq_len < core_len:
            raise ValueError("seq_len must be at least 5 for addition examples.")
        if not (0 <= self.cfg.plus_token < self.cfg.vocab_size):
            raise ValueError("plus_token must lie within the vocabulary range.")
        if self.cfg.plus_token == self.cfg.pad_token:
            raise ValueError("plus_token must differ from pad_token.")
        if self.cfg.max_addend < 0:
            raise ValueError("max_addend must be non-negative.")
        if self.cfg.max_addend > 9:
            raise ValueError("max_addend must be ≤ 9 so digits fit in the 0-10 vocab.")

    def sample(self, key: jax.random.PRNGKey):
        cfg = self.cfg
        B = cfg.batch_size
        k_a, k_b = jax.random.split(key)
        addend_a = jax.random.randint(
            k_a, (B,), minval=0, maxval=cfg.max_addend + 1, dtype=jnp.int32
        )
        addend_b = jax.random.randint(
            k_b, (B,), minval=0, maxval=cfg.max_addend + 1, dtype=jnp.int32
        )
        total = addend_a + addend_b
        tens = total // 10
        ones = total % 10

        plus = jnp.full((B,), cfg.plus_token, dtype=jnp.int32)
        core_tokens = jnp.stack([addend_a, plus, addend_b, tens, ones], axis=1)

        pad_len = cfg.seq_len - core_tokens.shape[1]
        if pad_len > 0:
            pad = jnp.full((B, pad_len), cfg.pad_token, dtype=jnp.int32)
            tokens = jnp.concatenate([core_tokens, pad], axis=1)
            mask_core = jnp.ones_like(core_tokens, dtype=jnp.bool_)
            mask_pad = jnp.zeros((B, pad_len), dtype=jnp.bool_)
            mask = jnp.concatenate([mask_core, mask_pad], axis=1)
        else:
            tokens = core_tokens
            mask = jnp.ones_like(core_tokens, dtype=jnp.bool_)

        batch_dict = {
            "inputs": tokens,
            "labels": tokens,
            "input_mask": mask,
            "output_mask": mask,
            "task_tokens": jnp.zeros((B,), dtype=jnp.int32),
        }
        return batch_dict
