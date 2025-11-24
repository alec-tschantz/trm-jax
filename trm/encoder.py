import math
from dataclasses import dataclass
from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from trm.nn import Block, Embedding, Linear, RotaryEmbedding, Transformer
from trm.utils import trunc_normal


DEFAULT_IGNORE_LABEL_ID = -100


@dataclass
class EncoderConfig:
    seq_len: int
    vocab_size: int
    pad_id: int
    ignore_label_id: int = DEFAULT_IGNORE_LABEL_ID

    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    expansion: float = 4.0
    proj_dim: int = 128

    forward_dtype: str = "bfloat16"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0


class Encoder(eqx.Module):
    config: EncoderConfig = eqx.field(static=True)
    forward_dtype: jnp.dtype = eqx.field(static=True)
    embed_scale: float = eqx.field(static=True)

    token_embed: Embedding
    segment_embed: Embedding
    rotary_emb: RotaryEmbedding
    encoder: Transformer
    projector: Linear

    def __init__(self, config: EncoderConfig, *, key):
        self.config = config
        self.forward_dtype = getattr(jnp, config.forward_dtype)
        self.embed_scale = math.sqrt(config.hidden_size)

        k_tok, k_seg, k_proj, k_layers = jax.random.split(key, 4)
        init_std = 1.0 / self.embed_scale

        self.token_embed = Embedding(
            config.vocab_size,
            config.hidden_size,
            init_std=init_std,
            cast_to=self.forward_dtype,
            key=k_tok,
        )
        self.segment_embed = Embedding(
            2,
            config.hidden_size,
            init_std=init_std,
            cast_to=self.forward_dtype,
            key=k_seg,
        )

        layer_keys = jax.random.split(k_layers, config.num_layers)
        self.encoder = Transformer(
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
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=2 * config.seq_len,
            base=config.rope_theta,
        )
        self.projector = Linear(
            config.hidden_size, config.proj_dim, bias=True, key=k_proj
        )

    def encode_examples(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """
        Encode flattened (input, label) examples into latent vectors.

        Args:
            inputs: (N, seq_len) tokenized inputs.
            labels: (N, seq_len) tokenized labels (may contain ignore ids).
        """
        tokens, segments, mask = self._build_tokens(inputs, labels)
        cos_sin = self.rotary_emb()

        tok_emb = self.token_embed(tokens.astype(jnp.int32))
        seg_emb = self.segment_embed(segments)
        h0 = (tok_emb + seg_emb) * self.embed_scale
        h0 = h0.astype(self.forward_dtype)

        zeros = jnp.zeros_like(h0)
        hidden = eqx.filter_checkpoint(self.encoder)(zeros, h0, cos_sin)
        hidden = hidden.astype(jnp.float32)

        pooled = _masked_mean(hidden, mask)
        proj = self.projector(pooled).astype(jnp.float32)
        return proj

    def encode_views(
        self,
        view_inputs: jnp.ndarray,
        view_labels: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode a pair of views shaped (B, K, seq_len) into:
            example_embeddings: (B, K, proj_dim)
            group_embeddings: (B, proj_dim) aggregated over examples
        """
        bsz, k, _ = view_inputs.shape
        flat_inputs = view_inputs.reshape(-1, self.config.seq_len)
        flat_labels = view_labels.reshape(-1, self.config.seq_len)

        example_embeddings = self.encode_examples(flat_inputs, flat_labels)
        example_embeddings = example_embeddings.reshape(bsz, k, -1)
        group_embeddings = jnp.mean(example_embeddings, axis=1)
        return example_embeddings, group_embeddings

    def _build_tokens(
        self, inputs: jnp.ndarray, labels: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        input_mask = inputs != self.config.pad_id
        label_mask = labels != self.config.ignore_label_id
        safe_labels = jnp.where(label_mask, labels, self.config.pad_id)

        tokens = jnp.concatenate([inputs, safe_labels], axis=1)
        segments = jnp.concatenate(
            [
                jnp.zeros_like(inputs, dtype=jnp.int32),
                jnp.ones_like(labels, dtype=jnp.int32),
            ],
            axis=1,
        )
        mask = jnp.concatenate([input_mask, label_mask], axis=1)
        return tokens, segments, mask


def info_nce_loss(
    z1: jnp.ndarray,
    z2: jnp.ndarray,
    *,
    temperature: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Symmetric InfoNCE loss for two batches of embeddings."""
    z1_norm = _l2_normalize(z1)
    z2_norm = _l2_normalize(z2)

    logits = jnp.matmul(z1_norm, jnp.transpose(z2_norm, (1, 0))) / temperature
    labels = jnp.arange(logits.shape[0], dtype=jnp.int32)

    loss12 = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss21 = optax.softmax_cross_entropy_with_integer_labels(
        jnp.transpose(logits, (1, 0)), labels
    )
    loss = 0.5 * (jnp.mean(loss12) + jnp.mean(loss21))

    diag = jnp.sum(z1_norm * z2_norm, axis=-1)
    off_diag = jnp.where(jnp.eye(logits.shape[0], dtype=bool), -1e9, logits)

    metrics: Dict[str, jnp.ndarray] = {
        "info_nce_loss": loss,
        "pos_sim": jnp.mean(diag),
        "logits_std": jnp.std(logits),
        "top1_i2t": jnp.mean(
            (jnp.argmax(logits, axis=1) == labels).astype(jnp.float32)
        ),
        "top1_t2i": jnp.mean(
            (jnp.argmax(logits, axis=0) == labels).astype(jnp.float32)
        ),
        "max_neg_logit": jnp.max(off_diag),
        "mean_norm": jnp.mean(
            jnp.concatenate(
                [jnp.linalg.norm(z1, axis=-1), jnp.linalg.norm(z2, axis=-1)]
            )
        ),
    }
    return loss, metrics


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    denom = jnp.linalg.norm(x, axis=-1, keepdims=True)
    denom = jnp.maximum(denom, eps)
    return x / denom


def _masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    mask_f = mask.astype(values.dtype)[..., None]
    total = jnp.sum(values * mask_f, axis=1)
    counts = jnp.maximum(jnp.sum(mask_f, axis=1), 1.0)
    return total / counts
