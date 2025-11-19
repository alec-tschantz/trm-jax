from typing import Dict

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from pydantic import BaseModel

from trm.energy import Energy


IGNORE_LABEL_ID = -100


class ModelConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    z_vocab_size: int

    y_cycles: int
    z_cycles: int
    num_layers: int

    hidden_size: int
    expansion: float
    num_heads: int

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    forward_dtype: str = "bfloat16"
    energy_noise_scale: float = 0.1
    energy_step_size_min: float = 0.05
    energy_step_size_max: float = 0.15
    max_outer_steps: int = 20


class ModelState(eqx.Module):
    y: jnp.ndarray
    z: jnp.ndarray


class Model(eqx.Module):
    config: ModelConfig = eqx.field(static=True)
    forward_dtype: jnp.dtype = eqx.field(static=True)
    energy: Energy

    def __init__(self, cfg: dict, *, key):
        config = ModelConfig(**cfg)
        self.config = config
        dtype = getattr(jnp, config.forward_dtype)
        self.forward_dtype = dtype
        self.energy = Energy(
            vocab_size=config.vocab_size,
            z_vocab_size=config.z_vocab_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expansion=config.expansion,
            num_layers=config.num_layers,
            rms_norm_eps=config.rms_norm_eps,
            seq_len=config.seq_len,
            rope_theta=config.rope_theta,
            forward_dtype=config.forward_dtype,
            key=key,
        )

    def initial_state(self, batch_size: int) -> ModelState:
        seq = self.config.seq_len
        y = jnp.zeros((batch_size, seq, self.config.vocab_size), dtype=self.forward_dtype)
        z = jnp.zeros((seq, self.config.z_vocab_size), dtype=self.forward_dtype)
        return ModelState(y=y, z=z)

    def _energy_from_components(
        self,
        x_embed: jnp.ndarray,
        y_logits: jnp.ndarray,
        z_logits: jnp.ndarray,
        cos_sin,
    ) -> jnp.ndarray:
        y_embed = self.energy.logits_to_embeddings(y_logits, weight=self.energy.embed_tokens.weight)
        z_embed = self.energy.logits_to_embeddings(z_logits, weight=self.energy.z_embed_tokens.weight)
        return self.energy.energy_map(x_embed, y_embed, z_embed, cos_sin)

    def _update_state(
        self,
        logits: jnp.ndarray,
        x_embed: jnp.ndarray,
        other_logits: jnp.ndarray,
        cos_sin,
        *,
        rng: jnp.ndarray,
        step_size: jnp.ndarray,
        training: bool,
        is_z: bool,
    ) -> jnp.ndarray:
        def energy_fn(cur_logits):
            if is_z:
                return jnp.sum(
                    jnp.mean(
                        self._energy_from_components(
                            x_embed,
                            other_logits,
                            cur_logits,
                            cos_sin,
                        ),
                        axis=0,
                    )
                )
            energy_map = self._energy_from_components(
                x_embed,
                cur_logits,
                other_logits,
                cos_sin,
            )
            return jnp.sum(energy_map)

        logits32 = logits.astype(jnp.float32)
        step = step_size.astype(jnp.float32)
        grad_E = jax.grad(energy_fn)(logits32)
        new_logits = logits32 - step * grad_E

        if training and self.config.energy_noise_scale > 0:
            noise_scale = jnp.asarray(self.config.energy_noise_scale, dtype=new_logits.dtype)
            noise = noise_scale * jax.random.normal(rng, new_logits.shape, dtype=new_logits.dtype)
            new_logits = new_logits + noise
        return new_logits.astype(self.forward_dtype)

    def _run_iteration(
        self,
        state: ModelState,
        x_embed: jnp.ndarray,
        cos_sin,
        *,
        rng: jnp.ndarray,
        step_size: jnp.ndarray,
        training: bool,
        record: bool = False,
    ) -> Tuple[ModelState, jnp.ndarray, jnp.ndarray]:
        y_state = state.y.astype(self.forward_dtype)
        z_state = state.z.astype(self.forward_dtype)
        y_records = [] if record else None

        for z_idx in range(self.config.z_cycles):
            rng, z_rng = jax.random.split(rng)
            z_state = self._update_state(
                z_state,
                x_embed,
                y_state,
                cos_sin,
                rng=z_rng,
                step_size=step_size,
                training=training,
                is_z=True,
            )

            inner_records = [] if record else None
            for y_idx in range(self.config.y_cycles):
                rng, y_rng = jax.random.split(rng)
                y_state = self._update_state(
                    y_state,
                    x_embed,
                    z_state,
                    cos_sin,
                    rng=y_rng,
                    step_size=step_size,
                    training=training,
                    is_z=False,
                )
                if record:
                    inner_records.append(jax.lax.stop_gradient(y_state))
                if not (z_idx == self.config.z_cycles - 1 and y_idx == self.config.y_cycles - 1):
                    y_state = jax.lax.stop_gradient(y_state)

            if record:
                y_records.append(jnp.stack(inner_records))
            if z_idx < self.config.z_cycles - 1:
                z_state = jax.lax.stop_gradient(z_state)

        y_records = jnp.stack(y_records) if record else None
        logits = y_state.astype(jnp.float32)
        return ModelState(y=y_state, z=z_state), logits, y_records

    def _cross_entropy(self, logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        logprobs = jnn.log_softmax(logits.astype(jnp.float32), axis=-1)
        mask = labels != IGNORE_LABEL_ID
        safe_labels = jnp.where(mask, labels, 0)
        per_token = -jnp.take_along_axis(logprobs, safe_labels[..., None], axis=-1).squeeze(-1)
        loss = jnp.sum(jnp.where(mask, per_token, 0.0))
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        return loss / denom

    def _classification_stats(
        self, logits: jnp.ndarray, labels: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        preds = jnp.argmax(logits, axis=-1)
        mask = labels != IGNORE_LABEL_ID
        token_correct = jnp.sum(jnp.where(mask, (preds == labels).astype(jnp.float32), 0.0))
        token_count = jnp.sum(mask)

        seq_mask = jnp.sum(mask, axis=-1) > 0
        seq_correct = jnp.sum(
            jnp.where(
                seq_mask,
                jnp.all(jnp.logical_or(~mask, preds == labels), axis=-1).astype(jnp.float32),
                0.0,
            )
        )
        seq_count = jnp.sum(seq_mask.astype(jnp.float32))
        return token_correct, token_count, seq_correct, seq_count

    def rollout(
        self,
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        rng: jnp.ndarray,
        num_outer_steps: int,
        step_size: jnp.ndarray,
        training: bool,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        batch_size = inputs.shape[0]
        x_embed = self.energy.embed_inputs(inputs)
        cos_sin = self.energy.rotary_emb()
        state = self.initial_state(batch_size)

        max_steps = int(self.config.max_outer_steps)
        num_outer_steps = jnp.asarray(num_outer_steps, dtype=jnp.int32)
        num_outer_steps = jnp.clip(num_outer_steps, 1, max_steps)

        loss_init = jnp.array(0.0, dtype=jnp.float32)
        logits_init = jnp.zeros(state.y.shape, dtype=jnp.float32)

        def body(idx, carry):
            state, rng, loss_acc, last_logits = carry
            rng, step_rng = jax.random.split(rng)
            new_state, logits, _ = self._run_iteration(
                state,
                x_embed,
                cos_sin,
                rng=step_rng,
                step_size=step_size,
                training=training,
                record=False,
            )
            step_loss = self._cross_entropy(logits, labels)
            active = idx < num_outer_steps
            loss_acc = loss_acc + jnp.where(active, step_loss, 0.0)
            state = ModelState(
                y=jnp.where(active, new_state.y, state.y),
                z=jnp.where(active, new_state.z, state.z),
            )
            last_logits = jnp.where(active, logits, last_logits)
            return state, rng, loss_acc, last_logits

        state, rng, loss_total, final_logits = jax.lax.fori_loop(
            0,
            max_steps,
            body,
            (state, rng, loss_init, logits_init),
        )

        avg_loss = loss_total / num_outer_steps.astype(jnp.float32)
        token_correct, token_count, seq_correct, seq_count = self._classification_stats(final_logits, labels)
        token_denom = jnp.maximum(token_count, 1.0)
        seq_denom = jnp.maximum(seq_count, 1.0)
        metrics: Dict[str, jnp.ndarray] = {
            "loss": avg_loss,
            "token_accuracy": token_correct / token_denom,
            "seq_accuracy": seq_correct / seq_denom,
            "token_correct": token_correct,
            "token_count": token_count,
            "seq_correct": seq_correct,
            "seq_count": seq_count,
        }
        return avg_loss, metrics

    def loss(
        self,
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        rng: jnp.ndarray,
        num_outer_steps: int,
        step_size: jnp.ndarray,
        training: bool,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        return self.rollout(
            inputs,
            labels,
            rng=rng,
            num_outer_steps=num_outer_steps,
            step_size=step_size,
            training=training,
        )

    def logit_lens_states(
        self,
        batch: Dict[str, jnp.ndarray],
        *,
        rng: jnp.ndarray,
        step_size: jnp.ndarray,
    ) -> jnp.ndarray:
        state = self.initial_state(batch["inputs"].shape[0])
        x_embed = self.energy.embed_inputs(batch["inputs"])
        cos_sin = self.energy.rotary_emb()
        _, _, y_hist = self._run_iteration(
            state,
            x_embed,
            cos_sin,
            rng=rng,
            step_size=step_size,
            training=False,
            record=True,
        )
        return y_hist
