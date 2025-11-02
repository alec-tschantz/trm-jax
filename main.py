# pretrain.py
# Minimal pretraining loop for TRM in JAX/Optax.
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import optax

from dataset import AdditionDataset, AdditionDatasetConfig
from losses import act_loss
from model import Config as ModelConfig, Model


# -------------------------
# Training configuration
# -------------------------
@dataclass
class TrainConfig:
    # model (reuse your Config dataclass for architecture)
    model: ModelConfig

    # training
    epochs: int = 1
    steps_per_epoch: int = 100
    seed: int = 0

    # optimizer hyperparameters
    lr: float = 3e-4
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 500
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    max_addend: int = 9
    grad_clip_norm: float = 1.0


# -------------------------
# Learning rate schedules
# -------------------------
def cosine_with_warmup(
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
):
    """Cosine schedule to base_lr * min_ratio at the end, with linear warmup to base_lr."""
    warmup = optax.linear_schedule(
        init_value=0.0, end_value=base_lr, transition_steps=warmup_steps
    )
    decay = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=max(1, total_steps - warmup_steps),
        alpha=min_ratio,
    )
    return optax.join_schedules(schedules=[warmup, decay], boundaries=[warmup_steps])


# -------------------------
# Optimizer
# -------------------------
def make_optimizer(cfg: TrainConfig):
    total_steps = cfg.epochs * cfg.steps_per_epoch

    sched = cosine_with_warmup(
        base_lr=cfg.lr,
        total_steps=total_steps,
        warmup_steps=cfg.lr_warmup_steps,
        min_ratio=cfg.lr_min_ratio,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(
            learning_rate=sched,
            b1=cfg.beta1,
            b2=cfg.beta2,
            weight_decay=cfg.weight_decay,
        ),
    )
    return tx


# -------------------------
# One training step (JITed)
# -------------------------
def make_train_step(tx):
    def loss_only(model: Model, batch, key):
        # Uses your act_loss (stablemax variant by default)
        _, total_loss, metrics, _, _ = act_loss(
            model,
            batch=batch,
            key=key,
            loss_type="stablemax_cross_entropy",
        )
        # Optional: you can return metrics as aux if you want to print them
        return total_loss

    loss_and_grad = eqx.filter_value_and_grad(loss_only)

    @jax.jit
    def step(model: Model, opt_state, batch, key):
        loss, grads = loss_and_grad(model, batch, key)
        updates, opt_state = tx.update(grads, opt_state, params=model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return step


# -------------------------
# Main training
# -------------------------
def main():
    # ----- config -----
    B = 64
    T = 5
    V = 11
    cfg = TrainConfig(
        model=ModelConfig(
            batch_size=B,
            seq_len=T,
            task_embedding_dim=512,
            num_task_embeddings=1,
            vocab_size=V,
            H_cycles=3,
            L_cycles=6,
            L_layers=2,
            hidden_size=512,
            expansion=4.0,
            num_heads=8,
            rope_theta=10000.0,
            halt_max_steps=4,
            halt_exploration_prob=0.1,
            forward_dtype="float32",
            task_token_length=1,
        ),
        epochs=1,
        steps_per_epoch=200,
        seed=0,
        lr=3e-4,
        lr_min_ratio=0.1,
        lr_warmup_steps=200,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        max_addend=9,
    )

    # ----- rng, model, optimizer -----
    key = jax.random.PRNGKey(cfg.seed)
    key, key_model = jax.random.split(key)

    model = Model(cfg.model, key=key_model)
    tx = make_optimizer(cfg)
    opt_state = tx.init(model)

    # ----- dataset -----
    data_cfg = AdditionDatasetConfig(
        batch_size=cfg.model.batch_size,
        seq_len=cfg.model.seq_len,
        max_addend=cfg.max_addend,
        vocab_size=cfg.model.vocab_size,
    )
    dataset = AdditionDataset(data_cfg)

    # ----- training loop -----
    train_step = make_train_step(tx)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    for step_idx in range(total_steps):
        key, data_key, step_key = jax.random.split(key, 3)
        batch = dataset.sample(data_key)  # dict with tokens, masks, and task tokens
        model, opt_state, loss = train_step(model, opt_state, batch, step_key)

        # very light progress
        if (step_idx + 1) % 5 == 0:
            print(f"step {step_idx + 1}/{total_steps}  loss={float(loss):.4f}")

    # final summary
    print("Done.")


if __name__ == "__main__":
    main()
