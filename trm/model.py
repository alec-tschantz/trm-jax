from typing import Tuple, List, Dict
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel


from trm.utils import trunc_normal_init_
from trm.layers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CastedEmbedding,
    CastedLinear,
    CosSin,
)
from trm.sparse_embedding import CastedSparseEmbedding


@dataclass
class InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class Carry:
    inner_carry: InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


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
    forward_dtype: str = "bfloat16"
    puzzle_emb_len: int = 16


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        out = self.mlp(hidden_states)
        return rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)


class ReasoningModule(nn.Module):
    def __init__(self, layers: List[Block]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class Inner(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.q_head = CastedLinear(self.config.hidden_size, 1, bias=True)
        self.puzzle_emb_len = self.config.puzzle_emb_len
        self.puzzle_emb = CastedSparseEmbedding(
            self.config.num_puzzle_identifiers,
            self.config.puzzle_emb_ndim,
            batch_size=self.config.batch_size,
            init_std=0,
            cast_to=self.forward_dtype,
        )
        self.rotary_emb = RotaryEmbedding(
            dim=self.config.hidden_size // self.config.num_heads,
            max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
            base=self.config.rope_theta,
        )
        self.L_level = ReasoningModule(
            layers=[Block(self.config) for _i in range(self.config.L_layers)]
        )
        h_init_tensor = trunc_normal_init_(
            torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
        )
        self.register_buffer("H_init", h_init_tensor)
        l_init_tensor = trunc_normal_init_(
            torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
        )
        self.register_buffer("L_init", l_init_tensor)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))
        puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
        pad_count = (
            self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
        )
        if pad_count > 0:
            puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
        embedding = torch.cat(
            (
                puzzle_embedding.view(
                    -1, self.puzzle_emb_len, self.config.hidden_size
                ),
                embedding,
            ),
            dim=-2,
        )
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: InnerCarry):
        return InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[InnerCarry, torch.Tensor, torch.Tensor]:
        seq_info = dict(cos_sin=self.rotary_emb())
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )
        z_H, z_L = carry.z_H, carry.z_L
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_halt_logits = self.q_head(z_H[:, 0]).to(torch.float32).squeeze(-1)
        return new_carry, output, q_halt_logits


class Model(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ModelConfig(**config_dict)
        self.inner = Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }
        new_inner_carry, logits, q_halt_logits = self.inner(
            new_inner_carry, new_current_data
        )
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
        }
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step
            if self.training and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > 0)
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                ) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)
        return (
            Carry(new_inner_carry, new_steps, halted, new_current_data),
            outputs,
        )
