from typing import Any

import torch
import torch.nn as nn
from mambapy.jamba import Jamba, JambaLMConfig  # pyright: ignore[reportMissingTypeStubs]

from src.models.config import Config
from src.training.strategies.base import TorchStrategy


class JambaStrategy(TorchStrategy):
    """Jamba hybrid SSM+Attention model — Mamba layers interleaved with attention."""

    def __init__(
        self,
        config: Config,
        hidden_size: int = 64,
        num_layers: int = 4,
        num_attention_heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        super().__init__(
            config,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            dropout=dropout,
        )
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads

    @property
    def model_type(self) -> str:
        return "Jamba"

    @property
    def name(self) -> str:
        return (
            f"Jamba(seq={self.sequence_length}, h={self.hidden_size}, "
            f"L={self.num_layers}, heads={self.num_attention_heads})"
        )

    @property
    def training_signature(self) -> str:
        return f"{super().training_signature}|L={self.num_layers}|heads={self.num_attention_heads}"

    def _build_model(self, num_features: int) -> nn.Module:
        jamba_config = JambaLMConfig(
            d_model=self.hidden_size,
            n_layers=self.num_layers,
            mlp_size=self.hidden_size * 2,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,
            num_experts=1,  # No MoE — keep it simple
            attn_layer_offset=1,  # Attention at layers 1, 3, 5, ...
            attn_layer_period=2,  # Every other layer
            vocab_size=1,  # Unused — we bypass the LM head
            attention_dropout=self.dropout,
        )
        backbone = Jamba(jamba_config)
        return _JambaRegressor(num_features, backbone, self.hidden_size, self.dropout)


class _JambaRegressor(nn.Module):
    """Regression wrapper: input projection -> dropout -> Jamba backbone -> output projection."""

    def __init__(
        self, input_size: int, backbone: nn.Module, hidden_size: int, dropout: float
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.backbone = backbone
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)  # [batch, seq, hidden]
        x = self.drop(x)  # Regularize before backbone (SSM layers have no dropout)
        out: Any = self.backbone(x)  # Returns (tensor, router_logits)
        x = out[0]  # [batch, seq, hidden]
        x = x[:, -1, :]  # Last timestep
        return self.output_proj(x)  # [batch, 1]
