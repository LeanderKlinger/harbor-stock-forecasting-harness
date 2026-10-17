import torch
import torch.nn as nn

from src.models.config import Config
from src.training.strategies.base import TorchStrategy


class GRUStrategy(TorchStrategy):
    """GRU recurrent model — direct comparison baseline for xLSTM."""

    def __init__(
        self,
        config: Config,
        hidden_size: int = 64,
        num_layers: int = 2,
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

    @property
    def model_type(self) -> str:
        return "GRU"

    @property
    def name(self) -> str:
        return f"GRU(seq={self.sequence_length}, h={self.hidden_size}, L={self.num_layers})"

    @property
    def training_signature(self) -> str:
        return f"{super().training_signature}|L={self.num_layers}"

    def _build_model(self, num_features: int) -> nn.Module:
        return _GRURegressor(
            num_features, self.hidden_size, self.num_layers, self.dropout
        )


class _GRURegressor(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)  # [batch, seq, hidden]
        last = out[:, -1, :]  # Last timestep
        return self.output_proj(last)  # [batch, 1]
