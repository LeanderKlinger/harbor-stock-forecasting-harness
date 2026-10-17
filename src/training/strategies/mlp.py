import torch
import torch.nn as nn

from src.models.config import Config
from src.training.strategies.base import TorchStrategy


_MIN_LAYER_SIZE = 8


class MLPStrategy(TorchStrategy):
    """
    Non-temporal feedforward baseline.

    Flattens the (seq, features) input window into a single vector.
    Tests whether temporal ordering matters — if GRU/xLSTM don't beat this,
    the recurrence isn't adding value.
    """

    def __init__(
        self,
        config: Config,
        hidden_size: int = 128,
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
        return "MLP"

    @property
    def name(self) -> str:
        return f"MLP(seq={self.sequence_length}, h={self.hidden_size}, L={self.num_layers})"

    @property
    def training_signature(self) -> str:
        return f"{super().training_signature}|L={self.num_layers}"

    def _build_model(self, num_features: int) -> nn.Module:
        flat_size = self.sequence_length * num_features
        return _MLPRegressor(flat_size, self.hidden_size, self.num_layers, self.dropout)


class _MLPRegressor(nn.Module):
    def __init__(
        self, flat_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]

        in_size = flat_size
        for i in range(num_layers):
            out_size = max(hidden_size // (2**i), _MIN_LAYER_SIZE)
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = out_size

        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [batch, 1]
