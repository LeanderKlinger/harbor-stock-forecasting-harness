import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig  # pyright: ignore[reportMissingTypeStubs]

from src.models.config import Config
from src.training.strategies.base import TorchStrategy


class MambaStrategy(TorchStrategy):
    """Mamba SSM model — structured state space alternative to recurrent models."""

    def __init__(
        self,
        config: Config,
        hidden_size: int = 64,
        num_layers: int = 2,
        d_state: int = 16,
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
        self.d_state = d_state

    @property
    def model_type(self) -> str:
        return "Mamba"

    @property
    def name(self) -> str:
        return f"Mamba(seq={self.sequence_length}, h={self.hidden_size}, L={self.num_layers})"

    @property
    def training_signature(self) -> str:
        return (
            f"{super().training_signature}|L={self.num_layers}|d_state={self.d_state}"
        )

    def _build_model(self, num_features: int) -> nn.Module:
        mamba_config = MambaConfig(
            d_model=self.hidden_size,
            n_layers=self.num_layers,
            d_state=self.d_state,
        )
        backbone = Mamba(mamba_config)
        return _MambaRegressor(num_features, backbone, self.hidden_size, self.dropout)


class _MambaRegressor(nn.Module):
    """Regression wrapper: input projection -> dropout -> Mamba backbone -> output projection."""

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
        x = self.drop(x)  # Regularize before backbone (MambaConfig has no dropout)
        x = self.backbone(x)  # [batch, seq, hidden]
        x = x[:, -1, :]  # Last timestep
        return self.output_proj(x)  # [batch, 1]
