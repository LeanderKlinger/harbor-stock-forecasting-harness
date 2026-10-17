# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false
import torch
import torch.nn as nn
from xlstm import mLSTMBlockConfig, xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig

from src.models.config import Config
from src.training.strategies.base import TorchStrategy


class XLSTMStrategy(TorchStrategy):
    """xLSTM hybrid model — NX-AI xLSTM v2, native PyTorch kernels (MPS compatible)."""

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
        self.num_blocks = num_layers

    @property
    def model_type(self) -> str:
        return "xLSTM"

    @property
    def name(self) -> str:
        return f"xLSTM(seq={self.sequence_length}, h={self.hidden_size})"

    @property
    def training_signature(self) -> str:
        return f"{super().training_signature}|blocks={self.num_blocks}"

    def _build_model(self, num_features: int) -> nn.Module:
        layer_config = mLSTMLayerConfig(num_heads=4)
        block_config = mLSTMBlockConfig(mlstm=layer_config)

        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=block_config,
            context_length=self.sequence_length,
            num_blocks=self.num_blocks,
            embedding_dim=self.hidden_size,
            dropout=self.dropout,
        )

        backbone = xLSTMBlockStack(xlstm_config)
        return _XLSTMRegressor(num_features, backbone, self.hidden_size)


class _XLSTMRegressor(nn.Module):
    """Regression wrapper: input projection -> xLSTM backbone -> output projection."""

    def __init__(self, input_size: int, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.backbone = backbone
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)  # [batch, seq, hidden]
        x = self.backbone(x)  # [batch, seq, hidden]
        x = x[:, -1, :]  # Take last timestep
        return self.output_proj(x)  # [batch, 1]
