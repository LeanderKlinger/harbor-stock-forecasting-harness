import abc
import copy
import uuid
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.config import Config
from src.models.input import StockDataPairings

# Fraction of training data withheld for validation when early stopping is enabled
_VALIDATION_FRACTION = 0.1


class TorchStrategy(abc.ABC):
    """
    Abstract base for PyTorch-based prediction models.

    Subclasses only need to implement _build_model() and name.
    All models in a run share config.sequence_length and
    config.rebalance_horizon_days for fair architectural comparison.

    Features (width set by config.feature_set):
      close_only (1): close
      numeric (7): close, volume, log_return, volatility, macd, ret_1w, ret_2w
      full (16): numeric + news_count, avg_sentiment, variance, has_news,
                 days_since_news, sentiment_emas (per config), momentum
    """

    def __init__(
        self,
        config: Config,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        dropout: float = 0.2,
    ):
        self.config = config
        self.sequence_length = config.sequence_length
        self.prediction_horizon = config.rebalance_horizon_days
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

        self._model: nn.Module | None = None
        self._device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._history: list[StockDataPairings] = []
        self._feature_mean: torch.Tensor | None = None
        self._feature_std: torch.Tensor | None = None

    @property
    def uses_semantic(self) -> bool:
        return self.config.feature_set == "full"

    @property
    def parallelizable(self) -> bool:
        return True

    @property
    @abc.abstractmethod
    def model_type(self) -> str: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    def training_signature(self) -> str:
        # Subclasses with extra training-affecting knobs override and append:
        #   return f"{super().training_signature}|L={self.num_layers}"
        # features= is emitted unconditionally like every other knob: a uniform
        # "every knob always in the signature" beats an omit-when-default special
        # case that every future knob would inherit.
        return (
            f"{self.model_type}"
            f"|seq={self.sequence_length}"
            f"|h={self.hidden_size}"
            f"|lr={self.learning_rate}"
            f"|epochs={self.epochs}"
            f"|batch={self.batch_size}"
            f"|dropout={self.dropout}"
            f"|features={self.config.feature_set}"
        )

    @abc.abstractmethod
    def _build_model(self, num_features: int) -> nn.Module: ...

    def _extract_features(self, pairing: StockDataPairings) -> list[float]:
        """Extract features from a single pairing; width follows config.feature_set."""
        n = pairing.numeric_data
        if self.config.feature_set == "close_only":
            return [n.close]

        features: list[float] = [
            n.close,
            float(n.volume),
            n.log_return,
            n.rolling_volatility_1w,
            n.macd,
            n.annualized_return_1w,
            n.annualized_return_2w,
        ]
        if self.config.feature_set == "numeric":
            return features

        s = pairing.semantic_data
        if s is None:
            raise ValueError(
                f"semantic_data is None for {pairing.stock_id} @ {pairing.timestamp}. "
                f"enrich_temporal_features() must run before training."
            )

        features.extend(
            [
                float(s.news_count),
                s.average_sentiment,
                s.sentiment_variance,
                float(s.has_news),
                float(s.days_since_news),
            ]
        )
        features.extend(s.sentiment_emas)
        features.append(s.sentiment_momentum)

        return features

    def _build_sequences(
        self,
        pairings: list[StockDataPairings],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build (x, y) tensors for training.

        x: [num_samples, sequence_length, num_features]
        y: [num_samples, 1] — return at t + prediction_horizon
        """
        features = [self._extract_features(p) for p in pairings]

        x_list: list[list[list[float]]] = []
        y_list: list[list[float]] = []

        horizon = self.prediction_horizon
        for i in range(len(features) - self.sequence_length - horizon):
            seq = features[i : i + self.sequence_length]

            current_close = pairings[i + self.sequence_length - 1].numeric_data.close
            future_close = pairings[
                i + self.sequence_length - 1 + horizon
            ].numeric_data.close
            target = (future_close - current_close) / current_close

            x_list.append(seq)
            y_list.append([target])

        return (
            torch.tensor(x_list, dtype=torch.float32),
            torch.tensor(y_list, dtype=torch.float32),
        )

    def _normalize(self, X: torch.Tensor, fit: bool = False) -> torch.Tensor:
        """Z-score normalization. Stats frozen from training data."""
        if fit:
            self._feature_mean = X.mean(dim=(0, 1), keepdim=True)
            self._feature_std = X.std(dim=(0, 1), keepdim=True) + 1e-8
        assert self._feature_mean is not None and self._feature_std is not None
        return (X - self._feature_mean) / self._feature_std

    def fit(self, history: list[StockDataPairings]) -> None:
        """Train on historical data. Model is frozen after training."""
        self._history = list(history)

        min_required = self.sequence_length + self.prediction_horizon + 1
        if len(history) < min_required:
            raise ValueError(
                f"Need at least {min_required} samples "
                f"(seq={self.sequence_length} + horizon={self.prediction_horizon} + 1), "
                f"got {len(history)}"
            )

        x_all, y_all = self._build_sequences(history)
        x_all = self._normalize(x_all, fit=True)

        # Split into train/val chronologically when early stopping is enabled
        patience = self.config.early_stopping_patience
        x_val: torch.Tensor | None = None
        y_val: torch.Tensor | None = None
        if patience is not None:
            n_total = x_all.shape[0]
            n_val = max(1, int(n_total * _VALIDATION_FRACTION))
            n_train = n_total - n_val
            x_train = x_all[:n_train]
            y_train = y_all[:n_train]
            x_val = x_all[n_train:].to(self._device)
            y_val = y_all[n_train:].to(self._device)
        else:
            x_train = x_all
            y_train = y_all

        self._model = self._build_model(num_features=int(x_train.shape[2]))
        self._model.to(self._device)

        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        patience_counter = 0
        self._actual_epochs = 0

        for epoch in range(self.epochs):
            self._model.train()
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                pred = self._model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()  # pyright: ignore[reportUnknownMemberType]

            self._actual_epochs = epoch + 1

            if patience is not None and x_val is not None and y_val is not None:
                # Evaluate on validation set
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(x_val)
                    val_loss = loss_fn(val_pred, y_val).item()

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(self._model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        # Restore best weights if early stopping was used
        if best_state is not None:
            self._model.load_state_dict(best_state)

        # Freeze after training
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

    def update(self, observation: StockDataPairings) -> None:
        """Extend history with new observation. Model stays frozen."""
        self._history.append(observation)

    def refit(self) -> None:
        """No-op for frozen mode."""

    def cache_fit(self, path: Path) -> None:
        assert self._model is not None
        # Save state_dict tensors as CPU so the cache file is device-agnostic and portable
        # across machines. The restoring instance moves the model to its own device after load.
        payload = {
            "state_dict": {k: v.cpu() for k, v in self._model.state_dict().items()},
            "feature_mean": self._feature_mean,  # already CPU (built from torch.tensor in fit())
            "feature_std": self._feature_std,  # already CPU
            "history": self._history,
        }
        # Per-write unique .tmp suffix so two concurrent workers writing the same key
        # don't truncate each other's tmp file mid-write. os.replace is last-writer-wins
        # on the final path — fine since artifacts are byte-identical for the same key.
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        torch.save(payload, tmp)  # pyright: ignore[reportUnknownMemberType]
        tmp.replace(path)  # atomic on POSIX

    def restore_fit(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            # No map_location: payload was saved with all tensors on CPU. Model is then
            # moved to self._device, mirroring the end state of fit() exactly.
            # weights_only=False: history holds Pydantic StockDataPairings.
            # Safe — we only load files we wrote.
            payload = torch.load(path, weights_only=False)  # pyright: ignore[reportUnknownMemberType]
        except Exception:
            return False
        num_features = int(payload["feature_mean"].shape[-1])
        self._model = self._build_model(num_features)
        self._model.load_state_dict(payload["state_dict"])
        self._model.to(self._device)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False
        self._feature_mean = payload["feature_mean"]
        self._feature_std = payload["feature_std"]
        self._history = payload["history"]
        return True

    def predict(self, horizon_days: int) -> float:
        """Predict return using most recent sequence."""
        assert horizon_days == self.prediction_horizon, (
            f"Model trained for horizon={self.prediction_horizon}, "
            f"called with horizon_days={horizon_days}"
        )
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if len(self._history) < self.sequence_length:
            raise RuntimeError(
                f"Need {self.sequence_length} observations, have {len(self._history)}"
            )

        recent = self._history[-self.sequence_length :]
        features = [self._extract_features(p) for p in recent]

        x_input = torch.tensor([features], dtype=torch.float32)
        x_input = self._normalize(x_input, fit=False)
        x_input = x_input.to(self._device)

        with torch.no_grad():
            pred = self._model(x_input)

        return float(pred[0, 0].cpu())
