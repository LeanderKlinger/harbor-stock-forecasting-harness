from typing import Protocol

from src.models.input import StockDataPairings


class PredictionModel(Protocol):
    """
    Interface for all prediction models.

    Walk-forward usage:
        1. fit() on training data
        2. For each test day:
           a. predict() for horizon
           b. update() with actual observation (extends context without retraining)
        3. Optional: refit() to retrain on extended data

    Models receive StockDataPairings which contain both numeric and optional
    semantic data. Models that only use numeric data (e.g., ARIMA) simply
    ignore the semantic_data field.
    """

    @property
    def model_type(self) -> str:
        """Clean model type name for grouping (e.g. 'xLSTM', 'ARIMA')."""
        ...

    @property
    def name(self) -> str:
        """Full parametric model name (e.g. 'xLSTM(seq=60, h=64)'). Used by FE display."""
        ...

    @property
    def training_signature(self) -> str:
        """Stable identifier for caching: must encode every training-affecting hyperparam.

        Forgetting a knob → silent cache poisoning for numerical-only changes,
        loud RuntimeError on shape mismatch for structural changes.
        """
        ...

    @property
    def uses_semantic(self) -> bool:
        """Whether this model consumes semantic features."""
        ...

    @property
    def parallelizable(self) -> bool:
        """Whether per-stock training benefits from a process pool.

        True for models with non-trivial fit cost. False for microsecond-fit baselines
        where pool overhead dominates.
        """
        ...

    def fit(self, history: list[StockDataPairings]) -> None:
        """Train model on historical data (numeric + optional semantic)."""
        ...

    def predict(self, horizon_days: int) -> float:
        """
        Predict return for the given horizon.

        Returns:
            Predicted return as a decimal (e.g., 0.02 = 2% return).
        """
        ...

    def update(self, observation: StockDataPairings) -> None:
        """
        Extend model with new observation without retraining.

        This is analogous to providing context at inference time.
        The model sees the new data for forecasting but doesn't
        update its learned parameters.
        """
        ...

    def refit(self) -> None:
        """
        Retrain model on all observations seen so far.

        Call this instead of update() when retraining mode is enabled.
        """
        ...
