import pickle
import uuid
from pathlib import Path

from statsmodels.tsa.holtwinters import ExponentialSmoothing  # pyright: ignore[reportMissingTypeStubs]
from statsmodels.tsa.holtwinters.results import HoltWintersResultsWrapper  # pyright: ignore[reportMissingTypeStubs]

from src.models.input import StockDataPairings


class HoltsStrategy:
    """
    Holt's Linear Exponential Smoothing with damped trend.

    Uses additive trend with damping, no seasonality.
    Numeric-only — ignores semantic data.

    Unlike ARIMA, statsmodels Holt-Winters doesn't support append(refit=False),
    so predict() refits on the full history each call (~1ms, negligible).
    """

    def __init__(self) -> None:
        self._all_prices: list[float] = []
        self._fitted_model: HoltWintersResultsWrapper | None = None  # pyright: ignore[reportUnknownVariableType]

    @property
    def model_type(self) -> str:
        return "HoltsLES"

    @property
    def name(self) -> str:
        return "HoltsLES(damped)"

    @property
    def training_signature(self) -> str:
        return "HoltsLES|damped=True"

    @property
    def uses_semantic(self) -> bool:
        return False

    @property
    def parallelizable(self) -> bool:
        return True

    def fit(self, history: list[StockDataPairings]) -> None:
        """Fit on closing prices."""
        self._all_prices = [p.numeric_data.close for p in history]
        self._fit_model()

    def update(self, observation: StockDataPairings) -> None:
        """Extend price history. Model refits on next predict()."""
        self._all_prices.append(observation.numeric_data.close)
        self._fitted_model = None

    def refit(self) -> None:
        """Refit on all observations seen so far."""
        if not self._all_prices:
            raise RuntimeError("No data to refit. Call fit() first.")
        self._fit_model()

    def predict(self, horizon_days: int) -> float:
        """
        Predict return for the given horizon.

        Refits if model was invalidated by update().
        """
        if not self._all_prices:
            raise RuntimeError("No data. Call fit() first.")

        if self._fitted_model is None:
            self._fit_model()

        assert self._fitted_model is not None
        forecast = self._fitted_model.forecast(steps=horizon_days)  # pyright: ignore[reportUnknownMemberType]
        predicted_price = float(forecast[-1])  # pyright: ignore[reportUnknownArgumentType]

        current_price = self._all_prices[-1]
        return (predicted_price - current_price) / current_price

    def cache_fit(self, path: Path) -> None:
        # NOTE: HoltsLES.update() nulls _fitted_model; backtest fires update() at step 0 before
        # any predict(), so the cached model is discarded immediately. Cache saves training
        # time only (~880s → ~10ms on hit), not inference time.
        payload = {"prices": self._all_prices, "model": self._fitted_model}
        # Per-write unique .tmp — see comment in TorchStrategy.cache_fit
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        tmp.replace(path)

    def restore_fit(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)  # noqa: S301 — internal cache, trusted source
        except Exception:
            return False
        self._all_prices = payload["prices"]
        self._fitted_model = payload["model"]
        return True

    def _fit_model(self) -> None:
        model = ExponentialSmoothing(  # pyright: ignore[reportUnknownVariableType]
            self._all_prices,
            trend="add",
            damped_trend=True,
            seasonal=None,
        )
        self._fitted_model = model.fit(optimized=True)  # pyright: ignore[reportUnknownMemberType]
