import contextlib
import pickle
import uuid
import warnings
from collections.abc import Generator
from pathlib import Path

from statsmodels.tools.sm_exceptions import ConvergenceWarning  # pyright: ignore[reportMissingTypeStubs]
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper

from src.models.input import StockDataPairings


@contextlib.contextmanager
def _silence_arima_fit_noise() -> Generator[None]:
    """Suppress statsmodels ARIMA fit warnings that are informational only.

    ConvergenceWarning (MLE didn't hit gradient tolerance) and the
    "Non-invertible/Non-stationary starting parameters" UserWarnings don't
    affect forecast usability — statsmodels returns last-iteration params.
    LU decomposition errors still raise as exceptions; this only silences warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message="Non-invertible starting MA")
        warnings.filterwarnings("ignore", message="Non-stationary starting")
        yield


class ARIMAStrategy:
    """
    ARIMA-based prediction model.

    Trains on closing prices and forecasts future prices,
    converting to returns for the prediction interface.

    Supports two modes:
    - update(): Extend observations without retraining (append with refit=False)
    - refit(): Retrain on all observations seen so far

    Note: ARIMA only uses numeric data from StockDataPairings.
    """

    def __init__(self, order: tuple[int, int, int] = (5, 1, 2)):
        self.order = order
        self._fitted_model: ARIMAResultsWrapper | None = None
        self._all_prices: list[float] = []

    def fit(self, history: list[StockDataPairings]) -> None:
        """Fit ARIMA model on closing prices."""
        self._all_prices = [p.numeric_data.close for p in history]

        model = ARIMA(self._all_prices, order=self.order)
        with _silence_arima_fit_noise():
            self._fitted_model = model.fit()

    def update(self, observation: StockDataPairings) -> None:
        """
        Extend model with new observation without retraining.

        Uses statsmodels append(refit=False) to extend the series
        while keeping the original fitted parameters.
        """
        if self._fitted_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        new_price = observation.numeric_data.close
        self._all_prices.append(new_price)

        # Extend model without refitting parameters
        self._fitted_model = self._fitted_model.append([new_price], refit=False)

    def refit(self) -> None:
        """
        Retrain model on all observations seen so far.

        Use this when daily retraining mode is enabled.
        """
        if not self._all_prices:
            raise RuntimeError("No data to refit. Call fit() first.")

        model = ARIMA(self._all_prices, order=self.order)
        with _silence_arima_fit_noise():
            self._fitted_model = model.fit()

    def predict(self, horizon_days: int) -> float:
        """
        Predict return for the given horizon.

        ARIMA forecasts price at t+horizon. Return is calculated
        relative to the most recent observed price.
        """
        if self._fitted_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        forecast = self._fitted_model.forecast(steps=horizon_days)
        predicted_price = float(forecast[-1])

        current_price = self._all_prices[-1]
        return (predicted_price - current_price) / current_price

    def cache_fit(self, path: Path) -> None:
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

    @property
    def model_type(self) -> str:
        return "ARIMA"

    @property
    def name(self) -> str:
        return f"ARIMA{self.order}"

    @property
    def training_signature(self) -> str:
        return f"ARIMA|order={self.order}"

    @property
    def uses_semantic(self) -> bool:
        return False

    @property
    def parallelizable(self) -> bool:
        return True
