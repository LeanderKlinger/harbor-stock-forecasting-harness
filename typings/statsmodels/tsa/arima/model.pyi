"""Minimal type stubs for statsmodels ARIMA model."""

from typing import Sequence
import numpy as np
import numpy.typing as npt

class ARIMAResultsWrapper:
    """Results from fitting an ARIMA model."""

    def forecast(self, steps: int = 1) -> npt.NDArray[np.float64]:
        """Forecast future values."""
        ...

    def append(
        self,
        endog: Sequence[float],
        refit: bool = True,
    ) -> ARIMAResultsWrapper:
        """Append new observations and optionally refit."""
        ...

class ARIMA:
    """ARIMA model."""

    def __init__(
        self,
        endog: Sequence[float],
        order: tuple[int, int, int] = (0, 0, 0),
    ) -> None: ...
    def fit(self) -> ARIMAResultsWrapper:
        """Fit the ARIMA model."""
        ...
