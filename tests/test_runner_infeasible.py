"""The runner's known-infeasible guard: ARIMA + "auto" cadence + h=1 skips up
front (deterministic LU failure) instead of burning 30-90 min per
guaranteed failure. Explicit cadence=1 must still run (escape hatch)."""

from datetime import datetime

from src.experiments.runner import _is_known_infeasible  # pyright: ignore[reportPrivateUsage]
from src.models.config import Config


def _config(cadence: int | str | None, horizon: int) -> Config:
    return Config(
        training_start=datetime(2020, 1, 1),
        training_end=datetime(2022, 12, 31),
        test_start=datetime(2023, 1, 1),
        test_end=datetime(2023, 3, 31),
        rebalance_horizon_days=horizon,
        retraining_cadence_days=cadence,  # pyright: ignore[reportArgumentType]
        stock_ids=["AMD"],
    )


def test_arima_auto_h1_is_infeasible() -> None:
    assert _is_known_infeasible("ARIMA", _config("auto", 1)) is True


def test_explicit_daily_cadence_is_allowed() -> None:
    assert _is_known_infeasible("ARIMA", _config(1, 1)) is False


def test_arima_auto_other_horizons_allowed() -> None:
    assert _is_known_infeasible("ARIMA", _config("auto", 5)) is False
    assert _is_known_infeasible("ARIMA", _config("auto", 20)) is False


def test_other_models_never_skipped() -> None:
    for model_type in ("HoltsLES", "xLSTM", "GRU", "MLP", "Momentum"):
        assert _is_known_infeasible(model_type, _config("auto", 1)) is False
