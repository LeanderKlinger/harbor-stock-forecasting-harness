from src.models.input import StockDataPairings


class BuyAndHoldStrategy:
    """
    Always predicts positive return for every stock.

    The experiment runner overrides top_k to stock count and sets a
    rebalance horizon spanning the full test period, so the backtester
    buys all stocks equally on day 0 and never rebalances again.
    """

    def __init__(self) -> None:
        self._history: list[StockDataPairings] = []

    def fit(self, history: list[StockDataPairings]) -> None:
        self._history = list(history)

    def predict(self, horizon_days: int) -> float:
        return 1.0

    def update(self, observation: StockDataPairings) -> None:
        self._history.append(observation)

    def refit(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return "BuyAndHold"

    @property
    def name(self) -> str:
        return "BuyAndHold"

    @property
    def training_signature(self) -> str:
        return "BuyAndHold"

    @property
    def uses_semantic(self) -> bool:
        return False

    @property
    def parallelizable(self) -> bool:
        return False


class EqualWeightStrategy:
    """
    Always predicts positive return for every stock.

    The experiment runner overrides top_k to stock count so the backtester
    rebalances ALL stocks to equal weight every period. Isolates the
    rebalancing effect from prediction quality.
    """

    def __init__(self) -> None:
        self._history: list[StockDataPairings] = []

    def fit(self, history: list[StockDataPairings]) -> None:
        self._history = list(history)

    def predict(self, horizon_days: int) -> float:
        return 1.0

    def update(self, observation: StockDataPairings) -> None:
        self._history.append(observation)

    def refit(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return "EqualWeight"

    @property
    def name(self) -> str:
        return "EqualWeight"

    @property
    def training_signature(self) -> str:
        return "EqualWeight"

    @property
    def uses_semantic(self) -> bool:
        return False

    @property
    def parallelizable(self) -> bool:
        return False


class MomentumStrategy:
    """
    Predicts future return as past-N-day simple return.

    Pure price momentum baseline — no model, no semantic data.
    """

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback
        self._prices: list[float] = []

    def fit(self, history: list[StockDataPairings]) -> None:
        self._prices = [p.numeric_data.close for p in history]

    def predict(self, horizon_days: int) -> float:
        if len(self._prices) < self.lookback:
            return 0.0
        current = self._prices[-1]
        past = self._prices[-self.lookback]
        return (current - past) / past

    def update(self, observation: StockDataPairings) -> None:
        self._prices.append(observation.numeric_data.close)

    def refit(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return "Momentum"

    @property
    def name(self) -> str:
        return f"Momentum(lookback={self.lookback})"

    @property
    def training_signature(self) -> str:
        return f"Momentum|lookback={self.lookback}"

    @property
    def uses_semantic(self) -> bool:
        return False

    @property
    def parallelizable(self) -> bool:
        return False
