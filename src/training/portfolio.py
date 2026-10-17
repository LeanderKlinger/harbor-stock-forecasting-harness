from dataclasses import dataclass, field
from datetime import datetime
from src.models.output import Action, TradeRecord, PortfolioSnapshot


@dataclass
class PortfolioState:
    """
    Tracks portfolio state during walk-forward backtesting.

    Handles:
    - Cash and position tracking
    - Trade execution with transaction costs
    - Rebalancing to target allocations
    - Snapshot generation for evaluation
    """

    cash: float
    positions: dict[str, float] = field(default_factory=lambda: {})
    peak_value: float = 0.0
    _previous_value: float = 0.0
    _initial_capital: float = 0.0

    def __post_init__(self) -> None:
        if self._initial_capital == 0.0:
            self._initial_capital = self.cash
        if self.peak_value == 0.0:
            self.peak_value = self.cash
        if self._previous_value == 0.0:
            self._previous_value = self.cash

    def total_value(self, prices: dict[str, float]) -> float:
        """Calculate total portfolio value at current prices."""
        position_value = 0.0
        for stock_id, qty in self.positions.items():
            if qty == 0.0:
                continue
            position_value += qty * prices[stock_id]
        return self.cash + position_value

    def execute_trade(
        self,
        stock_id: str,
        action: Action,
        price: float,
        transaction_cost_pct: float,
        timestamp: datetime,
        amount: float | None = None,
    ) -> TradeRecord | None:
        """
        Execute a trade and update portfolio state.

        BUY: spend `amount` dollars (or all cash if None) on the stock.
        SELL: sell `amount` dollars worth (or entire position if None).
        HOLD: no-op, returns None.
        """
        if action == Action.HOLD:
            return None

        if action == Action.BUY:
            buy_budget = amount if amount is not None else self.cash
            buy_budget = min(buy_budget, self.cash)

            if buy_budget <= 0:
                return None

            cost = buy_budget * transaction_cost_pct
            available = buy_budget - cost
            quantity = available / price

            self.cash -= buy_budget
            self.positions[stock_id] = self.positions.get(stock_id, 0.0) + quantity

            return TradeRecord(
                timestamp=timestamp,
                stock_id=stock_id,
                action=action,
                quantity=quantity,
                execution_price=price,
                transaction_cost=cost,
            )

        if action == Action.SELL:
            held_qty = self.positions.get(stock_id, 0.0)
            if held_qty <= 0:
                return None

            if amount is not None:
                # Partial sell: sell dollar amount worth of shares
                sell_qty = min(amount / price, held_qty)
            else:
                sell_qty = held_qty

            gross_proceeds = sell_qty * price
            cost = gross_proceeds * transaction_cost_pct
            net_proceeds = gross_proceeds - cost

            self.positions[stock_id] = held_qty - sell_qty
            self.cash += net_proceeds

            return TradeRecord(
                timestamp=timestamp,
                stock_id=stock_id,
                action=action,
                quantity=sell_qty,
                execution_price=price,
                transaction_cost=cost,
            )

        return None

    def rebalance(
        self,
        target_allocations: dict[str, float],
        prices: dict[str, float],
        transaction_cost_pct: float,
        timestamp: datetime,
    ) -> list[TradeRecord]:
        """
        Delta-based rebalance: only trade the difference.

        1. Full-sell positions not in targets
        2. Partial-sell overweight positions
        3. Buy into underweight / new positions

        Weights sum to <= 1.0; remainder stays as cash.
        """
        trades: list[TradeRecord] = []
        total_value = self.total_value(prices)

        # 1. Full-sell stocks no longer in targets
        for stock_id in list(self.positions.keys()):
            if self.positions[stock_id] > 0 and stock_id not in target_allocations:
                trade = self.execute_trade(
                    stock_id=stock_id,
                    action=Action.SELL,
                    price=prices[stock_id],
                    transaction_cost_pct=transaction_cost_pct,
                    timestamp=timestamp,
                )
                if trade is not None:
                    trades.append(trade)

        # 2. Partial-sell overweight positions (frees cash for buys)
        for stock_id, target_weight in target_allocations.items():
            held_qty = self.positions.get(stock_id, 0.0)
            if held_qty <= 0:
                continue
            current_value = held_qty * prices[stock_id]
            target_value = total_value * target_weight
            excess = current_value - target_value
            if excess > 0:
                trade = self.execute_trade(
                    stock_id=stock_id,
                    action=Action.SELL,
                    price=prices[stock_id],
                    transaction_cost_pct=transaction_cost_pct,
                    timestamp=timestamp,
                    amount=excess,
                )
                if trade is not None:
                    trades.append(trade)

        # 3. Buy underweight / new positions
        for stock_id, target_weight in target_allocations.items():
            held_qty = self.positions.get(stock_id, 0.0)
            current_value = held_qty * prices[stock_id]
            target_value = total_value * target_weight
            deficit = target_value - current_value
            if deficit > 0:
                trade = self.execute_trade(
                    stock_id=stock_id,
                    action=Action.BUY,
                    price=prices[stock_id],
                    transaction_cost_pct=transaction_cost_pct,
                    timestamp=timestamp,
                    amount=deficit,
                )
                if trade is not None:
                    trades.append(trade)

        return trades

    def snapshot(
        self,
        timestamp: datetime,
        prices: dict[str, float],
    ) -> PortfolioSnapshot:
        """Generate a PortfolioSnapshot for current state."""
        current_value = self.total_value(prices)

        daily_return = (
            (current_value - self._previous_value) / self._previous_value
            if self._previous_value > 0
            else 0.0
        )

        self.peak_value = max(self.peak_value, current_value)

        cumulative_return = (
            current_value - self._initial_capital
        ) / self._initial_capital

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            portfolio_value=current_value,
            daily_return=daily_return,
            positions=dict(self.positions),
            free_cash=self.cash,
            cumulative_return=cumulative_return,
            peak_value=self.peak_value,
        )

        self._previous_value = current_value
        return snapshot
