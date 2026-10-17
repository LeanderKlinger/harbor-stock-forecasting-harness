from pathlib import Path

import numpy as np

from src.evaluation.persistence import Persistence
from src.models.evaluation import PipelineEvaluation
from src.models.input import PipelineInput
from src.models.output import Direction, PipelineOutput


class EvaluationModule:
    """Stateless orchestrator — owns a Persistence instance but stores NO
    per-run state. Safe to reuse across runs in a loop."""

    def __init__(self) -> None:
        self.persistence = Persistence()

    def evaluate(self, output: PipelineOutput) -> PipelineEvaluation:
        """Calculate all evaluation metrics from pipeline output. Pure function."""
        return _evaluate(output)

    def persist(
        self,
        pipeline_input: PipelineInput,
        output: PipelineOutput,
        evaluation: PipelineEvaluation,
    ) -> Path:
        """Delegates to self.persistence.persist_run()."""
        return self.persistence.persist_run(pipeline_input, output, evaluation)


def _evaluate(output: PipelineOutput) -> PipelineEvaluation:
    """Calculate all evaluation metrics from pipeline output."""

    # Error-based metrics from prediction records
    predictions = output.prediction_records
    if predictions:
        errors = [p.predicted_return - p.actual_return for p in predictions]
        rmse = float(np.sqrt(np.mean([e**2 for e in errors])))
        mae = float(np.mean([abs(e) for e in errors]))
        # MAPE — skip zero actual returns to avoid division by zero
        ape = [
            abs(e) / abs(p.actual_return)
            for e, p in zip(errors, predictions)
            if p.actual_return != 0.0
        ]
        mape = float(np.mean(ape)) if ape else 0.0
    else:
        rmse, mae, mape = 0.0, 0.0, 0.0

    # Directional accuracy and confusion matrix
    tp = tn = fp = fn = 0
    for p in predictions:
        pred_up = p.predicted_direction == Direction.UP
        actual_up = p.actual_direction == Direction.UP

        if pred_up and actual_up:
            tp += 1
        elif not pred_up and not actual_up:
            tn += 1
        elif pred_up and not actual_up:
            fp += 1
        else:
            fn += 1

    total_predictions = tp + tn + fp + fn
    directional_accuracy = (
        (tp + tn) / total_predictions if total_predictions > 0 else 0.0
    )

    # Return-based metrics from snapshots
    snapshots = output.snapshot_records
    daily_returns = [s.daily_return for s in snapshots] if snapshots else []

    total_return = (
        output.final_capital - output.initial_capital
    ) / output.initial_capital

    # Annualized return (assuming output.config.trading_days trading days)
    n_days = len(daily_returns)
    annualized_return = (
        ((1 + total_return) ** (output.config.trading_days / n_days) - 1)
        if n_days > 0
        else 0.0
    )

    # Volatility
    volatility = (
        float(np.std(daily_returns) * np.sqrt(output.config.trading_days))
        if daily_returns
        else 0.0
    )

    # Downside volatility (only negative returns)
    negative_returns = [r for r in daily_returns if r < 0]
    downside_volatility = (
        float(np.std(negative_returns) * np.sqrt(output.config.trading_days))
        if negative_returns
        else 0.0
    )

    # Sharpe ratio
    daily_rf = output.config.risk_free_rate / output.config.trading_days
    excess_returns = [r - daily_rf for r in daily_returns]
    sharpe_ratio = (
        float(
            np.mean(excess_returns)
            / np.std(excess_returns)
            * np.sqrt(output.config.trading_days)
        )
        if excess_returns and np.std(excess_returns) > 0
        else 0.0
    )

    # Sortino ratio (uses downside deviation)
    downside_excess = [r - daily_rf for r in daily_returns if r < daily_rf]
    downside_std = float(np.std(downside_excess)) if downside_excess else 0.0
    sortino_ratio = (
        float(
            np.mean(excess_returns) / downside_std * np.sqrt(output.config.trading_days)
        )
        if downside_std > 0
        else 0.0
    )

    # Maximum drawdown
    max_drawdown = 0.0
    for s in snapshots:
        drawdown = (
            (s.peak_value - s.portfolio_value) / s.peak_value
            if s.peak_value > 0
            else 0.0
        )
        max_drawdown = max(max_drawdown, drawdown)

    # CVaR (Conditional Value at Risk) at 95% confidence
    if daily_returns:
        sorted_returns = sorted(daily_returns)
        var_index = int(len(sorted_returns) * 0.05)
        cvar = float(
            -np.mean(sorted_returns[: max(var_index, 1)])
            if var_index > 0
            else -sorted_returns[0]
        )
    else:
        cvar = 0.0

    # Trade statistics
    trades = output.trade_records
    total_trades = len(trades)
    total_transaction_cost = sum(t.transaction_cost for t in trades)

    # Win ratio and avg trade return (pair buys with subsequent sells per stock)
    trade_returns: list[float] = []
    open_positions: dict[
        str, list[tuple[float, float]]
    ] = {}  # stock_id -> [(price, qty)]

    for t in sorted(trades, key=lambda x: x.timestamp):
        if t.action.value == 1:  # BUY
            if t.stock_id not in open_positions:
                open_positions[t.stock_id] = []
            open_positions[t.stock_id].append((t.execution_price, t.quantity))
        elif t.action.value == -1:  # SELL
            if t.stock_id in open_positions and open_positions[t.stock_id]:
                buy_price, _ = open_positions[t.stock_id].pop(0)
                trade_return = (t.execution_price - buy_price) / buy_price
                trade_returns.append(trade_return)

    winning_trades = sum(1 for r in trade_returns if r > 0)
    win_ratio = winning_trades / len(trade_returns) if trade_returns else 0.0
    avg_trade_return = float(np.mean(trade_returns)) if trade_returns else 0.0

    return PipelineEvaluation(
        rmse=rmse,
        mae=mae,
        mape=mape,
        directional_accuracy=directional_accuracy,
        fp=fp,
        fn=fn,
        tp=tp,
        tn=tn,
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        CVar=cvar,
        maximum_drawdown=max_drawdown,
        volatility=volatility,
        downside_volatility=downside_volatility,
        total_trades=total_trades,
        win_ratio=win_ratio,
        avg_trade_return=avg_trade_return,
        total_transaction_cost=total_transaction_cost,
    )
