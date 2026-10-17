from pydantic import BaseModel


# RMSE, Directional Accuracy (FP, TP, FN, TN), Sharpe Ratio, MDD, Total Return, Sortino Ratio and CVar from return type.
class PipelineEvaluation(BaseModel):
    # Error-based
    rmse: float
    mae: float
    mape: float

    # Accuracy-based
    directional_accuracy: float
    fp: int
    fn: int
    tp: int
    tn: int

    # Return-based
    total_return: float
    annualized_return: float

    # - Risk
    sharpe_ratio: float
    sortino_ratio: float
    CVar: float
    maximum_drawdown: float

    # - Volatility
    volatility: float
    downside_volatility: float

    # Trade statistic
    total_trades: float
    win_ratio: float
    avg_trade_return: float
    total_transaction_cost: float  # May be necessary
