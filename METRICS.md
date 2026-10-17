# Metrics Reference

The 16 core metrics (three families, taxonomy adapted from Zou et al.) plus 4 descriptive trading statistics computed for every run. Implementation: `src/evaluation/evaluation_module.py`.

## Error-based (3)

| Metric | Definition                                                                                                                                                                            |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RMSE   | Root mean squared error of predicted vs realized next-horizon returns                                                                                                                 |
| MAE    | Mean absolute error of predicted returns                                                                                                                                              |
| MAPE   | Mean absolute percentage error. **Caveat:** computed on returns, near-zero denominators inflate it - values of 600%+ are normal and expected; use RMSE/MAE for cross-model comparison |

## Classification-based (5)

| Metric            | Definition                                                                                                              |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------- |
| DA                | Directional accuracy: share of predictions with the correct sign, over all 9 stocks' predictions (not only traded ones) |
| TP / FP / FN / TN | Confusion-matrix counts for direction (UP = positive class)                                                             |

## Return–risk-based (8)

| Metric              | Definition                                                                                                                                                      |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Total Return        | Portfolio return over the full test window                                                                                                                      |
| Annualized Return   | Geometric annualization of total return                                                                                                                         |
| Sharpe              | **Annualized**: mean daily excess return / std of daily returns, × √252; risk-free rate 2% p.a.                                                                 |
| Sortino             | **Annualized**, same convention as Sharpe but denominator uses downside deviation only (penalizes negative volatility, not upside) - the paper's primary metric |
| CVaR                | Conditional value at risk (expected shortfall) of daily returns                                                                                                 |
| Maximum Drawdown    | Largest peak-to-trough decline of portfolio value                                                                                                               |
| Volatility          | Std of daily portfolio returns                                                                                                                                  |
| Downside Volatility | Std of negative daily portfolio returns                                                                                                                         |

## Descriptive trading statistics (4)

| Metric                 | Definition                                                                                                                                |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Total Trades           | Number of executed position changes                                                                                                       |
| Win Rate               | Share of closed positions (FIFO round trips) sold at a profit relative to gross entry price, regardless of holding length; gross of costs |
| Avg Trade Return       | Mean per-trade return                                                                                                                     |
| Total Transaction Cost | Cumulative cost paid at the configured rate (0.1% in the canonical experiment)                                                            |

## Conventions

- **Sortino interpretation bands used in the paper:** < 0.0 below risk-free performance · < 1.0 suboptimal · 1.0–2.0 good · 2.0–3.0 very good.
- Neural results report **mean ± sample standard deviation across seeds** [42, 69, 456, 789, 1337]; deterministic models run once.
- All portfolio metrics are computed on the shared walk-forward test window (2020-01-01 – 2023-12-31) under the run's configured transaction-cost regime.
