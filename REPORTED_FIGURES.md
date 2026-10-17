# Reported Figures

Every number printed in the paper, in the order the paper prints it, with the place it comes from. Reading this file needs no setup. Each row names the script section that recomputes the figure from the shipped run records; the commands are at the bottom.

**Terms used below.** The _headline environment_ is the setting of Table 3: 0.1% transaction cost, volatility-scaled conviction, 1% minimum predicted return, EMA spans [5,10,20]. A _cell_ is one model in one configuration; neural cells hold five seeds and are reported as mean ± sample standard deviation.

**Source key.** `canon §n`: section n of `scripts/paper/canon_results_stats.py`. `robustness §n`: section n of `scripts/paper/robustness_stats.py`. `audit`: the paper's Table 1, the audit of ten published works. `data`: the shipped data files.

## Abstract and Introduction

| Figure                                                      | Value                       | Source                                    |
| ----------------------------------------------------------- | --------------------------- | ----------------------------------------- |
| HoltsLES, best configuration (h=20), 0.1% cost              | Sortino 1.59, 42.14% annual | canon §5                                  |
| One identical MLP configuration (h=1), across seeds         | Sortino −0.70 to 1.22       | canon §1                                  |
| Audit: buy-and-hold / any market / any statistical baseline | 0/10, 3/10, 4/10            | audit                                     |
| Audit: seeding unreported / seed value reported             | 9/10, 1/10                  | audit                                     |
| Canonical grid                                              | 1440 configurations         | `experiments/10_rerun_big_eval_grid.yaml` |

## Related Work (Table 1)

| Figure                                 | Value | Source |
| -------------------------------------- | ----- | ------ |
| Audit: naive / persistence floor       | 0/10  | audit  |
| Audit: explicit multi-seed reporting   | 0/10  | audit  |
| Audit: a single metric family reported | 9/10  | audit  |

## Method

| Figure                                         | Value  | Source                                                                                |
| ---------------------------------------------- | ------ | ------------------------------------------------------------------------------------- |
| News articles across the nine stocks           | 65,647 | data: rows of the nine CSVs in `data/articles/`                                       |
| Cost reduction, headline-only vs full articles | 75×    | measured when designing the extraction (token cost); not recomputed from shipped data |

## Experiments and Results

| Figure        | Value                                                                        | Source          |
| ------------- | ---------------------------------------------------------------------------- | --------------- |
| Runs analysed | 1424 of 1440 (16 ARIMA h=1 excluded: deterministic LU-decomposition failure) | canon run index |

### 4.1 Leaderboard and Metric Floors

Table 2, each model's best configuration at 0.1% cost (canon §5):

| Model       | Configuration     | Sortino    | Annual return | Max drawdown |
| ----------- | ----------------- | ---------- | ------------- | ------------ |
| HoltsLES    | h=20, vol, 1%, LE | 1.59       | 42.1%         | 27.0%        |
| Momentum    | h=5, fix, 1%, LE  | 1.56       | 35.9%         | 33.7%        |
| ARIMA       | h=20, vol, 1%, LE | 1.38       | 28.8%         | 23.5%        |
| xLSTM       | h=5, vol, 1%, LE  | 1.30 ±0.35 | 31.1%         | 39.9%        |
| GRU         | h=20, vol, 1%, SE | 1.26 ±0.29 | 27.1%         | 34.6%        |
| BuyAndHold  | (none)            | 1.20       | 25.3%         | 36.4%        |
| EqualWeight | h=20, vol, 1%, LE | 1.17       | 24.6%         | 37.6%        |
| MLP         | h=5, vol, 0%, SE  | 1.01 ±0.27 | 19.8%         | 35.3%        |
| Jamba       | h=20, fix, 1%, SE | 0.98 ±0.26 | 20.7%         | 39.5%        |
| Mamba       | h=5, fix, 1%, LE  | 0.97 ±0.48 | 21.1%         | 37.9%        |

Configuration: horizon in days, conviction (vol = volatility-scaled, fix = fixed), minimum return threshold, EMA spans (SE = [2,5,10], LE = [5,10,20]).

| Figure                                             | Value                                                                                    | Source    |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------- |
| BuyAndHold passive floor                           | Sortino 1.20, 25.35% annual                                                              | canon §5  |
| Momentum outperforms all five neural architectures | in 23 of 24 shared configurations (the exception: h=20, fixed, 1%, SE; GRU 1.25 vs 1.20) | canon §7  |
| Top win-rate GRU instance (h=20, vol, 0%, SE)      | win rate 70.32% ±2.28, DA 48.62% ±2.55                                                   | canon §10 |
| EqualWeight floor at h=20                          | win rate 76.14%, DA 56.67%                                                               | canon §10 |
| Gap of that GRU instance below the floor           | 5.82pp win rate, 8.05pp DA (8.04pp before rounding)                                      | canon §10 |
| Neural directional accuracy, all configurations    | 44.4% to 52.7%                                                                           | canon §2  |

### 4.2 Horizon Stratification

Table 3, Sortino at the headline environment (canon §8):

| Model       | h=1        | h=5        | h=20       |
| ----------- | ---------- | ---------- | ---------- |
| BuyAndHold  | 1.20       | 1.20       | 1.20       |
| EqualWeight | 1.13       | 1.13       | 1.17       |
| Momentum    | 1.30       | 1.42       | 1.28       |
| HoltsLES    | 0.22       | 0.49       | 1.59       |
| ARIMA       | (excluded) | 0.03       | 1.38       |
| xLSTM       | 0.59 ±0.36 | 1.30 ±0.35 | 1.06 ±0.25 |
| GRU         | 0.83 ±0.32 | 0.91 ±0.36 | 1.13 ±0.24 |
| Mamba       | 0.35 ±0.30 | 0.89 ±0.47 | 0.91 ±0.24 |
| Jamba       | 0.33 ±0.26 | 0.80 ±0.18 | 0.85 ±0.17 |
| MLP         | 0.13 ±0.73 | 0.84 ±0.25 | 0.84 ±0.28 |

BuyAndHold is horizon-independent: one purchase, held throughout.

| Figure                           | Value                                  | Source    |
| -------------------------------- | -------------------------------------- | --------- |
| xLSTM at h=1, 0% vs 0.1% cost    | Sortino 1.36 ±0.33 falls to 0.59 ±0.36 | canon §11 |
| Momentum at h=1, 0% vs 0.1% cost | Sortino 1.79 falls to 1.30             | canon §11 |

### 4.3 Seed Instability

| Figure                                                | Value                       | Source       |
| ----------------------------------------------------- | --------------------------- | ------------ |
| One MLP configuration at h=1, across seeds            | −0.70 to 1.22 (spread 1.92) | canon §1, §9 |
| Jamba's seed spread per horizon, headline environment | 0.43 to 0.62                | canon §9     |
| Mean seed spread across all 120 neural cells          | 0.73                        | canon §1     |

### 4.4 Discussion

| Figure                                                              | Value                                                            | Source   |
| ------------------------------------------------------------------- | ---------------------------------------------------------------- | -------- |
| Best single-seed neural run: xLSTM, h=5, seed 1337                  | Sortino 1.90, +51.3% annual                                      | canon §3 |
| The same xLSTM cell over five seeds                                 | 1.30 ±0.35                                                       | canon §5 |
| The MLP configuration from 4.3 with the threshold off               | −0.12 to 0.75                                                    | canon §1 |
| RMSE variance explained by seed (Saidd's decomposition, these runs) | 2.28%, 7.67%, 11.13% at h=1, 5, 20; seed × model residual 20–29% | canon §6 |

### 4.5 Robustness

Sequence-length sweep, 150 runs: five neural models × two horizons (5, 20) × three lengths (60, 120, 250 days) × five seeds.

| Figure                                          | Value                                                         | Source        |
| ----------------------------------------------- | ------------------------------------------------------------- | ------------- |
| Longer context worse than 60 days (5-seed mean) | 11 of 20 comparisons                                          | robustness §2 |
| Best sweep cell                                 | xLSTM, h=20, 120 days: 1.29 ±0.24                             | robustness §2 |
| Mamba, h=5, 250 vs 60 days                      | better on 5 of 5 seeds, +0.32 mean                            | robustness §2 |
| Training time, 250 vs 60 days                   | 1.8× (MLP) to 14.5× (xLSTM); median fresh fits on one machine | robustness §2 |

Input ablation: 267 runs executed; the shipped index holds 239, because 28 baseline runs duplicated existing results exactly and were removed. Criteria fixed before execution: two-sided sign tests over 75 seed-matched comparisons per contrast, α = 0.05.

| Figure                                        | Value                 | Source        |
| --------------------------------------------- | --------------------- | ------------- |
| Removing the semantic features: RMSE improves | 47 of 75, p = 0.037   | robustness §1 |
| Change in DA, both contrasts                  | p = 0.91 and p = 0.73 | robustness §1 |
| Close-only vs numeric input: RMSE worsens     | 57 of 75, p = 7×10⁻⁶  | robustness §1 |
| MLP at h=20, full vs numeric input            | Sortino 0.84 to 1.79  | robustness §1 |
| Best close-only cell                          | GRU, h=20: 1.46 ±0.43 | robustness §1 |

## Conclusion

| Figure                                              | Value                                                     | Source       |
| --------------------------------------------------- | --------------------------------------------------------- | ------------ |
| NVIDIA gain over the test window                    | +725.7% (close 59.98 on 2020-01-02, 495.22 on 2023-12-28) | canon §12    |
| Lucky xLSTM seed above the best reported result     | 0.31 (1.90 vs 1.59)                                       | canon §3, §5 |
| The xLSTM cell's rank among the best configurations | fourth                                                    | canon §5     |

## Regenerating

```bash
uv sync
uv run python scripts/paper/canon_results_stats.py   # canon §1-12
uv run python scripts/paper/robustness_stats.py      # robustness §1-2
```

Both scripts read only the shipped run index (`data/system/execution_output/runs_index.json`) and stock data; neither trains a model or calls an API.
