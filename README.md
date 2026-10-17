# HARBOR: Harness for AnchoR-Based Objective Reporting

An anchor-first evaluation harness for stock-price forecasting research, accompanying the ACM ICAIF 2026 paper _"Simpler Is Sometimes Better: A Statistical Baseline Beats Neural Models and the Case for Shared-Condition Evaluation"_ ([doi:10.1145/3858063.3858830](https://doi.org/10.1145/3858063.3858830)).

## What is this?

The paper argues that evaluation practice in ML stock forecasting (missing baselines, unreported seeds, incompatible metrics) can manufacture or dissolve state-of-the-art claims. HARBOR is the instrument behind that demonstration: a pipeline that backtests **10 forecasting models under identical conditions** (same data, features, costs, training loop and evaluation) and reports ~20 metrics across error, classification and return–risk families, with multi-seed averaging.

The models: market baselines (BuyAndHold, EqualWeight, Momentum), statistical (ARIMA, Holt's LES) and neural (MLP, GRU, xLSTM, Mamba, Jamba). The paper's headline findings, the statistical baseline beating every neural architecture and identical neural configurations swinging from a Sortino of −0.70 to +1.22 on random seed alone, come out of this repository.

## I want to...

| Goal                               | Do this                                                                                 |
| ---------------------------------- | --------------------------------------------------------------------------------------- |
| Verify the paper's numbers         | Read [`REPORTED_FIGURES.md`](REPORTED_FIGURES.md), **no setup needed**                  |
| Understand the metrics             | Read [`METRICS.md`](METRICS.md)                                                         |
| Understand the code                | Read [`ARCHITECTURE.md`](ARCHITECTURE.md): module map and invariants                    |
| See the pipeline run               | `./first-setup.sh`, then the smoke test below (minutes)                                 |
| Reproduce the canonical experiment | The 1440-configuration grid below (**≈30h** on a 28-core Apple M3 Ultra, CPU)           |
| Browse results interactively       | `./dev.sh`: dashboard with leaderboards, per-run detail and filtering                   |
| Regenerate the paper's statistics  | `scripts/paper/canon_results_stats.py` and `robustness_stats.py` (seconds, no training) |

## Setup

Prerequisites: [`uv`](https://docs.astral.sh/uv/) (Python package manager; installs the right Python) and optionally [Bun](https://bun.sh) for the React dashboard. An `ANTHROPIC_API_KEY` is only needed to extract sentiment beyond the shipped corpus: cached LLM scores for the full 9-stock universe are included (`data/system/parsing_checkpoints/`), so the shipped experiments re-run without API spend.

```bash
./first-setup.sh            # installs deps, creates .env, sanity-checks the pipeline
# then add your ANTHROPIC_API_KEY to .env (only needed for new sentiment extraction)
```

## Running experiments

Experiments are declarative YAML files (`experiments/`); `00_template.yaml` documents every available knob. The runner expands the grid axes into individual runs via cartesian product and executes them in parallel.

```bash
# Smoke test: verifies the whole pipeline in minutes
uv run python -m src.experiments run experiments/06_smoke_test.yaml

# The paper's canonical experiment: 1440 unique configurations
uv run python -m src.experiments run experiments/10_rerun_big_eval_grid.yaml --workers auto
```

Outputs land in `data/system/execution_output/` (one folder per run: canonical JSON, metrics, frontend detail) and are indexed in `runs_index.json`. A disk-backed fit cache (`data/training_cache/`) makes reruns and overlapping grids dramatically cheaper. Start `./dev.sh` to inspect results in the dashboard.

Remote execution is turnkey: `scripts/setup_remote.sh` bootstraps an SSH host once, `scripts/deploy.sh user@host <yaml>` stages and launches in a detached tmux session, `scripts/fetch_results.sh user@host` pulls results home and reindexes.

## Data

`data/stock_data/` ships daily OHLCV CSVs for the 9-stock universe and `data/articles/` the per-stock news corpus, both retrieved from the public [FNSPID dataset](https://arxiv.org/abs/2402.06698) without alterations. Stock selection criteria and the train/test windows are documented in the paper (§3.2).

**Batteries included:** the paper's results ship with the repository, so the dashboard works immediately (`./first-setup.sh && ./dev.sh`), no 30-hour rerun required.

- `data/system/execution_output/`: all 1813 runs behind the paper (canonical grid 1424, input ablation 239, sequence-length sweep 150) with per-run metrics and frontend detail, plus the run index. The raw per-run `pipeline_output.json` files (3.3 GB of full prediction/trade records) are omitted for size and available on request.
- `data/system/parsing_checkpoints/`: cached LLM sentiment scores for the full corpus; re-running experiments re-uses these instead of calling the API.
- The training fit-cache is not included; fresh runs retrain (the honest cost of full reproduction, see runtime expectations above).
- Do **not** run `python -m src.experiments reindex --force`: it rebuilds the run index from the omitted raw outputs and would empty the shipped one. New runs append to the index normally.

## Layout

```
src/extraction/    polars ingestion, LLM sentiment scoring, temporal enrichment
src/training/      walk-forward backtesting, per-stock parallel training, fit cache
src/evaluation/    metrics (see METRICS.md) + persistence
src/experiments/   YAML grid runner
src/models/        pydantic data contracts at module boundaries
experiments/       experiment definitions (00_template.yaml = reference)
scripts/paper/     reproducible home of every statistic reported in the paper
frontend/          run-inspection dashboard (Bun + Vite)
tests/             pytest suite (runs on real data)
```

Quality gate: `./scripts/check.sh` (ruff + pyright strict + pytest + frontend tsc/eslint).

## Citing

```bibtex
@inproceedings{klinger2026simplerissometimesbetter,
  author    = {Klinger, Leander Immanuel},
  title     = {Simpler Is Sometimes Better: A Statistical Baseline Beats Neural Models and the Case for Shared-Condition Evaluation},
  booktitle = {7th ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026},
  address   = {Milan, Italy},
  publisher = {ACM},
  doi       = {10.1145/3858063.3858830}
}
```

## AI-assistance disclosure

Development of this codebase and research support were substantially AI-assisted (Anthropic Claude). All design decisions, findings and the paper's claims are the author's sole responsibility.

## License

- **Code:** [MIT](LICENSE).
- **Data** (`data/stock_data/`, `data/articles/`, and everything derived from them under `data/system/`): from the FNSPID dataset (Dong, Fan and Peng, 2024), released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Non-commercial use only, with attribution to FNSPID.
