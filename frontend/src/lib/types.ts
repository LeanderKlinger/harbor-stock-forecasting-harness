// Types matching the JSON schemas produced by the Python backend

// --- Shared metrics (same shape in runs_index.json and run_detail.json) ---

export interface Metrics {
  rmse: number;
  mae: number;
  mape: number;
  directional_accuracy: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  total_return: number;
  annualized_return: number;
  maximum_drawdown: number;
  cvar: number;
  volatility: number;
  downside_volatility: number;
  win_ratio: number;
  total_trades: number;
  avg_trade_return: number;
  total_transaction_cost: number;
}

// --- runs_index.json entries ---

export interface RunIndexEntry {
  run_id: string;
  run_dir: string;
  model_type: string;
  model_name: string;
  execution_name: string;
  description: string;
  experiment_name: string | null;
  seed: number | null;
  timestamp: string;
  training_duration_s: number;
  backtest_duration_s: number;
  config_summary: ConfigSummary;
  metrics: Metrics;
}

export interface ConfigSummary {
  test_start: string;
  test_end: string;
  market_condition: string;
  headline_only: boolean;
  rebalance_horizon_days: number;
  stock_ids: string[];
  sentiment_ema_spans: number[];
  transaction_cost_pct: number;
  top_k: number;
  min_predicted_return: number;
  conviction_mode: string;
  early_stopping_patience: number | null;
  retraining_cadence_days: number | "auto" | null;
  // Optional because pre-lift index entries don't have it. `reindex --force`
  // backfills from each pipeline_output.json (which carries the real value).
  sequence_length?: number;
  // Optional for the same reason; backfills to "full" via the Config default.
  feature_set?: FeatureSet;
}

export type FeatureSet = "full" | "numeric" | "close_only";

// --- run_detail.json ---

export interface RunDetail {
  run_id: string;
  model_type: string;
  model_name: string;
  run_timestamp: string;
  training_duration_s: number;
  backtest_duration_s: number;
  config: RunDetailConfig;
  metrics: Metrics;
  equity_curve: EquityPoint[];
  stock_charts: Record<string, StockChartData>;
  threshold_data: ThresholdPoint[] | null;
}

export interface RunDetailConfig {
  test_start: string;
  test_end: string;
  training_start: string;
  training_end: string;
  market_condition: string;
  headline_only: boolean;
  rebalance_horizon_days: number;
  sentiment_ema_spans: number[];
  transaction_cost_pct: number;
  risk_free_rate: number;
  stock_ids: string[];
  top_k: number;
  min_predicted_return: number;
  conviction_mode: string;
  early_stopping_patience: number | null;
  sequence_length: number;
  retraining_cadence_days: number | "auto" | null;
  feature_set?: FeatureSet;
}

export interface EquityPoint {
  timestamp: string;
  value: number;
}

// Subset returned by POST /api/run_details when fields=["equity_curve"].
export type EquityCurveSubset = Pick<RunDetail, "equity_curve">;
export type RunDetailBatchValue = RunDetail | EquityCurveSubset | null;

export interface StockChartData {
  trades: TradeMarker[];
  held_intervals: HeldInterval[];
  not_held_intervals: Interval[];
}

export interface TradeMarker {
  timestamp: string;
  action: "BUY" | "SELL";
  quantity: number;
  price: number;
}

export interface HeldInterval {
  start: string;
  end: string;
  profitable: boolean;
  closed: boolean;
}

export interface Interval {
  start: string;
  end: string;
}

export interface ThresholdPoint {
  timestamp: string;
  value: number;
}

// --- .stock_prices/{STOCK}.json ---

export interface StockPrice {
  timestamp: string;
  close: number;
}

// --- Aggregated leaderboard types ---

export interface AggregatedGroup {
  groupKey: string;
  modelType: string;
  modelName: string;
  configSummary: ConfigSummary;
  experimentName: string | null;
  timestamp: string;
  nSeeds: number;
  meanMetrics: Record<string, number>;
  stdMetrics: Record<string, number>;
  meanTrainingS: number;
  meanBacktestS: number;
  children: RunIndexEntry[];
}
