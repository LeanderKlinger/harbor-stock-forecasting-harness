// Model color palette — matches Python html_common.py

export const MODEL_COLORS: Record<string, string> = {
  xLSTM: "#8b5cf6", // violet-500
  GRU: "#3b82f6", // blue-500
  MLP: "#14b8a6", // teal-500
  Mamba: "#f59e0b", // amber-500
  Jamba: "#f97316", // orange-500
  ARIMA: "#a1a1aa", // zinc-400
  HoltsLES: "#71717a", // zinc-500
  BuyAndHold: "#d6d3d1", // stone-300
  EqualWeight: "#a8a29e", // stone-400
  Momentum: "#06b6d4", // cyan-500
};

// Tailwind border classes per model type
export const MODEL_BORDER_COLORS: Record<string, string> = {
  xLSTM: "border-l-violet-500",
  GRU: "border-l-blue-500",
  MLP: "border-l-teal-500",
  Mamba: "border-l-amber-500",
  Jamba: "border-l-orange-500",
  ARIMA: "border-l-zinc-400",
  HoltsLES: "border-l-zinc-500",
  BuyAndHold: "border-l-stone-300",
  EqualWeight: "border-l-stone-400",
  Momentum: "border-l-cyan-500",
};

export const MODEL_TOP_BORDER: Record<string, string> = {
  xLSTM: "border-t-violet-500",
  GRU: "border-t-blue-500",
  MLP: "border-t-teal-500",
  Mamba: "border-t-amber-500",
  Jamba: "border-t-orange-500",
  ARIMA: "border-t-zinc-400",
  HoltsLES: "border-t-zinc-500",
  BuyAndHold: "border-t-stone-300",
  EqualWeight: "border-t-stone-400",
  Momentum: "border-t-cyan-500",
};

export const MODEL_DOT_COLORS: Record<string, string> = {
  xLSTM: "bg-violet-500",
  GRU: "bg-blue-500",
  MLP: "bg-teal-500",
  Mamba: "bg-amber-500",
  Jamba: "bg-orange-500",
  ARIMA: "bg-zinc-400",
  HoltsLES: "bg-zinc-500",
  BuyAndHold: "bg-stone-300",
  EqualWeight: "bg-stone-400",
  Momentum: "bg-cyan-500",
};

export const BASELINE_MODELS = new Set(["BuyAndHold", "EqualWeight"]);

// Config fields that don't affect a model's output — strip before grouping
export const IRRELEVANT_CONFIG: Record<string, Set<string>> = {
  BuyAndHold: new Set([
    "sentiment_ema_spans",
    "headline_only",
    "min_predicted_return",
    "conviction_mode",
    "rebalance_horizon_days",
    "top_k",
    "early_stopping_patience",
    "sequence_length",
    "feature_set",
  ]),
  EqualWeight: new Set([
    "sentiment_ema_spans",
    "headline_only",
    "min_predicted_return",
    "conviction_mode",
    "top_k",
    "early_stopping_patience",
    "sequence_length",
    "feature_set",
  ]),
  Momentum: new Set([
    "sentiment_ema_spans",
    "headline_only",
    "early_stopping_patience",
    "sequence_length",
    "feature_set",
  ]),
  ARIMA: new Set([
    "sentiment_ema_spans",
    "headline_only",
    "early_stopping_patience",
    "sequence_length",
    "feature_set",
  ]),
  HoltsLES: new Set([
    "sentiment_ema_spans",
    "headline_only",
    "early_stopping_patience",
    "sequence_length",
    "feature_set",
  ]),
};

// Metrics aggregated across seeds
export const METRIC_KEYS = [
  "sortino_ratio",
  "sharpe_ratio",
  "annualized_return",
  "maximum_drawdown",
  "directional_accuracy",
  "win_ratio",
  "total_trades",
] as const;

// --- Formatting utilities ---

const RATIO_DISPLAY_MAX = 999.0;

export function fmtRatio(value: number): string {
  if (Math.abs(value) > RATIO_DISPLAY_MAX) {
    return value.toExponential(1);
  }
  return value.toFixed(2);
}

export function fmtPct(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export function metricColor(
  value: number,
  goodThreshold: number,
  higherIsBetter = true
): string {
  if (higherIsBetter) {
    return value >= goodThreshold ? "text-emerald-500" : "text-red-400";
  }
  return value <= goodThreshold ? "text-emerald-500" : "text-red-400";
}

// Check if a config field is irrelevant for a given model type
export function isIrrelevant(modelType: string, field: string): boolean {
  return IRRELEVANT_CONFIG[modelType]?.has(field) ?? false;
}
