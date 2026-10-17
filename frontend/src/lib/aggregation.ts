import type { RunIndexEntry, ConfigSummary, AggregatedGroup } from "./types";
import { BASELINE_MODELS, IRRELEVANT_CONFIG, METRIC_KEYS } from "./constants";

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function stdev(values: number[]): number {
  if (values.length < 2) return 0;
  const avg = mean(values);
  const sqDiffs = values.map((v) => (v - avg) ** 2);
  return Math.sqrt(sqDiffs.reduce((a, b) => a + b, 0) / (values.length - 1));
}

function groupKey(entry: RunIndexEntry): string {
  const experiment = entry.experiment_name ?? "_manual";
  const modelType = entry.model_type;
  const cfg = { ...entry.config_summary } as Record<string, unknown>;

  // Strip irrelevant config fields for this model type
  const irrelevant = IRRELEVANT_CONFIG[modelType];
  if (irrelevant) {
    for (const key of irrelevant) {
      delete cfg[key];
    }
  }

  // conviction_mode is irrelevant when threshold is 0
  if ((cfg.min_predicted_return as number) === 0) {
    delete cfg.conviction_mode;
  }

  return `${experiment}|${modelType}|${JSON.stringify(cfg, Object.keys(cfg).sort())}`;
}

function normalizeBaselines(entries: RunIndexEntry[]): void {
  const nonBaseline = entries.filter(
    (e) => !BASELINE_MODELS.has(e.model_type)
  );
  if (nonBaseline.length === 0) return;

  // Find mode top_k and horizon from non-baseline runs
  const topKs = nonBaseline.map((e) => e.config_summary.top_k);
  const modeTopK = mode(topKs);

  const horizons = nonBaseline.map(
    (e) => e.config_summary.rebalance_horizon_days
  );
  const modeHorizon = mode(horizons);

  for (const entry of entries) {
    if (!BASELINE_MODELS.has(entry.model_type)) continue;
    const irrelevant = IRRELEVANT_CONFIG[entry.model_type];
    if (!irrelevant) continue;
    if (irrelevant.has("top_k")) {
      entry.config_summary.top_k = modeTopK;
    }
    if (irrelevant.has("rebalance_horizon_days")) {
      entry.config_summary.rebalance_horizon_days = modeHorizon;
    }
  }
}

function mode<T>(values: T[]): T {
  const counts = new Map<T, number>();
  for (const v of values) {
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  let best = values[0];
  let bestCount = 0;
  for (const [v, c] of counts) {
    if (c > bestCount) {
      best = v;
      bestCount = c;
    }
  }
  return best;
}

export function aggregateRuns(
  entries: RunIndexEntry[]
): AggregatedGroup[] {
  // Work on a shallow copy to avoid mutating the original
  const copy = entries.map((e) => ({
    ...e,
    config_summary: { ...e.config_summary },
  }));

  normalizeBaselines(copy);

  // Group by key
  const groups = new Map<string, RunIndexEntry[]>();
  for (const entry of copy) {
    const key = groupKey(entry);
    const group = groups.get(key);
    if (group) {
      group.push(entry);
    } else {
      groups.set(key, [entry]);
    }
  }

  const aggregates: AggregatedGroup[] = [];

  for (const [key, children] of groups) {
    // Sort children by Sortino descending (best seed first)
    children.sort(
      (a, b) =>
        (b.metrics.sortino_ratio ?? 0) - (a.metrics.sortino_ratio ?? 0)
    );

    const first = children[0];
    const n = children.length;

    const meanMetrics: Record<string, number> = {};
    const stdMetrics: Record<string, number> = {};
    for (const mk of METRIC_KEYS) {
      const values = children.map((c) => c.metrics[mk] ?? 0);
      meanMetrics[mk] = mean(values);
      stdMetrics[mk] = stdev(values);
    }

    const trainTimes = children.map((c) => c.training_duration_s);
    const btTimes = children.map((c) => c.backtest_duration_s);

    aggregates.push({
      groupKey: key,
      modelType: first.model_type,
      modelName: first.model_name,
      configSummary: first.config_summary as ConfigSummary,
      experimentName: first.experiment_name,
      timestamp: children
        .map((c) => c.timestamp.slice(0, 10))
        .sort()
        .at(-1)!,
      nSeeds: n,
      meanMetrics,
      stdMetrics,
      meanTrainingS: mean(trainTimes),
      meanBacktestS: mean(btTimes),
      children,
    });
  }

  // Sort by mean Sortino descending
  aggregates.sort(
    (a, b) =>
      (b.meanMetrics.sortino_ratio ?? 0) - (a.meanMetrics.sortino_ratio ?? 0)
  );

  return aggregates;
}

// Find the best group (highest mean Sortino) per model type among visible groups
export function bestPerModelType(
  groups: AggregatedGroup[]
): Set<string> {
  const best = new Map<string, { key: string; sortino: number }>();
  for (const g of groups) {
    const s = g.meanMetrics.sortino_ratio ?? 0;
    const current = best.get(g.modelType);
    if (!current || s > current.sortino) {
      best.set(g.modelType, { key: g.groupKey, sortino: s });
    }
  }
  return new Set([...best.values()].map((v) => v.key));
}
