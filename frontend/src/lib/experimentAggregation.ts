// Helpers for the comparison page — port of html_comparison.py's _find_varying_axes,
// _split_tables, _format_axis_value, _column_values, plus equity-curve aggregation.

import { IRRELEVANT_CONFIG } from "./constants";
import type {
  AggregatedGroup,
  ConfigSummary,
  EquityCurveSubset,
  FeatureSet,
} from "./types";

export type Axis = keyof ConfigSummary;

// All ConfigSummary values are flat (no nested objects); JSON.stringify is deterministic.
function configValueKey(value: ConfigSummary[Axis]): string {
  return JSON.stringify(value);
}

// --- Human-readable axis formatters (mirror _AXIS_FORMATTERS) ---

// Mapped type — each formatter receives the exact value type for its key.
type AxisFormatters = {
  [K in Axis]?: (v: ConfigSummary[K]) => string;
};

const AXIS_FORMATTERS: AxisFormatters = {
  rebalance_horizon_days: (v) => `${v}d`,
  min_predicted_return: (v) => (v > 0 ? `${(v * 100).toFixed(1)}%` : "none"),
  top_k: (v) => `top-${v}`,
  conviction_mode: (v) => v,
  early_stopping_patience: (v) => (v === null ? "off" : String(v)),
  transaction_cost_pct: (v) => `${(v * 100).toFixed(2)}%`,
  sentiment_ema_spans: (v) => `[${v.join(", ")}]`,
  retraining_cadence_days: (v) =>
    v === null ? "frozen" : v === "auto" ? "auto" : `${v}d`,
  sequence_length: (v) => (v === undefined ? "?" : `seq=${v}`),
  feature_set: (v) => formatFeatureSet(v),
};

const AXIS_DISPLAY_NAMES: Partial<Record<Axis, string>> = {
  rebalance_horizon_days: "Horizon",
  min_predicted_return: "Threshold",
  top_k: "Top K",
  conviction_mode: "Conviction",
  early_stopping_patience: "Early Stop",
  transaction_cost_pct: "Cost",
  sentiment_ema_spans: "EMA",
  retraining_cadence_days: "Retrain",
  sequence_length: "Seq Len",
  feature_set: "Features",
};

// Absent = pre-flag run = the canonical full vector.
export function formatFeatureSet(v: FeatureSet | undefined): string {
  if (v === undefined || v === "full") return "full-feat";
  return v === "numeric" ? "numeric" : "close-only";
}

export function formatAxisValue<K extends Axis>(axis: K, value: ConfigSummary[K]): string {
  const formatter = AXIS_FORMATTERS[axis];
  return formatter ? formatter(value) : String(value);
}

export function axisDisplayName(axis: Axis): string {
  return AXIS_DISPLAY_NAMES[axis] ?? axis;
}

// --- Varying-axis discovery ---

const ALL_AXES: readonly Axis[] = [
  "test_start",
  "test_end",
  "market_condition",
  "headline_only",
  "rebalance_horizon_days",
  "stock_ids",
  "sentiment_ema_spans",
  "transaction_cost_pct",
  "top_k",
  "min_predicted_return",
  "conviction_mode",
  "early_stopping_patience",
  "retraining_cadence_days",
  "sequence_length",
  "feature_set",
] as const;

/** Config keys whose values differ across groups, in canonical order. */
export function findVaryingAxes(groups: AggregatedGroup[]): Axis[] {
  if (groups.length === 0) return [];
  const varying: Axis[] = [];
  for (const axis of ALL_AXES) {
    const values = new Set<string>();
    for (const g of groups) {
      values.add(configValueKey(g.configSummary[axis]));
    }
    if (values.size > 1) varying.push(axis);
  }
  return varying;
}

// --- Page title (experiment-vs-comparison header) ---

export interface PageTitle {
  title: string;
  meta: string;
}

export function buildPageTitle(groups: AggregatedGroup[]): PageTitle {
  const total = groups.length;
  const totalRuns = groups.reduce((sum, g) => sum + g.nSeeds, 0);
  const experimentNames = new Set(
    groups.map((g) => g.experimentName).filter((n): n is string => n !== null),
  );
  if (experimentNames.size === 1 && groups.every((g) => g.experimentName)) {
    const name = [...experimentNames][0];
    return {
      title: `Experiment: ${name}`,
      meta: `${totalRuns} runs × ${total} groups`,
    };
  }
  if (experimentNames.size === 0) {
    return { title: "Comparison", meta: `${totalRuns} runs × ${total} groups` };
  }
  return {
    title: "Comparison",
    meta: `${totalRuns} runs × ${total} groups across ${experimentNames.size} experiments`,
  };
}

// --- Table splitting (port of _split_tables) ---

export interface ResultsTable {
  /** Empty for single-table output. Otherwise e.g. "Horizon = 20d". */
  title: string;
  /** Inner-axis names that the columns split on. */
  innerAxes: Axis[];
  /** For each column, one formatted value per inner axis (rendered as separate lines). */
  columnValues: string[][];
  /** Inner-axis-value keys aligned with columnValues (lookup index). */
  innerKeys: string[];
  /** Groups assigned to this table. */
  groups: AggregatedGroup[];
}

/** Match Python's "try float first, fall back to str" sort semantics. */
function tryNumericSort<T>(arr: T[], pick: (item: T) => unknown): T[] {
  const numericForms = arr.map((item) => {
    const v = pick(item);
    const n = typeof v === "number" ? v : Number(v);
    return Number.isFinite(n) ? n : null;
  });
  if (numericForms.every((n) => n !== null)) {
    return arr
      .map((item, i) => ({ item, key: numericForms[i] as number }))
      .sort((a, b) => a.key - b.key)
      .map((e) => e.item);
  }
  return [...arr].sort((a, b) => String(pick(a)).localeCompare(String(pick(b))));
}

function innerColumnKey(group: AggregatedGroup, innerAxes: Axis[]): string {
  return innerAxes.map((a) => configValueKey(group.configSummary[a])).join("|");
}

function innerColumnValues(group: AggregatedGroup, innerAxes: Axis[]): string[] {
  return innerAxes.map((a) => formatAxisValue(a, group.configSummary[a]));
}

/**
 * Split groups into one or more tables.
 *
 * 0-1 varying axes → single table.
 * 2+ varying axes → split by axis with most unique values, remaining axes become columns.
 */
export function splitTables(
  groups: AggregatedGroup[],
  varyingAxes: Axis[],
): ResultsTable[] {
  if (varyingAxes.length <= 1) {
    return [buildSingleTable(groups, varyingAxes)];
  }

  const axisCounts = varyingAxes.map((axis) => {
    const seen = new Set<string>();
    for (const g of groups) {
      seen.add(configValueKey(g.configSummary[axis]));
    }
    return { axis, count: seen.size };
  });
  axisCounts.sort((a, b) => b.count - a.count);
  const outerAxis = axisCounts[0].axis;
  const innerAxes = varyingAxes.filter((a) => a !== outerAxis);

  const seenOuter = new Set<string>();
  const outerEntries: { value: ConfigSummary[Axis]; key: string }[] = [];
  for (const g of groups) {
    const v = g.configSummary[outerAxis];
    const k = configValueKey(v);
    if (!seenOuter.has(k)) {
      seenOuter.add(k);
      outerEntries.push({ value: v, key: k });
    }
  }
  const outerSorted = tryNumericSort(outerEntries, (e) => e.value);

  return outerSorted
    .map((outer): ResultsTable | null => {
      const tableGroups = groups.filter(
        (g) => configValueKey(g.configSummary[outerAxis]) === outer.key,
      );
      if (tableGroups.length === 0) return null;
      const title = `${axisDisplayName(outerAxis)} = ${formatAxisValue(outerAxis, outer.value)}`;
      const { columnValues, innerKeys } = collectInnerColumns(tableGroups, innerAxes);
      return { title, innerAxes, columnValues, innerKeys, groups: tableGroups };
    })
    .filter((t): t is ResultsTable => t !== null);
}

function buildSingleTable(groups: AggregatedGroup[], varyingAxes: Axis[]): ResultsTable {
  if (varyingAxes.length === 0) {
    return {
      title: "",
      innerAxes: [],
      columnValues: [["All"]],
      innerKeys: [""],
      groups,
    };
  }
  const { columnValues, innerKeys } = collectInnerColumns(groups, varyingAxes);
  return { title: "", innerAxes: varyingAxes, columnValues, innerKeys, groups };
}

function collectInnerColumns(
  groups: AggregatedGroup[],
  innerAxes: Axis[],
): { columnValues: string[][]; innerKeys: string[] } {
  const seen = new Set<string>();
  const entries: {
    sortValue: ConfigSummary[Axis] | 0;
    key: string;
    values: string[];
  }[] = [];
  for (const g of groups) {
    const key = innerColumnKey(g, innerAxes);
    if (seen.has(key)) continue;
    seen.add(key);
    const sortValue = innerAxes.length ? g.configSummary[innerAxes[0]] : 0;
    entries.push({ sortValue, key, values: innerColumnValues(g, innerAxes) });
  }
  const sorted = tryNumericSort(entries, (e) => e.sortValue);
  return {
    columnValues: sorted.map((e) => e.values),
    innerKeys: sorted.map((e) => e.key),
  };
}

/** Group lookup keyed on (model_type, relevant-stripped-inner-key).
 *
 * Axes listed in IRRELEVANT_CONFIG[modelType] are replaced with "*" before
 * keying — so e.g. Momentum (EMA-irrelevant) populates every column that
 * shares its (cost, threshold, conviction) regardless of EMA, and BuyAndHold
 * (everything-irrelevant) populates every column for its model row.
 */
export function buildGroupLookup(
  groups: AggregatedGroup[],
  innerAxes: Axis[],
): Map<string, AggregatedGroup> {
  const map = new Map<string, AggregatedGroup>();
  for (const g of groups) {
    const irrelevant = IRRELEVANT_CONFIG[g.modelType];
    const parts = innerAxes.map((a) => {
      if (irrelevant?.has(a)) return "*";
      return configValueKey(g.configSummary[a]);
    });
    map.set(`${g.modelType}|${parts.join("|")}`, g);
  }
  return map;
}

/** Build the lookup key for a (model_type, columnInnerKey) cell, with the
 * model's irrelevant axes stripped to "*" — must match buildGroupLookup. */
export function lookupKey(
  modelType: string,
  innerAxes: Axis[],
  columnInnerKey: string,
): string {
  const irrelevant = IRRELEVANT_CONFIG[modelType];
  if (!irrelevant || irrelevant.size === 0) {
    return `${modelType}|${columnInnerKey}`;
  }
  const parts = columnInnerKey.split("|");
  const stripped = innerAxes.map((a, i) => (irrelevant.has(a) ? "*" : parts[i]));
  return `${modelType}|${stripped.join("|")}`;
}

/** Models in the table, ordered by mean Sortino across all columns descending. */
export function orderedModelsForTable(groups: AggregatedGroup[]): string[] {
  const sortinos = new Map<string, number[]>();
  for (const g of groups) {
    const arr = sortinos.get(g.modelType) ?? [];
    arr.push(g.meanMetrics.sortino_ratio ?? 0);
    sortinos.set(g.modelType, arr);
  }
  const meanOf = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
  return [...sortinos.keys()].sort(
    (a, b) => meanOf(sortinos.get(b)!) - meanOf(sortinos.get(a)!),
  );
}

// --- BuyAndHold expansion (port of normalize_baseline_configs) ---

/**
 * BuyAndHold's runner override gives it a fake giant horizon, so by default it
 * appears once across the whole experiment. To match Python's comparison view
 * (where BuyAndHold appears in every horizon table as a universal baseline),
 * duplicate each unique BuyAndHold group across the unique horizons present
 * among non-baseline groups in the selection.
 *
 * EqualWeight is not expanded — its horizon already tracks the grid value.
 */
export function expandBuyAndHoldAcrossHorizons(
  groups: AggregatedGroup[],
): AggregatedGroup[] {
  const horizonsFromOthers = new Set<number>();
  for (const g of groups) {
    if (g.modelType !== "BuyAndHold") {
      horizonsFromOthers.add(g.configSummary.rebalance_horizon_days);
    }
  }
  if (horizonsFromOthers.size <= 1) return groups;
  const sortedHorizons = [...horizonsFromOthers].sort((a, b) => a - b);

  const result: AggregatedGroup[] = [];
  for (const g of groups) {
    if (g.modelType !== "BuyAndHold") {
      result.push(g);
      continue;
    }
    for (const horizon of sortedHorizons) {
      result.push({
        ...g,
        groupKey: `${g.groupKey}|h=${horizon}`,
        configSummary: { ...g.configSummary, rebalance_horizon_days: horizon },
      });
    }
  }
  return result;
}

// --- Equity curve aggregation (port of _aggregate_groups) ---

export interface AggregatedEquity {
  groupKey: string;
  modelType: string;
  label: string;
  timestamps: string[];
  /** Mean across seeds, normalized so each input curve started at 1.0. */
  mean: number[];
  /** Std across seeds at each timestamp (0 when nSeeds < 2). */
  std: number[];
}

/**
 * For each group, normalize each child's equity curve to start at 1.0,
 * then take per-timestamp mean and std across seeds. Truncates to the shortest
 * curve length (timestamps in run_detail.json should already align across seeds).
 */
export function buildAggregatedEquity(
  groups: AggregatedGroup[],
  runDetails: Map<string, EquityCurveSubset>,
  labelForGroup: (g: AggregatedGroup) => string,
): AggregatedEquity[] {
  const result: AggregatedEquity[] = [];
  for (const group of groups) {
    const curves: { timestamps: string[]; values: number[] }[] = [];
    for (const child of group.children) {
      const detail = runDetails.get(child.run_dir);
      if (!detail || detail.equity_curve.length === 0) continue;
      curves.push({
        timestamps: detail.equity_curve.map((p) => p.timestamp),
        values: detail.equity_curve.map((p) => p.value),
      });
    }
    if (curves.length === 0) continue;

    const normalized: number[][] = [];
    for (const c of curves) {
      if (c.values[0] !== 0) {
        normalized.push(c.values.map((v) => v / c.values[0]));
      }
    }
    if (normalized.length === 0) continue;

    // Use the SHORTEST curve's timestamps (not curves[0]) — if seeds diverge
    // in length, curves[0]'s timestamps may not align with the truncated data.
    let minIdx = 0;
    for (let i = 1; i < normalized.length; i++) {
      if (normalized[i].length < normalized[minIdx].length) minIdx = i;
    }
    const minLen = normalized[minIdx].length;
    const timestamps = curves[minIdx].timestamps.slice(0, minLen);

    const mean: number[] = [];
    const std: number[] = [];
    for (let i = 0; i < minLen; i++) {
      const points = normalized.map((n) => n[i]);
      const m = points.reduce((a, b) => a + b, 0) / points.length;
      mean.push(m);
      if (points.length > 1) {
        const variance =
          points.reduce((acc, v) => acc + (v - m) ** 2, 0) / (points.length - 1);
        std.push(Math.sqrt(variance));
      } else {
        std.push(0);
      }
    }

    result.push({
      groupKey: group.groupKey,
      modelType: group.modelType,
      label: labelForGroup(group),
      timestamps,
      mean,
      std,
    });
  }
  return result;
}

// --- Shared-config detection (for SharedConfigStrip) ---

const STRIP_FIELDS: readonly Axis[] = [
  "test_start",
  "test_end",
  "stock_ids",
  "headline_only",
  "transaction_cost_pct",
  "early_stopping_patience",
  "top_k",
  "min_predicted_return",
  "conviction_mode",
  "retraining_cadence_days",
  "feature_set",
] as const;

export interface SharedConfigEntry {
  field: Axis;
  value: ConfigSummary[Axis];
}

/** Fields that are equal across all groups, in canonical display order. */
export function findSharedConfig(groups: AggregatedGroup[]): SharedConfigEntry[] {
  if (groups.length === 0) return [];
  const out: SharedConfigEntry[] = [];
  for (const field of STRIP_FIELDS) {
    const refKey = configValueKey(groups[0].configSummary[field]);
    const allSame = groups.every(
      (g) => configValueKey(g.configSummary[field]) === refKey,
    );
    if (allSame) out.push({ field, value: groups[0].configSummary[field] });
  }
  return out;
}
