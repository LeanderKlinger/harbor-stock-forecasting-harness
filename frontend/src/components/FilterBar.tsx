import { useMemo } from "react";

import type { FilterState } from "@/pages/Leaderboard";
import type { FeatureSet, RunIndexEntry } from "@/lib/types";
import { formatFeatureSet } from "@/lib/experimentAggregation";
import { DynamicFilter } from "./DynamicFilter";

interface Props {
  filters: FilterState;
  onChange: (filters: FilterState) => void;
  /** Used to derive dynamic option lists for horizon, threshold,
   *  retraining_cadence, sequence_length. */
  entries: RunIndexEntry[];
}

type ToggleValue = string | number | boolean;

interface PillGroupDef {
  key: keyof FilterState;
  options: { label: string; value: ToggleValue }[];
}

// Inherently binary filters keep hardcoded options.
const STATIC_PILL_ROWS: PillGroupDef[][] = [
  [
    {
      key: "headline",
      options: [
        { label: "headline", value: true },
        { label: "article", value: false },
      ],
    },
    {
      key: "stockCount",
      options: [
        { label: "single-stock", value: "single" },
        { label: "multi-stock", value: "multi" },
      ],
    },
    {
      key: "cost",
      options: [
        { label: "cost=0", value: "free" },
        { label: "cost>0", value: "paid" },
      ],
    },
    {
      key: "conviction",
      options: [
        { label: "fixed", value: "fixed" },
        { label: "vol-scaled", value: "volatility_scaled" },
      ],
    },
  ],
];

function PillButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center rounded-full px-1.5 py-px text-[0.625rem] font-medium leading-4 cursor-pointer hover:opacity-80 transition-colors ${
        active ? "bg-zinc-700 text-white" : "bg-zinc-100 text-zinc-600"
      }`}
    >
      {label}
    </button>
  );
}

// Distinct, sorted values for a numeric / mixed-type axis from the loaded
// entries. `undefined` is filtered out (old pre-lift entries lack the field).
function distinctSorted<T>(
  entries: RunIndexEntry[],
  pick: (e: RunIndexEntry) => T | undefined,
): T[] {
  const seen = new Set<string>();
  const out: T[] = [];
  for (const e of entries) {
    const v = pick(e);
    if (v === undefined) continue;
    const k = JSON.stringify(v);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(v);
  }
  // Numeric sort if every value is numeric, else string sort.
  if (out.every((v) => typeof v === "number")) {
    return (out as unknown as number[])
      .slice()
      .sort((a, b) => a - b) as unknown as T[];
  }
  return out.slice().sort((a, b) => String(a).localeCompare(String(b)));
}

export function FilterBar({ filters, onChange, entries }: Props) {
  function toggle(key: keyof FilterState, value: ToggleValue) {
    const current = filters[key];
    onChange({ ...filters, [key]: current === value ? null : value });
  }

  const horizonValues = useMemo(
    () =>
      distinctSorted(entries, (e) => e.config_summary.rebalance_horizon_days).map(
        (v) => ({ label: `H=${v}`, value: v }),
      ),
    [entries],
  );

  const thresholdValues = useMemo(
    () =>
      distinctSorted(entries, (e) => e.config_summary.min_predicted_return).map(
        (v) => ({
          label: v > 0 ? `thr=${(v * 100).toFixed(1)}%` : "thr=0",
          value: v,
        }),
      ),
    [entries],
  );

  // Cadence is mixed-type: null | "auto" | int. Encode null as the sentinel
  // string "frozen" so the filter UI can distinguish "no filter" (filter null)
  // from "match frozen runs" (filter "frozen").
  const cadenceValues = useMemo(() => {
    const raw = new Set<string>();
    for (const e of entries) {
      const v = e.config_summary.retraining_cadence_days;
      if (v === undefined) continue;
      if (v === null) raw.add("frozen");
      else if (v === "auto") raw.add("auto");
      else raw.add(String(v));
    }
    const sorted = [...raw].sort((a, b) => {
      // "frozen" first, then "auto", then numeric ascending.
      const order = (s: string): [number, number] => {
        if (s === "frozen") return [0, 0];
        if (s === "auto") return [1, 0];
        const n = Number(s);
        return [2, Number.isFinite(n) ? n : 0];
      };
      const oa = order(a);
      const ob = order(b);
      return oa[0] - ob[0] || oa[1] - ob[1];
    });
    return sorted.map((v) => ({
      label: v === "frozen" ? "frozen" : v === "auto" ? "auto" : `${v}d`,
      value: v,
    }));
  }, [entries]);

  const sequenceValues = useMemo(
    () =>
      distinctSorted(entries, (e) => e.config_summary.sequence_length).map(
        (v) => ({ label: `seq=${v}`, value: v }),
      ),
    [entries],
  );

  // Absent feature_set (pre-flag entries) normalizes to "full" so the pill
  // set reflects the real distinction, not index-schema history.
  const featureSetValues = useMemo(() => {
    const distinct = new Set<FeatureSet>();
    for (const e of entries) {
      distinct.add(e.config_summary.feature_set ?? "full");
    }
    const order: FeatureSet[] = ["full", "numeric", "close_only"];
    return order
      .filter((v) => distinct.has(v))
      .map((v) => ({ label: formatFeatureSet(v), value: v }));
  }, [entries]);

  return (
    <div className="flex flex-col gap-2 mb-6">
      {STATIC_PILL_ROWS.map((row, ri) => (
        <div key={ri} className="flex flex-wrap items-center gap-3">
          {row.map((group) => (
            <div key={group.key} className="flex gap-1">
              {group.options.map((opt) => (
                <PillButton
                  key={String(opt.value)}
                  label={opt.label}
                  active={filters[group.key] === opt.value}
                  onClick={() => toggle(group.key, opt.value)}
                />
              ))}
            </div>
          ))}
        </div>
      ))}

      <div className="flex flex-wrap items-center gap-3">
        {horizonValues.length >= 2 && (
          <DynamicFilter
            label="H"
            values={horizonValues}
            active={filters.horizon}
            onChange={(v) => onChange({ ...filters, horizon: v })}
          />
        )}
        {thresholdValues.length >= 2 && (
          <DynamicFilter
            label="thr"
            values={thresholdValues}
            active={filters.threshold}
            onChange={(v) => onChange({ ...filters, threshold: v })}
          />
        )}
        {cadenceValues.length >= 2 && (
          <DynamicFilter
            label="retrain"
            values={cadenceValues}
            active={filters.retraining_cadence}
            onChange={(v) => onChange({ ...filters, retraining_cadence: v })}
          />
        )}
        {sequenceValues.length >= 2 && (
          <DynamicFilter
            label="seq"
            values={sequenceValues}
            active={filters.sequence_length}
            onChange={(v) => onChange({ ...filters, sequence_length: v })}
          />
        )}
        {featureSetValues.length >= 2 && (
          <DynamicFilter
            label="features"
            values={featureSetValues}
            active={filters.feature_set}
            onChange={(v) => onChange({ ...filters, feature_set: v })}
          />
        )}

        <div className="flex items-center gap-2 text-xs text-zinc-500 ml-auto">
          <label>Since:</label>
          <input
            type="date"
            value={filters.since}
            onChange={(e) => onChange({ ...filters, since: e.target.value })}
            className="bg-zinc-100 rounded px-1.5 py-0.5 text-xs border border-zinc-300"
          />
          <label>To:</label>
          <input
            type="date"
            value={filters.to}
            onChange={(e) => onChange({ ...filters, to: e.target.value })}
            className="bg-zinc-100 rounded px-1.5 py-0.5 text-xs border border-zinc-300"
          />
        </div>
      </div>
    </div>
  );
}
