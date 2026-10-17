import { useState, useMemo, useCallback, Fragment } from "react";
import { Link } from "react-router";
import type { AggregatedGroup, RunIndexEntry } from "@/lib/types";
import {
  MODEL_BORDER_COLORS,
  MODEL_TOP_BORDER,
  fmtRatio,
  fmtPct,
  metricColor,
} from "@/lib/constants";
import { compareUrl } from "@/lib/api";
import { ConfigBadges } from "./ConfigBadges";

interface Props {
  groups: AggregatedGroup[];
  bestKeys: Set<string>;
  /** When true, render selection checkboxes on parent + expanded seed rows. */
  selectMode: boolean;
  /** run_id set — both Compare and Delete operate on these. */
  selectedRunIds: Set<string>;
  /** Toggle add/remove for a list of run_ids. Used for both parent (all children)
   *  and seed (single id) clicks. Semantics: if every passed id is already in the
   *  set, remove them all; otherwise add the missing ones. */
  onToggleRuns: (runIds: string[]) => void;
  /** Replace the full selection set (used by select-all). */
  onSetSelected: (next: Set<string>) => void;
  /** Map of experimentName → all group keys in that experiment (across filters). */
  experimentGroupKeys: Map<string, string[]>;
}

type SortKey = string;
type SortDir = "asc" | "desc";

const COLUMNS: { label: string; key: SortKey; numeric: boolean }[] = [
  { label: "#", key: "rank", numeric: false },
  { label: "Run", key: "run", numeric: false },
  { label: "Date", key: "timestamp", numeric: false },
  { label: "Stocks", key: "stocks", numeric: false },
  { label: "Sortino", key: "sortino_ratio", numeric: true },
  { label: "Sharpe", key: "sharpe_ratio", numeric: true },
  { label: "Ann. Return", key: "annualized_return", numeric: true },
  { label: "Max DD", key: "maximum_drawdown", numeric: true },
  { label: "Dir. Acc.", key: "directional_accuracy", numeric: true },
  { label: "Win %", key: "win_ratio", numeric: true },
  { label: "Trades", key: "total_trades", numeric: true },
  { label: "Time", key: "time", numeric: true },
];

function getSortValue(group: AggregatedGroup, key: SortKey): number | string {
  if (key === "timestamp") return group.timestamp;
  if (key === "stocks") return group.configSummary.stock_ids.length;
  if (key === "time") return group.meanTrainingS + group.meanBacktestS;
  return group.meanMetrics[key] ?? 0;
}

function formatTime(seconds: number): string {
  return seconds >= 60 ? `${(seconds / 60).toFixed(1)}m` : `${seconds.toFixed(0)}s`;
}

function MetricWithStd({
  mean,
  std,
  nSeeds,
  formatter,
}: {
  mean: number;
  std: number;
  nSeeds: number;
  formatter: (v: number) => string;
}) {
  if (nSeeds <= 1 || std === 0) {
    return <>{formatter(mean)}</>;
  }
  return (
    <>
      {formatter(mean)}{" "}
      <span className="text-zinc-400 text-xs">± {formatter(std)}</span>
    </>
  );
}

const RANK_BADGES = [
  { badge: "bg-amber-400 text-amber-950", row: "bg-amber-50/50" },
  { badge: "bg-zinc-300 text-zinc-700", row: "bg-zinc-100/50" },
  { badge: "bg-orange-300 text-orange-800", row: "bg-orange-50/50" },
];

function RankCell({ rank }: { rank: number }) {
  if (rank <= 3) {
    const style = RANK_BADGES[rank - 1];
    return (
      <span
        className={`inline-flex items-center whitespace-nowrap rounded-full px-2 py-0.5 text-[0.65rem] font-bold ${style.badge}`}
      >
        ★ {rank}
      </span>
    );
  }
  return <>{rank}</>;
}

function SeedRow({
  run,
  selectMode,
  selected,
  onToggle,
}: {
  run: RunIndexEntry;
  selectMode: boolean;
  selected: boolean;
  onToggle: () => void;
}) {
  const m = run.metrics;
  const totalS = run.training_duration_s + run.backtest_duration_s;
  const borderCls = MODEL_BORDER_COLORS[run.model_type] ?? "";

  return (
    <tr className={`border-l-4 ${borderCls} border-b border-zinc-100 bg-zinc-50/60`}>
      <td className="px-2 py-1.5">
        {selectMode && (
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggle}
            className="cursor-pointer accent-zinc-700 ml-6"
            title="Select for comparison / delete"
          />
        )}
      </td>
      <td className="px-2 py-1.5 pl-6">
        <Link
          to={`/runs/${run.run_dir}`}
          className="text-blue-500 hover:underline text-xs"
        >
          seed={run.seed ?? "?"}
        </Link>
      </td>
      <td className="px-2 py-1.5 text-xs text-zinc-400 whitespace-nowrap">
        {run.timestamp.slice(0, 10)}
      </td>
      <td className="px-2 py-1.5" />
      <td className={`px-2 py-1.5 text-xs ${metricColor(m.sortino_ratio, 1.0)}`}>
        {fmtRatio(m.sortino_ratio)}
      </td>
      <td className={`px-2 py-1.5 text-xs ${metricColor(m.sharpe_ratio, 1.0)}`}>
        {fmtRatio(m.sharpe_ratio)}
      </td>
      <td className={`px-2 py-1.5 text-xs ${metricColor(m.annualized_return, 0.0)}`}>
        {fmtPct(m.annualized_return)}
      </td>
      <td className={`px-2 py-1.5 text-xs ${metricColor(m.maximum_drawdown, 0.15, false)}`}>
        {fmtPct(m.maximum_drawdown)}
      </td>
      <td className={`px-2 py-1.5 text-xs ${metricColor(m.directional_accuracy, 0.5)}`}>
        {fmtPct(m.directional_accuracy)}
      </td>
      <td className={`px-2 py-1.5 text-xs ${metricColor(m.win_ratio, 0.5)}`}>
        {fmtPct(m.win_ratio)}
      </td>
      <td className="px-2 py-1.5 text-xs">{m.total_trades}</td>
      <td className="px-2 py-1.5">
        <span className="text-xs text-zinc-400">{formatTime(totalS)}</span>
      </td>
    </tr>
  );
}

export function LeaderboardTable({
  groups,
  bestKeys,
  selectMode,
  selectedRunIds,
  onToggleRuns,
  onSetSelected,
  experimentGroupKeys,
}: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("sortino_ratio");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const sorted = useMemo(() => {
    const copy = [...groups];
    copy.sort((a, b) => {
      const va = getSortValue(a, sortKey);
      const vb = getSortValue(b, sortKey);
      if (va < vb) return sortDir === "asc" ? -1 : 1;
      if (va > vb) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return copy;
  }, [groups, sortKey, sortDir]);

  const handleSort = useCallback(
    (key: SortKey) => {
      if (key === "rank" || key === "run") return;
      if (sortKey === key) {
        setSortDir((d) => (d === "desc" ? "asc" : "desc"));
      } else {
        setSortKey(key);
        setSortDir("desc");
      }
    },
    [sortKey]
  );

  const toggleExpand = useCallback((groupKey: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(groupKey)) {
        next.delete(groupKey);
      } else {
        next.add(groupKey);
      }
      return next;
    });
  }, []);

  const allVisibleRunIds = useMemo(
    () => groups.flatMap((g) => g.children.map((c) => c.run_id)),
    [groups],
  );
  const allSelected =
    selectMode &&
    allVisibleRunIds.length > 0 &&
    allVisibleRunIds.every((id) => selectedRunIds.has(id));
  const someSelected =
    selectMode && allVisibleRunIds.some((id) => selectedRunIds.has(id));
  const isPartial = someSelected && !allSelected;

  const onToggleAll = () => {
    if (allSelected || isPartial) onSetSelected(new Set());
    else onSetSelected(new Set(allVisibleRunIds));
  };

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white shadow-sm p-3">
      <div className="overflow-x-auto">
        <table className="w-full">
        <thead>
          <tr>
            {COLUMNS.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                className={`text-left px-2 py-2 text-[0.65rem] font-semibold uppercase tracking-wider text-zinc-500 border-b border-zinc-200 ${
                  col.numeric ? "cursor-pointer select-none hover:opacity-70" : ""
                }`}
              >
                {col.key === "rank" && selectMode ? (
                  <span className="flex items-center gap-1.5">
                    <input
                      ref={(el) => {
                        if (el) el.indeterminate = isPartial;
                      }}
                      type="checkbox"
                      checked={allSelected}
                      onChange={onToggleAll}
                      onClick={(e) => e.stopPropagation()}
                      className="cursor-pointer accent-zinc-700"
                      title={
                        allSelected
                          ? "Deselect all visible"
                          : "Select all visible groups"
                      }
                    />
                    {col.label}
                  </span>
                ) : (
                  col.label
                )}
                {sortKey === col.key && (
                  <span className="ml-1">
                    {sortDir === "desc" ? "↓" : "↑"}
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        {sorted.map((group, idx) => {
          const rank = idx + 1;
          const m = group.meanMetrics;
          const s = group.stdMetrics;
          const n = group.nSeeds;
          const cfg = group.configSummary;
          const isExpanded = expanded.has(group.groupKey);
          const isBest = bestKeys.has(group.groupKey);
          const borderCls = MODEL_BORDER_COLORS[group.modelType] ?? "";
          const topBorderCls = MODEL_TOP_BORDER[group.modelType] ?? "";
          const rankBg = rank <= 3 ? RANK_BADGES[rank - 1].row : "";

          const stockStr =
            cfg.stock_ids.length <= 3
              ? cfg.stock_ids.join(", ")
              : `${cfg.stock_ids.length} stocks`;

          const totalS = group.meanTrainingS + group.meanBacktestS;

          const testStart = cfg.test_start.slice(0, 7).replace("-", ".");
          const testEnd = cfg.test_end.slice(0, 7).replace("-", ".");

          const expKeys = group.experimentName
            ? experimentGroupKeys.get(group.experimentName)
            : undefined;
          const expIsLink = expKeys !== undefined && expKeys.length > 1;

          return (
            <Fragment key={group.groupKey}>
              <tbody>
                <tr
                  className={`border-l-4 ${borderCls} ${rankBg} ${
                    isBest ? `border-t-2 ${topBorderCls}` : ""
                  } border-b border-zinc-200 text-sm`}
                >
                  <td className="px-2 py-2">
                    <div className="flex items-center gap-1">
                      {selectMode && (() => {
                        const childIds = group.children.map((c) => c.run_id);
                        const allChildrenSelected = childIds.every((id) =>
                          selectedRunIds.has(id),
                        );
                        const someChildrenSelected = childIds.some((id) =>
                          selectedRunIds.has(id),
                        );
                        const childPartial =
                          someChildrenSelected && !allChildrenSelected;
                        return (
                          <input
                            type="checkbox"
                            checked={allChildrenSelected}
                            ref={(el) => {
                              if (el) el.indeterminate = childPartial;
                            }}
                            onChange={() => onToggleRuns(childIds)}
                            className="cursor-pointer accent-zinc-700"
                            title="Select all seeds in this group"
                          />
                        );
                      })()}
                      <RankCell rank={rank} />
                      {n > 1 && (
                        <button
                          onClick={() => toggleExpand(group.groupKey)}
                          className="text-zinc-400 hover:text-zinc-700 text-xs leading-none cursor-pointer"
                          title="Expand seeds"
                        >
                          {isExpanded ? "▼" : "▶"}
                        </button>
                      )}
                    </div>
                  </td>
                  <td className="px-2 py-2">
                    <div className="flex justify-between items-center">
                      {n === 1 ? (
                        <Link
                          to={`/runs/${group.children[0].run_dir}`}
                          className="text-blue-500 hover:underline font-semibold text-sm"
                        >
                          {group.modelName}
                        </Link>
                      ) : (
                        <span className="font-semibold text-sm">
                          {group.modelName}
                        </span>
                      )}
                      {n > 1 && (
                        <span className="inline-flex items-center rounded-full px-2 py-0.5 text-[0.65rem] font-medium bg-sky-100 text-sky-700">
                          n={n}
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-zinc-600 font-semibold">
                      {testStart} – {testEnd}
                    </div>
                    {group.experimentName && expIsLink && expKeys && (
                      <Link
                        to={compareUrl(expKeys)}
                        className="text-[0.65rem] text-zinc-400 hover:text-zinc-700 hover:underline"
                        title={`Compare all ${expKeys.length} groups in ${group.experimentName}`}
                      >
                        {group.experimentName}
                      </Link>
                    )}
                    {group.experimentName && !expIsLink && (
                      <div className="text-[0.65rem] text-zinc-400">
                        {group.experimentName}
                      </div>
                    )}
                    <ConfigBadges config={cfg} />
                  </td>
                  <td className="px-2 py-2 whitespace-nowrap">{group.timestamp}</td>
                  <td className="px-2 py-2">
                    <span
                      className="text-xs"
                      title={cfg.stock_ids.join(", ")}
                    >
                      {stockStr}
                    </span>
                  </td>
                  <td
                    className={`px-2 py-2 tabular-nums ${metricColor(m.sortino_ratio ?? 0, 1.0)}`}
                  >
                    <MetricWithStd
                      mean={m.sortino_ratio ?? 0}
                      std={s.sortino_ratio ?? 0}
                      nSeeds={n}
                      formatter={fmtRatio}
                    />
                  </td>
                  <td
                    className={`px-2 py-2 tabular-nums ${metricColor(m.sharpe_ratio ?? 0, 1.0)}`}
                  >
                    <MetricWithStd
                      mean={m.sharpe_ratio ?? 0}
                      std={s.sharpe_ratio ?? 0}
                      nSeeds={n}
                      formatter={fmtRatio}
                    />
                  </td>
                  <td
                    className={`px-2 py-2 tabular-nums ${metricColor(m.annualized_return ?? 0, 0.0)}`}
                  >
                    <MetricWithStd
                      mean={m.annualized_return ?? 0}
                      std={s.annualized_return ?? 0}
                      nSeeds={n}
                      formatter={fmtPct}
                    />
                  </td>
                  <td
                    className={`px-2 py-2 tabular-nums ${metricColor(m.maximum_drawdown ?? 0, 0.15, false)}`}
                  >
                    <MetricWithStd
                      mean={m.maximum_drawdown ?? 0}
                      std={s.maximum_drawdown ?? 0}
                      nSeeds={n}
                      formatter={fmtPct}
                    />
                  </td>
                  <td
                    className={`px-2 py-2 tabular-nums ${metricColor(m.directional_accuracy ?? 0, 0.5)}`}
                  >
                    <MetricWithStd
                      mean={m.directional_accuracy ?? 0}
                      std={s.directional_accuracy ?? 0}
                      nSeeds={n}
                      formatter={fmtPct}
                    />
                  </td>
                  <td
                    className={`px-2 py-2 tabular-nums ${metricColor(m.win_ratio ?? 0, 0.5)}`}
                  >
                    <MetricWithStd
                      mean={m.win_ratio ?? 0}
                      std={s.win_ratio ?? 0}
                      nSeeds={n}
                      formatter={fmtPct}
                    />
                  </td>
                  <td className="px-2 py-2 tabular-nums">
                    {(m.total_trades ?? 0).toFixed(0)}
                  </td>
                  <td className="px-2 py-2">
                    <span className="text-xs text-zinc-400">
                      {formatTime(totalS)}
                    </span>
                  </td>
                </tr>
                {isExpanded &&
                  group.children.map((child) => (
                    <SeedRow
                      key={child.run_id}
                      run={child}
                      selectMode={selectMode}
                      selected={selectedRunIds.has(child.run_id)}
                      onToggle={() => onToggleRuns([child.run_id])}
                    />
                  ))}
              </tbody>
            </Fragment>
          );
        })}
        </table>
      </div>
    </div>
  );
}
