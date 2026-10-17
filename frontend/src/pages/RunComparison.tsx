import { useMemo } from "react";
import { Link, useLocation } from "react-router";

import { aggregateRuns } from "@/lib/aggregation";
import {
  buildAggregatedEquity,
  buildPageTitle,
  expandBuyAndHoldAcrossHorizons,
  findSharedConfig,
  findVaryingAxes,
  formatAxisValue,
  splitTables,
  type ResultsTable,
} from "@/lib/experimentAggregation";
import { useBatchRunDetails, useRunsIndex } from "@/lib/hooks";
import type { AggregatedGroup, EquityCurveSubset } from "@/lib/types";

import { AggregatedEquityChart } from "@/components/charts/AggregatedEquityChart";
import { ResultsMatrix } from "@/components/ResultsMatrix";
import { SharedConfigStrip } from "@/components/SharedConfigStrip";

function labelForGroup(group: AggregatedGroup, table: ResultsTable): string {
  // Model + the inner-axis-value column label (e.g. "GRU [5, 10, 20]").
  if (table.innerAxes.length === 0) return group.modelType;
  const colLabel = table.innerAxes
    .map((a) => formatAxisValue(a, group.configSummary[a]))
    .join(", ");
  return `${group.modelType} ${colLabel}`;
}

export function RunComparison() {
  // Selection lives in the URL fragment, not the query — see compareUrl() rationale.
  // Fragment is read via useLocation().hash; URLSearchParams parses it the same way.
  const { hash } = useLocation();
  const requestedGroupKeys = useMemo(() => {
    const params = new URLSearchParams(hash.startsWith("#") ? hash.slice(1) : hash);
    return new Set(params.getAll("groups"));
  }, [hash]);

  const runsIndex = useRunsIndex();

  const allGroups = useMemo(
    () => (runsIndex.data ? aggregateRuns(runsIndex.data) : []),
    [runsIndex.data],
  );

  // Filter to selected first, THEN BuyAndHold-expand (so expansion only
  // duplicates against the horizons the user actually picked).
  const selectedGroups = useMemo(() => {
    const matched = allGroups.filter((g) => requestedGroupKeys.has(g.groupKey));
    return expandBuyAndHoldAcrossHorizons(matched);
  }, [allGroups, requestedGroupKeys]);

  // Fetch run_detail.json for every child of every selected group via one POST.
  // BuyAndHold-expanded copies share children with their original — dedupe by run_dir.
  const childRunDirs = useMemo(() => {
    const dirs = new Set<string>();
    for (const g of selectedGroups) {
      for (const c of g.children) dirs.add(c.run_dir);
    }
    return [...dirs];
  }, [selectedGroups]);

  const detailsQuery = useBatchRunDetails(childRunDirs, ["equity_curve"]);

  const detailsMap = useMemo(() => {
    const m = new Map<string, EquityCurveSubset>();
    if (!detailsQuery.data) return m;
    for (const [dir, detail] of Object.entries(detailsQuery.data)) {
      if (detail && "equity_curve" in detail) {
        m.set(dir, { equity_curve: detail.equity_curve });
      }
    }
    return m;
  }, [detailsQuery.data]);

  if (runsIndex.isLoading) {
    return <PageShell>Loading runs index…</PageShell>;
  }
  if (runsIndex.isError) {
    return (
      <PageShell>
        <div className="text-red-500">
          Failed to load runs index: {String(runsIndex.error)}
        </div>
      </PageShell>
    );
  }

  if (requestedGroupKeys.size === 0) {
    return (
      <PageShell>
        <div className="text-zinc-500">
          No groups selected. Open the leaderboard, click "Compare runs", select
          ≥2 group rows, then click "Compare N selected".
        </div>
      </PageShell>
    );
  }

  if (selectedGroups.length < 2) {
    return (
      <PageShell>
        <div className="text-zinc-500">
          Need at least 2 groups to compare ({selectedGroups.length} matched the URL).
        </div>
      </PageShell>
    );
  }

  const { title, meta } = buildPageTitle(selectedGroups);
  const shared = findSharedConfig(selectedGroups);
  const varyingAxes = findVaryingAxes(selectedGroups);
  const tables = splitTables(selectedGroups, varyingAxes);

  const detailsLoading = detailsQuery.isLoading;

  // Use the first group's test window + stocks for the strip — they're shared by definition.
  const refConfig = selectedGroups[0].configSummary;

  return (
    <PageShell>
      <div className="mb-2">
        <Link to="/" className="text-blue-500 hover:underline text-sm">
          ← Back to leaderboard
        </Link>
      </div>
      <h1 className="text-2xl font-semibold text-zinc-900">{title}</h1>
      <div className="text-sm text-zinc-500 mt-1 mb-4">{meta}</div>

      <SharedConfigStrip
        testStart={refConfig.test_start}
        testEnd={refConfig.test_end}
        stockIds={refConfig.stock_ids}
        shared={shared}
      />

      <section className="mt-8">
        <h2 className="text-lg font-semibold text-zinc-800 mb-2">Results</h2>
        {tables.map((table, i) => (
          <ResultsMatrix key={i} table={table} />
        ))}
      </section>

      <section className="mt-8">
        <h2 className="text-lg font-semibold text-zinc-800 mb-4">Equity Curves</h2>
        {detailsLoading && (
          <div className="text-sm text-zinc-500 mb-3">Loading equity curves…</div>
        )}
        {tables.map((table, i) => {
          const series = buildAggregatedEquity(table.groups, detailsMap, (g) =>
            labelForGroup(g, table),
          );
          if (series.length === 0) return null;
          return (
            <div
              key={i}
              className="mb-6 rounded-2xl border border-zinc-200 bg-white shadow-sm px-6 py-4"
            >
              <AggregatedEquityChart
                series={series}
                title={
                  table.title
                    ? `Equity Curves (normalized to 1.0, mean ± std) — ${table.title}`
                    : "Equity Curves (normalized to 1.0, mean ± std)"
                }
              />
            </div>
          );
        })}
      </section>
    </PageShell>
  );
}

function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="max-w-7xl mx-auto px-6 py-8 text-zinc-900">{children}</div>
  );
}
