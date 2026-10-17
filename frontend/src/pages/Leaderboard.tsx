import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { useRunsIndex } from "@/lib/hooks";
import { aggregateRuns, bestPerModelType } from "@/lib/aggregation";
import { isIrrelevant } from "@/lib/constants";
import { compareUrl } from "@/lib/api";
import { useDeleteRuns } from "@/lib/mutations";
import { loadHiddenExperiments, saveHiddenExperiments } from "@/lib/storage";
import type { AggregatedGroup } from "@/lib/types";
import { FilterBar } from "@/components/FilterBar";
import { ModelLegend } from "@/components/ModelLegend";
import { LeaderboardTable } from "@/components/LeaderboardTable";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

export interface FilterState {
  headline: boolean | null;
  stockCount: "single" | "multi" | null;
  cost: "free" | "paid" | null;
  horizon: number | null;
  threshold: number | null;
  conviction: string | null;
  // String-encoded: "frozen" | "auto" | "<number>" | null (null = no filter).
  // Sentinel-string for `null`-config-value avoids the null-means-no-filter
  // and null-means-frozen-runs collision.
  retraining_cadence: string | null;
  sequence_length: number | null;
  feature_set: FeatureSet | null;
  since: string;
  to: string;
}

const DEFAULT_FILTERS: FilterState = {
  headline: true,
  stockCount: "multi",
  cost: "paid",
  horizon: null,
  threshold: null,
  conviction: null,
  retraining_cadence: null,
  sequence_length: null,
  feature_set: null,
  since: "",
  to: "",
};

function passesFilters(group: AggregatedGroup, filters: FilterState): boolean {
  const cfg = group.configSummary;
  const mt = group.modelType;

  if (
    filters.headline !== null &&
    !isIrrelevant(mt, "headline_only") &&
    cfg.headline_only !== filters.headline
  ) {
    return false;
  }

  if (filters.stockCount === "single" && cfg.stock_ids.length !== 1) {
    return false;
  }
  if (filters.stockCount === "multi" && cfg.stock_ids.length <= 1) {
    return false;
  }

  if (filters.cost !== null && !isIrrelevant(mt, "transaction_cost_pct")) {
    const hasCost = cfg.transaction_cost_pct > 0;
    if (filters.cost === "free" && hasCost) return false;
    if (filters.cost === "paid" && !hasCost) return false;
  }

  if (
    filters.horizon !== null &&
    !isIrrelevant(mt, "rebalance_horizon_days") &&
    cfg.rebalance_horizon_days !== filters.horizon
  ) {
    return false;
  }

  if (
    filters.threshold !== null &&
    !isIrrelevant(mt, "min_predicted_return") &&
    cfg.min_predicted_return !== filters.threshold
  ) {
    return false;
  }

  if (
    filters.conviction !== null &&
    !isIrrelevant(mt, "conviction_mode") &&
    cfg.conviction_mode !== filters.conviction
  ) {
    return false;
  }

  if (
    filters.retraining_cadence !== null &&
    !isIrrelevant(mt, "retraining_cadence_days")
  ) {
    const v = cfg.retraining_cadence_days;
    const encoded =
      v === null || v === undefined ? "frozen" : v === "auto" ? "auto" : String(v);
    if (encoded !== filters.retraining_cadence) return false;
  }

  if (
    filters.sequence_length !== null &&
    !isIrrelevant(mt, "sequence_length")
  ) {
    // undefined = pre-lift entry, treat as non-match when a specific filter is set.
    if (cfg.sequence_length !== filters.sequence_length) return false;
  }

  if (filters.feature_set !== null && !isIrrelevant(mt, "feature_set")) {
    // undefined = pre-flag entry = the canonical full vector (Config default).
    const v = cfg.feature_set ?? "full";
    if (v !== filters.feature_set) return false;
  }

  if (filters.since && group.timestamp < filters.since) return false;
  if (filters.to && group.timestamp > filters.to) return false;

  return true;
}

export function Leaderboard() {
  const navigate = useNavigate();
  const { data: entries, isLoading, error } = useRunsIndex();
  const deleteRunsMutation = useDeleteRuns();
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [viewMode, setViewMode] = useState<"all" | "byModel">("all");
  // Multi-experiment selection lives in the URL (?experiments=a,b) so a
  // composed view — e.g. a grid plus its companion experiments — is shareable
  // and survives reloads. Absent = canonical default; empty = "all experiments"
  // (minus globally hidden).
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedExperiments = useMemo(() => {
    const raw = searchParams.get("experiments");
    // Absent param -> default to the canonical experiment; explicit empty
    // string (user deselected everything) -> all experiments.
    if (raw === null) return new Set(["big_eval_canon"]);
    return new Set(raw.split(",").filter(Boolean));
  }, [searchParams]);

  const setSelectedExperiments = useCallback(
    (next: Set<string>) => {
      setSearchParams(
        (prev) => {
          const p = new URLSearchParams(prev);
          p.set("experiments", [...next].sort().join(","));
          return p;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const toggleExperiment = useCallback(
    (name: string) => {
      const next = new Set(selectedExperiments);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      setSelectedExperiments(next);
    },
    [selectedExperiments, setSelectedExperiments],
  );

  // Hide/unhide operates on a single experiment; only offered when exactly one
  // is selected (matches the old single-select behavior).
  const soleExperiment =
    selectedExperiments.size === 1 ? [...selectedExperiments][0] : null;
  const [selectMode, setSelectMode] = useState(false);
  const [selectedRunIds, setSelectedRunIds] = useState<Set<string>>(new Set());
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [hiddenExperiments, setHiddenExperiments] = useState<Set<string>>(() =>
    loadHiddenExperiments()
  );

  useEffect(() => {
    saveHiddenExperiments(hiddenExperiments);
  }, [hiddenExperiments]);

  const hideExperiment = useCallback((name: string) => {
    setHiddenExperiments((prev) => new Set(prev).add(name));
  }, []);

  const unhideExperiment = useCallback((name: string) => {
    setHiddenExperiments((prev) => {
      const next = new Set(prev);
      next.delete(name);
      return next;
    });
  }, []);

  const restoreHiddenExperiments = useCallback(() => {
    setHiddenExperiments(new Set());
  }, []);

  const toggleSelectMode = useCallback(() => {
    setSelectMode((m) => !m);
    setSelectedRunIds(new Set());
  }, []);

  // Toggle add/remove for a list of run_ids. If every passed id is already in
  // the set → remove them all; else → add the missing ones. Used by both the
  // parent-row checkbox (passes all children) and seed-row checkbox (one id).
  const toggleRuns = useCallback((runIds: string[]) => {
    setSelectedRunIds((prev) => {
      const next = new Set(prev);
      const allPresent = runIds.every((id) => next.has(id));
      if (allPresent) {
        for (const id of runIds) next.delete(id);
      } else {
        for (const id of runIds) next.add(id);
      }
      return next;
    });
  }, []);

  const experimentNames = useMemo(() => {
    if (!entries) return [];
    const names = new Set(entries.map((e) => e.experiment_name).filter(Boolean));
    return [...names].sort() as string[];
  }, [entries]);

  // When no experiment is selected, hide runs whose experiment_name was
  // globally hidden via localStorage. When experiments ARE selected, the
  // working set is the UNION of their runs and ALL of them show (the hide is
  // overrideable by explicit selection).
  const filteredByExperiment = useMemo(() => {
    if (!entries) return [];
    if (selectedExperiments.size > 0) {
      return entries.filter(
        (e) => e.experiment_name && selectedExperiments.has(e.experiment_name)
      );
    }
    if (hiddenExperiments.size === 0) return entries;
    return entries.filter(
      (e) => !e.experiment_name || !hiddenExperiments.has(e.experiment_name)
    );
  }, [entries, selectedExperiments, hiddenExperiments]);

  const aggregated = useMemo(
    () => aggregateRuns(filteredByExperiment),
    [filteredByExperiment]
  );

  // run_id → groupKey lookup so Compare can derive groupKeys from selectedRunIds.
  const runIdToGroupKey = useMemo(() => {
    const m = new Map<string, string>();
    for (const g of aggregated) {
      for (const c of g.children) m.set(c.run_id, g.groupKey);
    }
    return m;
  }, [aggregated]);

  // Compare needs ≥2 distinct groups from the selection.
  const selectedGroupKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const id of selectedRunIds) {
      const k = runIdToGroupKey.get(id);
      if (k) keys.add(k);
    }
    return keys;
  }, [selectedRunIds, runIdToGroupKey]);

  const submitCompare = useCallback(() => {
    if (selectedGroupKeys.size < 2) return;
    navigate(compareUrl(selectedGroupKeys));
  }, [navigate, selectedGroupKeys]);

  const submitDelete = useCallback(() => {
    if (selectedRunIds.size === 0) return;
    deleteRunsMutation.mutate([...selectedRunIds]);
    setDeleteDialogOpen(false);
    setSelectMode(false);
    setSelectedRunIds(new Set());
  }, [selectedRunIds, deleteRunsMutation]);

  const modelTypes = useMemo(
    () => new Set(aggregated.map((g) => g.modelType)),
    [aggregated]
  );

  const filtered = useMemo(
    () => aggregated.filter((g) => passesFilters(g, filters)),
    [aggregated, filters]
  );

  const bestKeys = useMemo(() => bestPerModelType(filtered), [filtered]);

  // experimentName → all groupKeys in that experiment **passing current filters**.
  // Computed from `filtered`, not `aggregated`, so the experiment-name link only
  // includes groups the user is actually looking at.
  const experimentGroupKeys = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const g of filtered) {
      if (!g.experimentName) continue;
      const arr = map.get(g.experimentName) ?? [];
      arr.push(g.groupKey);
      map.set(g.experimentName, arr);
    }
    return map;
  }, [filtered]);

  const visible =
    viewMode === "byModel"
      ? filtered.filter((g) => bestKeys.has(g.groupKey))
      : filtered;

  // Counted from `visible` so the caption always matches the table (in
  // "By Model" view `filtered` would over-report). Filter exclusion is
  // reported separately from the byModel reduction.
  const shownRuns = visible.reduce((n, g) => n + g.children.length, 0);
  const excludedGroups = aggregated.length - filtered.length;

  if (isLoading) {
    return (
      <div className="min-h-screen p-6 md:p-10 max-w-400 mx-auto">
        <p className="text-zinc-500">Loading runs index...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen p-6 md:p-10 max-w-400 mx-auto">
        <p className="text-red-500">Error: {error.message}</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 md:p-10 max-w-400 mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">
          Pipeline Leaderboard
        </h1>
        <p className="text-zinc-500 mt-1">
          Cross-seed aggregated runs ranked by mean Sortino ratio.
        </p>
      </div>

      <div className="flex items-center gap-4 mb-4">
        <div className="flex gap-1">
          <button
            onClick={() => setViewMode("all")}
            className={`inline-flex items-center rounded-full px-1.5 py-px text-[0.625rem] font-medium leading-4 cursor-pointer ${
              viewMode === "all"
                ? "bg-zinc-700 text-white"
                : "bg-zinc-100 text-zinc-600"
            }`}
          >
            All Runs
          </button>
          <button
            onClick={() => setViewMode("byModel")}
            className={`inline-flex items-center rounded-full px-1.5 py-px text-[0.625rem] font-medium leading-4 cursor-pointer ${
              viewMode === "byModel"
                ? "bg-zinc-700 text-white"
                : "bg-zinc-100 text-zinc-600"
            }`}
          >
            By Model
          </button>
        </div>

        {experimentNames.length > 1 && (
          <>
            <Popover>
              <PopoverTrigger asChild>
                <button className="text-xs bg-zinc-100 border border-zinc-300 rounded px-1.5 py-0.5 text-zinc-600 cursor-pointer hover:bg-zinc-50">
                  {selectedExperiments.size === 0
                    ? "All experiments"
                    : soleExperiment ?? `${selectedExperiments.size} experiments`}
                  {" ▾"}
                </button>
              </PopoverTrigger>
              <PopoverContent align="start" className="w-auto max-w-md p-2">
                <div className="flex flex-col gap-1 max-h-80 overflow-y-auto">
                  {experimentNames.map((name) => (
                    <label
                      key={name}
                      className="flex items-center gap-2 text-xs text-zinc-700 px-1 py-0.5 rounded hover:bg-zinc-50 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={selectedExperiments.has(name)}
                        onChange={() => toggleExperiment(name)}
                      />
                      <span className="truncate">
                        {name}
                        {hiddenExperiments.has(name) ? " (hidden)" : ""}
                      </span>
                    </label>
                  ))}
                </div>
                {selectedExperiments.size > 0 && (
                  <button
                    onClick={() => setSelectedExperiments(new Set())}
                    className="mt-2 w-full rounded border border-zinc-300 bg-white px-2 py-0.5 text-xs font-medium text-zinc-600 shadow-sm hover:bg-zinc-50 cursor-pointer"
                  >
                    Clear ({selectedExperiments.size} selected)
                  </button>
                )}
              </PopoverContent>
            </Popover>
            {soleExperiment && !hiddenExperiments.has(soleExperiment) && (
              <button
                onClick={() => {
                  hideExperiment(soleExperiment);
                  setSelectedExperiments(new Set());
                }}
                className="rounded border border-zinc-300 bg-white px-2 py-0.5 text-xs font-medium text-zinc-600 shadow-sm hover:bg-zinc-50 cursor-pointer"
                title="Hide this experiment from the unfiltered leaderboard"
              >
                Hide globally
              </button>
            )}
            {soleExperiment && hiddenExperiments.has(soleExperiment) && (
              <button
                onClick={() => unhideExperiment(soleExperiment)}
                className="rounded border border-zinc-300 bg-white px-2 py-0.5 text-xs font-medium text-zinc-600 shadow-sm hover:bg-zinc-50 cursor-pointer"
                title="Unhide this experiment"
              >
                Unhide
              </button>
            )}
          </>
        )}

        {!selectMode ? (
          <button
            onClick={toggleSelectMode}
            className="rounded border border-zinc-300 bg-white px-2 py-0.5 text-xs font-medium text-zinc-700 shadow-sm hover:bg-zinc-50 cursor-pointer"
          >
            Select runs
          </button>
        ) : (
          <div className="flex gap-1.5">
            <button
              onClick={submitCompare}
              disabled={selectedGroupKeys.size < 2}
              className={`rounded px-2 py-0.5 text-xs font-medium shadow-sm border ${
                selectedGroupKeys.size < 2
                  ? "bg-zinc-100 text-zinc-400 cursor-not-allowed border-zinc-200"
                  : "bg-emerald-600 text-white hover:bg-emerald-700 cursor-pointer border-emerald-700"
              }`}
            >
              Compare {selectedGroupKeys.size} group{selectedGroupKeys.size === 1 ? "" : "s"}
            </button>
            <button
              onClick={() => setDeleteDialogOpen(true)}
              disabled={selectedRunIds.size === 0}
              className={`rounded px-2 py-0.5 text-xs font-medium shadow-sm border ${
                selectedRunIds.size === 0
                  ? "bg-zinc-100 text-zinc-400 cursor-not-allowed border-zinc-200"
                  : "bg-rose-600 text-white hover:bg-rose-700 cursor-pointer border-rose-700"
              }`}
            >
              Delete {selectedRunIds.size} run{selectedRunIds.size === 1 ? "" : "s"}
            </button>
            <button
              onClick={toggleSelectMode}
              className="rounded border border-zinc-300 bg-white px-2 py-0.5 text-xs font-medium text-zinc-700 shadow-sm hover:bg-zinc-50 cursor-pointer"
            >
              Cancel
            </button>
          </div>
        )}
      </div>

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete {selectedRunIds.size} runs?</DialogTitle>
            <DialogDescription>
              This removes each run directory from disk and rewrites
              runs_index.json. Not recoverable through the UI.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              autoFocus
              onClick={() => setDeleteDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button variant="destructive" onClick={submitDelete}>
              Delete {selectedRunIds.size} runs
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {hiddenExperiments.size > 0 && selectedExperiments.size === 0 && (
        <div className="mb-3 text-xs text-zinc-500">
          Hidden:{" "}
          <span className="text-zinc-700">
            {[...hiddenExperiments].sort().join(", ")}
          </span>
          <button
            onClick={restoreHiddenExperiments}
            className="ml-2 text-blue-500 hover:underline cursor-pointer"
          >
            restore all
          </button>
        </div>
      )}

      <FilterBar
        filters={filters}
        onChange={setFilters}
        entries={filteredByExperiment}
      />
      <ModelLegend modelTypes={modelTypes} />

      {/* Always rendered, including "0 groups" — the empty case is the most
          valuable feedback (the table itself has no empty state). */}
      <p className="mt-2 mb-1 text-xs text-zinc-500">
        {visible.length} groups · {shownRuns} runs
        {excludedGroups > 0 && (
          <span className="text-zinc-400">
            {" "}
            (filters exclude {excludedGroups} of {aggregated.length} groups)
          </span>
        )}
      </p>

      <LeaderboardTable
        groups={visible}
        bestKeys={bestKeys}
        selectMode={selectMode}
        selectedRunIds={selectedRunIds}
        onToggleRuns={toggleRuns}
        onSetSelected={setSelectedRunIds}
        experimentGroupKeys={experimentGroupKeys}
      />
    </div>
  );
}
