import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  queryKeys,
  fetchRunsIndex,
  fetchRunDetail,
  fetchRunDetailsBatch,
  fetchStockPrices,
} from "./api";

export function useRunsIndex() {
  return useQuery({
    queryKey: queryKeys.runsIndex,
    queryFn: fetchRunsIndex,
    staleTime: 5 * 60 * 1000, // 5 min — re-fetch if user refreshes after new runs
  });
}

export function useRunDetail(runDir: string) {
  return useQuery({
    queryKey: queryKeys.runDetail(runDir),
    queryFn: () => fetchRunDetail(runDir),
    staleTime: Infinity, // run data never changes
  });
}

export function useStockPrices(stockId: string) {
  return useQuery({
    queryKey: queryKeys.stockPrices(stockId),
    queryFn: () => fetchStockPrices(stockId),
    staleTime: Infinity, // price history is static
  });
}

export function useBatchRunDetails(runDirs: string[], fields?: string[]) {
  // Sort for a stable query key — order shouldn't affect cache identity.
  const sortedDirs = useMemo(() => [...runDirs].sort(), [runDirs]);
  return useQuery({
    queryKey: queryKeys.runDetailsBatch(sortedDirs, fields),
    queryFn: () => fetchRunDetailsBatch(sortedDirs, fields),
    enabled: sortedDirs.length > 0,
    staleTime: Infinity,
    gcTime: 1000 * 60 * 5, // 5 min — release after navigating away
    retry: false, // don't replay 1000-element POSTs on flake
  });
}
