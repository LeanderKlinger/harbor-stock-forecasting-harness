import type {
  RunIndexEntry,
  RunDetail,
  RunDetailBatchValue,
  StockPrice,
} from "./types";

const DATA_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:3001";

async function fetchJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${DATA_BASE}${path}`, { signal });
  if (!res.ok) throw new Error(`GET ${path}: ${res.status}`);
  return res.json();
}

export const queryKeys = {
  runsIndex: ["runsIndex"] as const,
  runDetail: (runDir: string) => ["runDetail", runDir] as const,
  stockPrices: (stockId: string) => ["stockPrices", stockId] as const,
  // Joined-string keys keep this bounded — spreading 1500 dirs into the key
  // array makes TanStack Query re-hash a huge tuple every render.
  runDetailsBatch: (dirs: string[], fields?: string[]) =>
    ["runDetailsBatch", dirs.join(","), (fields ?? []).join(",")] as const,
};

// runs_index.json is mutable — deletes / new runs change it. Bypass the
// browser's HTTP cache (TanStack Query handles freshness above) so a recent
// mutation doesn't get served the pre-mutation copy on the next reload.
export const fetchRunsIndex = async (): Promise<RunIndexEntry[]> => {
  const res = await fetch(`${DATA_BASE}/runs_index.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`GET /runs_index.json: ${res.status}`);
  return res.json();
};

export const fetchRunDetail = (runDir: string, signal?: AbortSignal) =>
  fetchJson<RunDetail>(`/${runDir}/run_detail.json`, signal);

export const fetchStockPrices = (stockId: string) =>
  fetchJson<StockPrice[]>(`/.stock_prices/${stockId}.json`);

export async function fetchRunDetailsBatch(
  runDirs: string[],
  fields?: string[],
): Promise<Record<string, RunDetailBatchValue>> {
  const res = await fetch(`${DATA_BASE}/api/run_details`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_dirs: runDirs, fields }),
  });
  if (!res.ok) throw new Error(`POST /api/run_details: ${res.status}`);
  return res.json();
}

interface DeleteRunsResponse {
  deleted: string[];
  not_found: string[];
  failed: string[];
}

export async function deleteRuns(
  runIds: string[],
): Promise<DeleteRunsResponse> {
  const res = await fetch(`${DATA_BASE}/api/runs`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_ids: runIds }),
  });
  if (!res.ok) throw new Error(`DELETE /api/runs: ${res.status}`);
  return res.json();
}

/** Build /compare#groups=k1&groups=k2&... URL.
 *
 * Selection lives in the URL fragment (not a query string) because at big_eval
 * scale (250+ groups × ~1-2KB groupKey each) the URL exceeds Node's 8KB request
 * line limit and the proxy returns HTTP 431 before the page even renders.
 * Fragments are never sent to the server — the browser handles them entirely
 * client-side, so URL length isn't a transport concern.
 */
export function compareUrl(groupKeys: Iterable<string>): string {
  const params = new URLSearchParams();
  for (const k of groupKeys) params.append("groups", k);
  return `/compare#${params.toString()}`;
}

export function sliceStockPrices(
  prices: StockPrice[],
  start: string,
  end: string
): StockPrice[] {
  return prices.filter((p) => p.timestamp >= start && p.timestamp <= end);
}
