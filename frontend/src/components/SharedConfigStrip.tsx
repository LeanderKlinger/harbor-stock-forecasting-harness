import type { ConfigSummary } from "@/lib/types";
import {
  formatAxisValue,
  type Axis,
  type SharedConfigEntry,
} from "@/lib/experimentAggregation";

interface Props {
  testStart: string;
  testEnd: string;
  stockIds: string[];
  shared: SharedConfigEntry[];
}

// test_start/test_end/stock_ids are rendered as the leading meta and stripped
// from the trailing badge list — they're shown more prominently.
const LEADING_FIELDS = new Set<Axis>(["test_start", "test_end", "stock_ids"]);

// Field-specific label for the strip. Self-describing values (top-4, headline)
// don't get a prefix; bare values (cost amount, threshold pct) do.
function fieldLabel<K extends Axis>(field: K, value: ConfigSummary[K]): string {
  switch (field) {
    case "headline_only":
      return value ? "headline" : "article";
    case "top_k":
      return formatAxisValue(field, value); // already "top-4"
    case "transaction_cost_pct":
      return `cost ${formatAxisValue(field, value)}`;
    case "early_stopping_patience":
      return `early stop ${formatAxisValue(field, value)}`;
    case "min_predicted_return":
      return value === 0 ? "no threshold" : `threshold ${formatAxisValue(field, value)}`;
    case "conviction_mode":
      return String(value);
    case "rebalance_horizon_days":
      return `horizon ${formatAxisValue(field, value)}`;
    case "sentiment_ema_spans":
      return `ema ${formatAxisValue(field, value)}`;
    case "market_condition":
      return String(value);
    case "retraining_cadence_days":
      return value === null
        ? "no retrain"
        : `retrain ${formatAxisValue(field, value)}`;
    default:
      return formatAxisValue(field, value);
  }
}

export function SharedConfigStrip({ testStart, testEnd, stockIds, shared }: Props) {
  const trailing = shared.filter((s) => !LEADING_FIELDS.has(s.field));
  return (
    <div className="rounded-2xl border border-zinc-200 bg-white shadow-sm px-6 py-3 text-sm text-zinc-500 leading-tight">
      <span className="font-medium text-zinc-600 uppercase tracking-wider text-xs">
        Shared Config:
      </span>{" "}
      <span>
        Test: {testStart.slice(0, 10)} to {testEnd.slice(0, 10)}
      </span>{" "}
      <span className="text-zinc-300">|</span>{" "}
      <span>{stockIds.length} stocks</span>
      {trailing.map((entry) => (
        <span key={entry.field}>
          {" "}
          <span className="text-zinc-300">|</span>{" "}
          <span>{fieldLabel(entry.field, entry.value)}</span>
        </span>
      ))}
    </div>
  );
}
