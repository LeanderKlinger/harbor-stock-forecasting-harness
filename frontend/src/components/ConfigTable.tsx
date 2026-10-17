import type { RunDetailConfig } from "@/lib/types";
import { fmtPct } from "@/lib/constants";

interface Props {
  config: RunDetailConfig;
}

export function ConfigTable({ config }: Props) {
  const rows: [string, string][] = [
    [
      "Test Period",
      `${config.test_start} to ${config.test_end}`,
    ],
    [
      "Training Period",
      `${config.training_start} to ${config.training_end}`,
    ],
    ["Market Condition", config.market_condition],
    ["Headline Only", String(config.headline_only)],
    ["Rebalance Horizon", `${config.rebalance_horizon_days} day(s)`],
    ["EMA Spans", String(config.sentiment_ema_spans)],
    ["Transaction Cost", fmtPct(config.transaction_cost_pct)],
    ["Risk-Free Rate", fmtPct(config.risk_free_rate)],
    ["Stocks", config.stock_ids.join(", ")],
  ];

  return (
    <div className="border border-zinc-200 rounded-xl p-6 my-8">
      <h3 className="text-sm font-semibold text-zinc-500 uppercase tracking-wider mb-4">
        Configuration
      </h3>
      <table className="w-full">
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label} className="border-b border-zinc-100 last:border-0">
              <td className="text-zinc-400 text-sm py-2 pr-4">{label}</td>
              <td className="font-medium py-2">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
