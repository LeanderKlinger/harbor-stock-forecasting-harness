import type { Metrics } from "@/lib/types";
import { fmtPct, fmtRatio, metricColor } from "@/lib/constants";

interface Props {
  metrics: Metrics;
}

interface MetricDef {
  label: string;
  value: string;
  colorClass: string | null;
}

function buildGroups(m: Metrics): [string, MetricDef[]][] {
  return [
    [
      "Error Metrics",
      [
        { label: "RMSE", value: m.rmse.toFixed(6), colorClass: null },
        { label: "MAE", value: m.mae.toFixed(6), colorClass: null },
        { label: "MAPE", value: fmtPct(m.mape), colorClass: null },
      ],
    ],
    [
      "Directional Accuracy",
      [
        {
          label: "Accuracy",
          value: fmtPct(m.directional_accuracy),
          colorClass: metricColor(m.directional_accuracy, 0.5),
        },
        { label: "TP / FP", value: `${m.tp} / ${m.fp}`, colorClass: null },
        { label: "TN / FN", value: `${m.tn} / ${m.fn}`, colorClass: null },
      ],
    ],
    [
      "Returns & Risk",
      [
        {
          label: "Total Return",
          value: fmtPct(m.total_return),
          colorClass: metricColor(m.total_return, 0.0),
        },
        {
          label: "Annualized",
          value: fmtPct(m.annualized_return),
          colorClass: metricColor(m.annualized_return, 0.0),
        },
        {
          label: "Sharpe",
          value: fmtRatio(m.sharpe_ratio),
          colorClass: metricColor(m.sharpe_ratio, 1.0),
        },
        {
          label: "Sortino",
          value: fmtRatio(m.sortino_ratio),
          colorClass: metricColor(m.sortino_ratio, 1.0),
        },
        {
          label: "Max Drawdown",
          value: fmtPct(m.maximum_drawdown),
          colorClass: metricColor(m.maximum_drawdown, 0.15, false),
        },
        { label: "CVaR (95%)", value: m.cvar.toFixed(6), colorClass: null },
        {
          label: "Volatility",
          value: m.volatility.toFixed(4),
          colorClass: null,
        },
      ],
    ],
    [
      "Trade Statistics",
      [
        {
          label: "Total Trades",
          value: String(m.total_trades),
          colorClass: null,
        },
        {
          label: "Win Ratio",
          value: fmtPct(m.win_ratio),
          colorClass: metricColor(m.win_ratio, 0.5),
        },
        {
          label: "Avg Trade Return",
          value: fmtPct(m.avg_trade_return),
          colorClass: metricColor(m.avg_trade_return, 0.0),
        },
        {
          label: "Transaction Costs",
          value: m.total_transaction_cost.toFixed(4),
          colorClass: null,
        },
      ],
    ],
  ];
}

export function MetricsCards({ metrics }: Props) {
  const groups = buildGroups(metrics);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
      {groups.map(([groupName, items]) => (
        <div
          key={groupName}
          className="border border-zinc-200 rounded-xl p-6"
        >
          <h3 className="text-sm font-semibold text-zinc-500 uppercase tracking-wider mb-4">
            {groupName}
          </h3>
          <div className="space-y-3">
            {items.map((item) => (
              <div
                key={item.label}
                className="flex justify-between items-baseline"
              >
                <span className="text-sm text-zinc-400">{item.label}</span>
                <span
                  className={`text-lg font-semibold tabular-nums ${item.colorClass ?? ""}`}
                >
                  {item.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
