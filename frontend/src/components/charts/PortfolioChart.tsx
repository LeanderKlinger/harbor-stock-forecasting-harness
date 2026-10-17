import Plot from "./Plot";
import type { EquityPoint } from "@/lib/types";
import { PLOTLY_LAYOUT } from "./theme";

interface Props {
  equityCurve: EquityPoint[];
  testStart: string;
  testEnd: string;
}

export function PortfolioChart({ equityCurve, testStart, testEnd }: Props) {
  return (
    <Plot
      data={[
        {
          x: equityCurve.map((p) => p.timestamp),
          y: equityCurve.map((p) => p.value),
          type: "scatter",
          mode: "lines",
          line: { color: "#3b82f6", width: 1.5 },
          hovertemplate: "%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
        },
      ]}
      layout={{
        ...PLOTLY_LAYOUT,
        title: { text: `Portfolio Value  (${testStart} to ${testEnd})` },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: "Value ($)" } },
        showlegend: false,
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: "100%", height: 400 }}
    />
  );
}
