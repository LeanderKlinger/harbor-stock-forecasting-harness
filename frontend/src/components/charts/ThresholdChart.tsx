import Plot from "./Plot";
import type { ThresholdPoint } from "@/lib/types";
import { PLOTLY_LAYOUT } from "./theme";

interface Props {
  thresholdData: ThresholdPoint[];
  baseThreshold: number;
  testStart: string;
  testEnd: string;
}

export function ThresholdChart({
  thresholdData,
  baseThreshold,
  testStart,
  testEnd,
}: Props) {
  return (
    <Plot
      data={[
        {
          x: thresholdData.map((p) => p.timestamp),
          y: thresholdData.map((p) => p.value),
          type: "scatter",
          mode: "lines",
          line: { color: "#8b5cf6", width: 1.5, shape: "hv" },
          name: "Effective threshold",
          hovertemplate: "%{x|%Y-%m-%d}<br>%{y:.5f}<extra></extra>",
        },
      ]}
      layout={{
        ...PLOTLY_LAYOUT,
        title: { text: `Vol-Scaled Conviction Threshold  (${testStart} to ${testEnd})` },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: "Threshold" } },
        showlegend: true,
        height: 350,
        shapes: [
          {
            type: "line",
            x0: testStart,
            x1: testEnd,
            y0: baseThreshold,
            y1: baseThreshold,
            line: { color: "#a1a1aa", width: 1, dash: "dash" },
          },
        ],
        annotations: [
          {
            x: testEnd,
            y: baseThreshold,
            text: `Base (${baseThreshold})`,
            showarrow: false,
            xanchor: "right",
            yanchor: "bottom",
            font: { color: "#71717a", size: 11 },
          },
        ],
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: "100%", height: 350 }}
    />
  );
}
