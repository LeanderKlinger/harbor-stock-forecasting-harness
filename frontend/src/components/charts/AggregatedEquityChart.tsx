import type { Data } from "plotly.js";
import Plot from "./Plot";
import { PLOTLY_LAYOUT } from "./theme";
import { MODEL_COLORS } from "@/lib/constants";
import type { AggregatedEquity } from "@/lib/experimentAggregation";

interface Props {
  series: AggregatedEquity[];
  title?: string;
}

const FALLBACK_COLOR = "#71717a"; // zinc-500
const BAND_ALPHA = 0.12; // ±std fill opacity
const LEGEND_RIGHT_MARGIN = 200; // px of right margin reserved for legend
const CHART_HEIGHT = 400; // px

function colorWithAlpha(hex: string, alpha: number): string {
  // hex like "#8b5cf6"
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function AggregatedEquityChart({ series, title }: Props) {
  const traces: Data[] = [];

  for (const s of series) {
    const color = MODEL_COLORS[s.modelType] ?? FALLBACK_COLOR;
    const upper = s.mean.map((m, i) => m + s.std[i]);
    const lower = s.mean.map((m, i) => m - s.std[i]);

    // Band first so the mean line draws on top.
    const hasBand = s.std.some((v) => v > 0);
    if (hasBand) {
      traces.push({
        x: [...s.timestamps, ...[...s.timestamps].reverse()],
        y: [...upper, ...[...lower].reverse()],
        type: "scatter",
        mode: "lines",
        line: { color: "transparent", width: 0 },
        fill: "toself",
        fillcolor: colorWithAlpha(color, BAND_ALPHA),
        hoverinfo: "skip",
        showlegend: false,
        legendgroup: s.groupKey,
      });
    }

    traces.push({
      x: s.timestamps,
      y: s.mean,
      type: "scatter",
      mode: "lines",
      name: s.label,
      legendgroup: s.groupKey,
      line: { color, width: 1.5 },
      hovertemplate: "%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>",
    });
  }

  return (
    <Plot
      data={traces}
      layout={{
        ...PLOTLY_LAYOUT,
        title: title ? { text: title } : undefined,
        yaxis: {
          ...PLOTLY_LAYOUT.yaxis,
          title: { text: "Relative Value" },
        },
        showlegend: true,
        legend: { orientation: "v", x: 1.02, y: 1, xanchor: "left" },
        margin: { ...PLOTLY_LAYOUT.margin, r: LEGEND_RIGHT_MARGIN },
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: "100%", height: CHART_HEIGHT }}
    />
  );
}
