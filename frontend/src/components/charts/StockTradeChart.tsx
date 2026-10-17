import Plot from "./Plot";
import type { Data, Shape } from "plotly.js";
import type { StockChartData } from "@/lib/types";
import { useStockPrices } from "@/lib/hooks";
import { sliceStockPrices } from "@/lib/api";
import { PLOTLY_LAYOUT } from "./theme";

interface Props {
  stockId: string;
  chartData: StockChartData;
  testStart: string;
  testEnd: string;
}

const GREEN_FILL = "rgba(34,197,94,0.20)";
const RED_FILL = "rgba(239,68,68,0.20)";
const GREEN_OPEN = "rgba(34,197,94,0.10)";
const RED_OPEN = "rgba(239,68,68,0.10)";
const GREY_FILL = "rgba(0,0,0,0.06)";
const GREEN_LINE = "#16a34a";
const RED_LINE = "#dc2626";

function makePolygon(
  timestamps: string[],
  prices: number[],
  fillcolor: string,
  name: string,
  showlegend: boolean,
  lineStyle?: { width: number; color: string; dash: string }
): Data {
  // Forward along price line, then back along y=0 baseline
  const polyX = [...timestamps, timestamps.at(-1)!, timestamps[0]];
  const polyY = [...prices, 0, 0];

  return {
    x: polyX,
    y: polyY,
    fill: "toself",
    fillcolor,
    line: lineStyle ?? { width: 0, color: "rgba(0,0,0,0)" },
    mode: "lines",
    name,
    showlegend,
    hoverinfo: "skip",
    type: "scatter",
  } as Data;
}

function sliceByInterval(
  allTs: string[],
  allPrices: number[],
  start: string,
  end: string
): { ts: string[]; prices: number[] } {
  const ts: string[] = [];
  const prices: number[] = [];
  for (let i = 0; i < allTs.length; i++) {
    if (allTs[i] >= start && allTs[i] <= end) {
      ts.push(allTs[i]);
      prices.push(allPrices[i]);
    }
  }
  return { ts, prices };
}

export function StockTradeChart({
  stockId,
  chartData,
  testStart,
  testEnd,
}: Props) {
  const { data: allPrices, isLoading } = useStockPrices(stockId);

  if (isLoading || !allPrices) {
    return <p className="text-zinc-400 text-sm">Loading {stockId} prices...</p>;
  }

  const sliced = sliceStockPrices(allPrices, testStart, testEnd);
  const timestamps = sliced.map((p) => p.timestamp);
  const prices = sliced.map((p) => p.close);

  if (timestamps.length === 0) {
    return <p className="text-zinc-400 text-sm">No price data for {stockId}</p>;
  }

  const traces: Data[] = [];
  const legendShown = new Set<string>();

  // Grey fills for not-held periods
  for (const interval of chartData.not_held_intervals) {
    const seg = sliceByInterval(timestamps, prices, interval.start, interval.end);
    if (seg.ts.length === 0) continue;
    const show = !legendShown.has("Not held");
    legendShown.add("Not held");
    traces.push(makePolygon(seg.ts, seg.prices, GREY_FILL, "Not held", show));
  }

  // Colored fills for held intervals
  for (const interval of chartData.held_intervals) {
    const seg = sliceByInterval(timestamps, prices, interval.start, interval.end);
    if (seg.ts.length === 0) continue;

    let color: string;
    let label: string;
    let lineStyle: { width: number; color: string; dash: string } | undefined;

    if (interval.closed) {
      color = interval.profitable ? GREEN_FILL : RED_FILL;
      label = interval.profitable ? "Held (profit)" : "Held (loss)";
    } else {
      color = interval.profitable ? GREEN_OPEN : RED_OPEN;
      label = "Open (unrealized)";
      lineStyle = {
        width: 1,
        color: interval.profitable ? GREEN_LINE : RED_LINE,
        dash: "dot",
      };
    }

    const show = !legendShown.has(label);
    legendShown.add(label);
    traces.push(makePolygon(seg.ts, seg.prices, color, label, show, lineStyle));
  }

  // Main close price line
  traces.push({
    x: timestamps,
    y: prices,
    type: "scatter",
    mode: "lines",
    line: { color: "#1a1a1a", width: 1.3 },
    name: "Close",
    showlegend: false,
    hovertemplate: "%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
  });

  // Trade vertical line shapes
  const yMax = Math.max(...prices);
  const shapes: Partial<Shape>[] = [];
  let buyLegendShown = false;
  let sellLegendShown = false;

  for (const trade of chartData.trades) {
    const isBuy = trade.action === "BUY";
    shapes.push({
      type: "line",
      x0: trade.timestamp,
      x1: trade.timestamp,
      y0: 0,
      y1: yMax,
      line: {
        color: isBuy ? GREEN_LINE : RED_LINE,
        width: 0.7,
        dash: isBuy ? "solid" : "dash",
      },
      opacity: 0.5,
    });

    // Invisible trace for legend
    if (isBuy && !buyLegendShown) {
      traces.push({
        x: [null],
        y: [null],
        type: "scatter",
        mode: "lines",
        line: { color: GREEN_LINE, width: 1.5 },
        name: "BUY",
        showlegend: true,
      } as Data);
      buyLegendShown = true;
    }
    if (!isBuy && !sellLegendShown) {
      traces.push({
        x: [null],
        y: [null],
        type: "scatter",
        mode: "lines",
        line: { color: RED_LINE, width: 1.5, dash: "dash" },
        name: "SELL",
        showlegend: true,
      } as Data);
      sellLegendShown = true;
    }
  }

  return (
    <Plot
      data={traces}
      layout={{
        ...PLOTLY_LAYOUT,
        title: { text: `${stockId}  (${testStart} to ${testEnd})` },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: "Close Price ($)" } },
        shapes,
        legend: {
          orientation: "h",
          yanchor: "top",
          y: -0.12,
          xanchor: "left",
          x: 0,
          font: { size: 10 },
        },
        height: 420,
        margin: { l: 60, r: 20, t: 40, b: 70 },
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: "100%" }}
    />
  );
}
