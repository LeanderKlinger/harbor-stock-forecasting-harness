import type { Layout } from "plotly.js";

const GRID_STYLE = {
  gridcolor: "rgba(0,0,0,0.06)",
  linecolor: "rgba(0,0,0,0.12)",
  zerolinecolor: "rgba(0,0,0,0.12)",
};

// Shared Plotly layout matching the Python _apply_theme()
// x-axis stays auto (date), y-axis forced to linear (prevents Plotly misdetecting numbers as timestamps)
export const PLOTLY_LAYOUT: Partial<Layout> = {
  template: "plotly_white" as unknown as Layout["template"],
  paper_bgcolor: "white",
  plot_bgcolor: "white",
  font: { family: "Inter, system-ui, sans-serif", color: "#1a1a1a", size: 12 },
  margin: { l: 60, r: 20, t: 60, b: 50 },
  xaxis: GRID_STYLE,
  yaxis: { ...GRID_STYLE, type: "linear" },
};
