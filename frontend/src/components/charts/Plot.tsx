import { useRef, useEffect, useCallback } from "react";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout, Config } from "plotly.js";

interface PlotProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  style?: React.CSSProperties;
}

export default function Plot({ data, layout = {}, config = {}, style }: PlotProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const initializedRef = useRef(false);

  const resizeHandler = useCallback(() => {
    if (containerRef.current) {
      Plotly.Plots.resize(containerRef.current);
    }
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    if (!initializedRef.current) {
      Plotly.newPlot(el, data, layout, config);
      initializedRef.current = true;
    } else {
      Plotly.react(el, data, layout, config);
    }
  }, [data, layout, config]);

  useEffect(() => {
    const el = containerRef.current;
    window.addEventListener("resize", resizeHandler);
    return () => {
      window.removeEventListener("resize", resizeHandler);
      if (el) {
        Plotly.purge(el);
      }
      initializedRef.current = false;
    };
  }, [resizeHandler]);

  return <div ref={containerRef} style={style} />;
}
