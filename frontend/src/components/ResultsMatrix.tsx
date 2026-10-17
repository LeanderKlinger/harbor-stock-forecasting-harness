import { fmtRatio } from "@/lib/constants";
import {
  axisDisplayName,
  buildGroupLookup,
  lookupKey,
  orderedModelsForTable,
  type ResultsTable,
} from "@/lib/experimentAggregation";

interface Props {
  table: ResultsTable;
}

// Matrix cells use 1-decimal % to match the reference HTML and stay compact.
function fmtPct1(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function CellMetric({
  mean,
  std,
  className,
}: {
  mean: number;
  std: number;
  className?: string;
}) {
  return (
    <div className={`whitespace-nowrap ${className ?? ""}`}>
      {fmtPct1(mean)}
      {std > 0 && (
        <span className="text-zinc-400 text-xs"> ±{fmtPct1(std)}</span>
      )}
    </div>
  );
}

export function ResultsMatrix({ table }: Props) {
  const { title, innerAxes, columnValues, innerKeys, groups } = table;
  const orderedModels = orderedModelsForTable(groups);
  const lookup = buildGroupLookup(groups, innerAxes);

  // Best mean-Sortino per column for emerald highlight (only when ≥2 models in column)
  const bestSortinoPerCol: Record<string, number> = {};
  for (const innerKey of innerKeys) {
    const sortinosInCol: number[] = [];
    for (const mt of orderedModels) {
      const g = lookup.get(lookupKey(mt, innerAxes, innerKey));
      if (g) sortinosInCol.push(g.meanMetrics.sortino_ratio ?? -Infinity);
    }
    if (sortinosInCol.length > 0) {
      bestSortinoPerCol[innerKey] = Math.max(...sortinosInCol);
    }
  }

  const axisHeader = innerAxes.length
    ? innerAxes.map((a) => axisDisplayName(a)).join(" / ")
    : null;

  return (
    <div className="my-8">
      {title && (
        <h3 className="text-lg font-semibold text-zinc-800 mt-8 mb-4">{title}</h3>
      )}
      <div className="rounded-2xl border border-zinc-200 bg-white shadow-sm p-3">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              {axisHeader && (
                <tr>
                  <th className="px-4 pt-2 pb-2 border-b border-zinc-100" />
                  <th
                    colSpan={columnValues.length}
                    className="text-center text-xs text-zinc-500 font-medium pt-2 pb-2 px-4 border-b border-zinc-100"
                  >
                    {axisHeader}
                  </th>
                </tr>
              )}
              <tr className="border-b border-zinc-200">
                <th className="text-left text-xs font-semibold text-zinc-600 uppercase tracking-wider px-4 py-3 align-bottom">
                  Model
                </th>
                {columnValues.map((values, ci) => (
                  <th
                    key={innerKeys[ci]}
                    className="text-center text-xs font-medium text-zinc-700 px-4 py-3 align-bottom"
                  >
                    {values.map((v, vi) => (
                      <div key={vi} className="whitespace-nowrap">
                        {v}
                        {vi < values.length - 1 ? "," : ""}
                      </div>
                    ))}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {orderedModels.map((mt, rowIdx) => (
                <tr
                  key={mt}
                  className={
                    rowIdx < orderedModels.length - 1
                      ? "border-b border-zinc-100"
                      : ""
                  }
                >
                  <td className="font-medium text-sm whitespace-nowrap px-4 py-3 text-zinc-800">
                    {mt}
                  </td>
                  {innerKeys.map((innerKey) => {
                    const g = lookup.get(lookupKey(mt, innerAxes, innerKey));
                    if (!g) {
                      return (
                        <td
                          key={innerKey}
                          className="text-center text-sm text-zinc-300 px-4 py-3"
                        >
                          —
                        </td>
                      );
                    }
                    const sortinoMean = g.meanMetrics.sortino_ratio ?? 0;
                    const isBest =
                      orderedModels.length > 1 &&
                      sortinoMean === bestSortinoPerCol[innerKey];
                    const cellBg = isBest ? "bg-emerald-50" : "";
                    const sortinoCls = isBest
                      ? "text-emerald-700 font-semibold"
                      : "font-semibold text-zinc-900";
                    return (
                      <td
                        key={innerKey}
                        className={`text-center text-sm leading-tight px-4 py-3 ${cellBg}`}
                      >
                        <div className={`whitespace-nowrap ${sortinoCls}`}>
                          {fmtRatio(sortinoMean)}
                          {(g.stdMetrics.sortino_ratio ?? 0) > 0 && (
                            <span className="text-zinc-500 text-xs font-normal">
                              {" "}
                              ±{fmtRatio(g.stdMetrics.sortino_ratio)}
                            </span>
                          )}
                          {g.nSeeds > 1 && (
                            <span className="text-zinc-400 text-xs font-normal">
                              {" "}
                              n={g.nSeeds}
                            </span>
                          )}
                        </div>
                        <CellMetric
                          mean={g.meanMetrics.annualized_return ?? 0}
                          std={g.stdMetrics.annualized_return ?? 0}
                          className="text-zinc-600"
                        />
                        <CellMetric
                          mean={g.meanMetrics.maximum_drawdown ?? 0}
                          std={g.stdMetrics.maximum_drawdown ?? 0}
                          className="text-zinc-600"
                        />
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="text-xs text-zinc-400 mt-2 ml-1">
        Cells: Sortino / Ann. Return / Max Drawdown (mean ± std across seeds)
      </div>
    </div>
  );
}
