import { useParams, Link } from "react-router";
import { useRunDetail } from "@/lib/hooks";
import { MetricsCards } from "@/components/MetricsCards";
import { ConfigTable } from "@/components/ConfigTable";
import { PortfolioChart } from "@/components/charts/PortfolioChart";
import { StockTradeChart } from "@/components/charts/StockTradeChart";
import { ThresholdChart } from "@/components/charts/ThresholdChart";

export function RunDetail() {
  const { modelType, runDir } = useParams();
  const fullDir = `${modelType}/${runDir}`;
  const { data: detail, isLoading, error } = useRunDetail(fullDir);

  if (isLoading) {
    return (
      <div className="min-h-screen p-6 md:p-10 max-w-7xl mx-auto">
        <p className="text-zinc-500">Loading run detail...</p>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="min-h-screen p-6 md:p-10 max-w-7xl mx-auto">
        <p className="text-red-500">
          Error: {error?.message ?? "Run not found"}
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 md:p-10 max-w-7xl mx-auto">
      <div className="mb-2">
        <Link to="/" className="text-blue-500 hover:underline text-sm">
          ← Leaderboard
        </Link>
      </div>

      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">
          {detail.model_name}
        </h1>
        <p className="text-zinc-500 mt-1">
          Run{" "}
          <code className="text-xs bg-zinc-100 px-1.5 py-0.5 rounded">
            {detail.run_id.slice(0, 8)}
          </code>{" "}
          — {detail.run_timestamp.slice(0, 16).replace("T", " ")} —{" "}
          {detail.config.test_start} to {detail.config.test_end} —{" "}
          {detail.config.stock_ids.join(", ")}
        </p>
      </div>

      <MetricsCards metrics={detail.metrics} />
      <ConfigTable config={detail.config} />

      <div className="my-8">
        <h2 className="text-xl font-semibold mb-4">Portfolio Value</h2>
        <div className="border border-zinc-200 rounded-xl p-6">
          <PortfolioChart
            equityCurve={detail.equity_curve}
            testStart={detail.config.test_start}
            testEnd={detail.config.test_end}
          />
        </div>
      </div>

      {detail.threshold_data && (
        <div className="my-8">
          <h2 className="text-xl font-semibold mb-4">
            Vol-Scaled Conviction Threshold
          </h2>
          <div className="border border-zinc-200 rounded-xl p-6">
            <ThresholdChart
              thresholdData={detail.threshold_data}
              baseThreshold={detail.config.min_predicted_return}
              testStart={detail.config.test_start}
              testEnd={detail.config.test_end}
            />
          </div>
        </div>
      )}

      <div className="my-8">
        <h2 className="text-xl font-semibold mb-4">Per-Stock Trades</h2>
        <div className="grid grid-cols-1 gap-6">
          {detail.config.stock_ids.map((stockId) => (
            <div
              key={stockId}
              className="border border-zinc-200 rounded-xl p-6"
            >
              <StockTradeChart
                stockId={stockId}
                chartData={detail.stock_charts[stockId]}
                testStart={detail.config.test_start}
                testEnd={detail.config.test_end}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
