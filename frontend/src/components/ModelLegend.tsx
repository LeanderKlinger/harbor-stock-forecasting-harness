import { MODEL_DOT_COLORS } from "@/lib/constants";

interface Props {
  modelTypes: Set<string>;
}

export function ModelLegend({ modelTypes }: Props) {
  return (
    <div className="flex flex-wrap gap-3 mb-4">
      {[...modelTypes].sort().map((mt) => (
        <span
          key={mt}
          className="inline-flex items-center gap-1 text-xs text-zinc-600"
        >
          <span
            className={`inline-block w-3 h-3 rounded-full ${MODEL_DOT_COLORS[mt] ?? "bg-zinc-400"}`}
          />
          {mt}
        </span>
      ))}
    </div>
  );
}
