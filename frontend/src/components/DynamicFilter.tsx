interface Option<T extends string | number> {
  label: string;
  value: T;
}

interface Props<T extends string | number> {
  label: string;
  values: Option<T>[];
  active: T | null;
  onChange: (value: T | null) => void;
  /** Render as a <select> dropdown above this many options; pills otherwise. */
  pillThreshold?: number;
}

/**
 * Filter primitive that adapts to cardinality:
 * - ≤ pillThreshold (default 5): pill row, toggle-on-second-click.
 * - > pillThreshold: native <select> with the same styling as the experiment
 *   selector.
 *
 * Parent is responsible for skipping the render when fewer than 2 distinct
 * values exist — this component trusts what it's given.
 */
export function DynamicFilter<T extends string | number>({
  label,
  values,
  active,
  onChange,
  pillThreshold = 5,
}: Props<T>) {
  if (values.length <= pillThreshold) {
    return (
      <div className="flex items-center gap-1">
        <span className="text-[0.625rem] text-zinc-400 mr-1">{label}:</span>
        {values.map((opt) => (
          <button
            key={String(opt.value)}
            onClick={() => onChange(active === opt.value ? null : opt.value)}
            className={`inline-flex items-center rounded-full px-1.5 py-px text-[0.625rem] font-medium leading-4 cursor-pointer hover:opacity-80 transition-colors ${
              active === opt.value
                ? "bg-zinc-700 text-white"
                : "bg-zinc-100 text-zinc-600"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1">
      <span className="text-[0.625rem] text-zinc-400 mr-1">{label}:</span>
      <select
        value={active === null ? "" : String(active)}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") {
            onChange(null);
            return;
          }
          // Reverse-lookup to preserve the original value type.
          const match = values.find((v) => String(v.value) === raw);
          if (match) onChange(match.value);
        }}
        className="text-xs bg-zinc-100 border border-zinc-300 rounded px-1.5 py-0.5 text-zinc-600"
      >
        <option value="">all</option>
        {values.map((opt) => (
          <option key={String(opt.value)} value={String(opt.value)}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
