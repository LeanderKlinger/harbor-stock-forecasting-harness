import type { ConfigSummary } from "@/lib/types";
import { formatFeatureSet } from "@/lib/experimentAggregation";

interface Props {
  config: ConfigSummary;
}

type BadgeVariant =
  | "default"
  | "accent"
  | "blue"
  | "green"
  | "purple"
  | "rose"
  | "cyan"
  | "teal";

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  default: "bg-zinc-50 text-zinc-400",
  accent: "bg-amber-50 text-amber-500",
  blue: "bg-blue-50 text-blue-400",
  green: "bg-emerald-50 text-emerald-400",
  purple: "bg-violet-50 text-violet-400",
  rose: "bg-rose-50 text-rose-400",
  cyan: "bg-cyan-50 text-cyan-400",
  teal: "bg-teal-50 text-teal-400",
};

function Badge({
  label,
  variant = "default",
  hidden = false,
}: {
  label: string;
  variant?: BadgeVariant;
  hidden?: boolean;
}) {
  return (
    <span
      className={`inline-flex items-center whitespace-nowrap px-1.5 rounded-full text-[0.625rem] font-medium leading-4 ${
        hidden ? "invisible" : VARIANT_CLASSES[variant]
      }`}
    >
      {label}
    </span>
  );
}

// Default config values for hiding "default" badges
const DEFAULT_TOP_K = 4;
const DEFAULT_SEQ_LEN = 60;
// Primary slots: fixed-width CSS grid columns so badges align across rows
const PRIMARY_SLOTS = [
  "mode",
  "rebal",
  "retrain",
  "seq_len",
  "feature_set",
  "cost",
  "ema",
  "early_stop",
] as const;

// Placeholder text per slot (reserves column width when badge is hidden)
const PRIMARY_PLACEHOLDERS: Record<string, string> = {
  mode: "headline",
  rebal: "rebal=20d",
  retrain: "retrain=auto",
  seq_len: "seq=250",
  feature_set: "close-only",
  cost: "cost=0.1%",
  ema: "ema=5+10+20",
  early_stop: "early_stop=10",
};

interface SlotBadge {
  label: string;
  variant: BadgeVariant;
}

function buildBadgeSlots(cfg: ConfigSummary): Record<string, SlotBadge> {
  const slots: Record<string, SlotBadge> = {};

  if (cfg.headline_only === true) {
    slots.mode = { label: "headline", variant: "accent" };
  } else if (cfg.headline_only === false) {
    slots.mode = { label: "article", variant: "accent" };
  }

  slots.rebal = {
    label: `rebal=${cfg.rebalance_horizon_days}d`,
    variant: "blue",
  };

  if (cfg.retraining_cadence_days != null) {
    const v = cfg.retraining_cadence_days;
    slots.retrain = {
      label: `retrain=${v === "auto" ? "auto" : `${v}d`}`,
      variant: "blue",
    };
  }

  if (cfg.sequence_length !== undefined && cfg.sequence_length !== DEFAULT_SEQ_LEN) {
    slots.seq_len = {
      label: `seq=${cfg.sequence_length}`,
      variant: "cyan",
    };
  }

  // Only non-full is badge-worthy; full/absent is the canonical default.
  if (cfg.feature_set !== undefined && cfg.feature_set !== "full") {
    slots.feature_set = {
      label: formatFeatureSet(cfg.feature_set),
      variant: "purple",
    };
  }

  if (cfg.transaction_cost_pct > 0) {
    slots.cost = {
      label: `cost=${(cfg.transaction_cost_pct * 100).toFixed(1)}%`,
      variant: "green",
    };
  }

  if (cfg.sentiment_ema_spans?.length) {
    slots.ema = {
      label: `ema=${cfg.sentiment_ema_spans.join("+")}`,
      variant: "purple",
    };
  }

  if (cfg.top_k !== DEFAULT_TOP_K) {
    slots.top_k = { label: `top_k=${cfg.top_k}`, variant: "cyan" };
  }

  if (cfg.min_predicted_return > 0) {
    slots.threshold = {
      label: `threshold=${(cfg.min_predicted_return * 100).toFixed(1)}%`,
      variant: "rose",
    };
  }

  if (cfg.conviction_mode === "volatility_scaled") {
    slots.conviction = { label: "vol-scaled", variant: "rose" };
  }

  if (cfg.early_stopping_patience != null) {
    slots.early_stop = {
      label: `early_stop=${cfg.early_stopping_patience}`,
      variant: "teal",
    };
  }

  if (cfg.market_condition !== "idyllic") {
    slots.market = { label: cfg.market_condition, variant: "accent" };
  }

  return slots;
}

export function ConfigBadges({ config }: Props) {
  const slots = buildBadgeSlots(config);

  // Secondary slots: rendered as a flex row after the primary grid
  const secondaryKeys = ["market", "top_k", "threshold", "conviction"];
  const secondaryBadges = secondaryKeys.filter((k) => k in slots);

  return (
    <div className="mt-1">
      {/* Primary: fixed grid so columns align across rows */}
      <div
        className="grid gap-1 items-center"
        style={{
          gridTemplateColumns: `repeat(${PRIMARY_SLOTS.length}, max-content)`,
        }}
      >
        {PRIMARY_SLOTS.map((slot) =>
          slots[slot] ? (
            <Badge
              key={slot}
              label={slots[slot].label}
              variant={slots[slot].variant}
            />
          ) : (
            <Badge
              key={slot}
              label={PRIMARY_PLACEHOLDERS[slot]}
              hidden
            />
          )
        )}
      </div>
      {/* Secondary: flex wrap */}
      {secondaryBadges.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-0.5">
          {secondaryBadges.map((key) => (
            <Badge
              key={key}
              label={slots[key].label}
              variant={slots[key].variant}
            />
          ))}
        </div>
      )}
    </div>
  );
}
