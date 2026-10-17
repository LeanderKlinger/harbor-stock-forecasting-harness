import dataclasses
import hashlib
import json
from pathlib import Path

from src.paths import TRAINING_CACHE_DIR


@dataclasses.dataclass(frozen=True)
class TrainingCacheKey:
    """Content-addressed key for a per-stock fitted model.

    All fields together MUST fully determine the trained artifact. Forgetting a
    training-affecting field is a silent correctness bug for numerical-only
    changes (loud RuntimeError on shape changes).
    """

    training_signature: str
    stock_id: str
    seed: int | None
    rebalance_horizon_days: int
    training_start: str
    training_end: str
    headline_only: bool
    semantic_model: str
    sentiment_ema_spans: tuple[int, ...]
    retraining_cadence_days: (
        int | str | None
    )  # int | "auto" | None — "auto" hashes to its own slot
    early_stopping_patience: (
        int | None
    )  # changes val split + epoch count → different weights

    def to_hash(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_path_for(key: TrainingCacheKey) -> Path:
    TRAINING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TRAINING_CACHE_DIR / f"{key.to_hash()}.bin"
