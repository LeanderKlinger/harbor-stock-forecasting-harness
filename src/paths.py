from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent  # Goes up to -> src -> root

DATA_DIR = PROJECT_ROOT / "data"
SYSTEM_DATA_DIR = DATA_DIR / "system"
ARTICLES_DIR = DATA_DIR / "articles"
STOCK_DATA_DIR = DATA_DIR / "stock_data"
VALIDATION_DIR = DATA_DIR / "validation"
EXECUTION_OUTPUT_ROOT = SYSTEM_DATA_DIR / "execution_output"
INPUT_CACHE_DIR = EXECUTION_OUTPUT_ROOT / ".input_cache"
TRAINING_CACHE_DIR = DATA_DIR / "training_cache"
RUNS_INDEX_PATH = EXECUTION_OUTPUT_ROOT / "runs_index.json"
PARSING_CHECKPOINTS_DIR = SYSTEM_DATA_DIR / "parsing_checkpoints"
