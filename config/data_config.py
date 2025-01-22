from pathlib import Path
from typing import Dict, List

# Base data directory
DATA_DIR = Path("datasets")

# Dataset configurations
DATASET_CONFIG = {
    "train": {
        "paths": [
            DATA_DIR / "train-*-of-*.parquet",  # Handles sharded files
            DATA_DIR / "train" / "*.jsonl",      # Also supports directory structure
            DATA_DIR / "train" / "*.csv",
            DATA_DIR / "train" / "*.parquet"
        ],
        "text_field": "text",
        "max_length": 512
    },
    "validation": {
        "paths": [
            DATA_DIR / "validation-*-of-*.parquet",
            DATA_DIR / "validation" / "*.jsonl",
            DATA_DIR / "validation" / "*.csv",
            DATA_DIR / "validation" / "*.parquet"
        ],
        "text_field": "text",
        "max_length": 512
    },
    "test": {
        "paths": [
            DATA_DIR / "test-*-of-*.parquet",
            DATA_DIR / "test" / "*.jsonl",
            DATA_DIR / "test" / "*.csv",
            DATA_DIR / "test" / "*.parquet"
        ],
        "text_field": "text",
        "max_length": 512
    }
} 