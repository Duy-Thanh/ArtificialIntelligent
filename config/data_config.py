from pathlib import Path
from typing import Dict, List

# Base data directory - update for Colab compatibility
DATA_DIR = Path("Datasets")  # Changed from "datasets" to "Datasets"

# Dataset configurations
DATASET_CONFIG = {
    "train": {
        "paths": [
            DATA_DIR / "train-*-of-*.parquet",
        ],
        "text_field": "text",
        "max_length": 512
    },
    "validation": {
        "paths": [
            DATA_DIR / "validation-*-of-*.parquet",
        ],
        "text_field": "text",
        "max_length": 512
    },
    "test": {
        "paths": [
            DATA_DIR / "test-*-of-*.parquet",
        ],
        "text_field": "text",
        "max_length": 512
    }
} 