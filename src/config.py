from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2