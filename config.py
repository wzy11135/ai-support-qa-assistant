from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHUNK_SIZE = 420
CHUNK_OVERLAP = 80
TOP_K = 4
MIN_RETRIEVAL_SCORE = 0.08

