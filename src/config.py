from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROMPT_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
INDEX_PATH = RESULTS_DIR / "chroma_retriever.json"
CHROMA_DIR = RESULTS_DIR / "chroma_db"
OLLAMA_MODEL = "llama3.2:3b"
SEMANTIC_SIMILARITY_THRESHOLD = 0.38

PLAY_FILES = {
    "hamlet": RAW_DATA_DIR / "hamlet.json",
    "macbeth": RAW_DATA_DIR / "macbeth.json",
    "romeo_and_juliet": RAW_DATA_DIR / "romeo_and_juliet.json",
}

DEFAULT_TOP_K = 1
CHUNK_WORDS = 260
CHUNK_OVERLAP = 45
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
