import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Core configuration with sensible defaults
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME: str = os.getenv("MODEL_NAME", "deepseek-r1")
REQUEST_TIMEOUT: float = float(os.getenv("REQUEST_TIMEOUT", "60"))  # seconds

# Model option defaults (Ollama options)
DEFAULT_TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
DEFAULT_TOP_P: float = float(os.getenv("TOP_P", "0.9"))
DEFAULT_NUM_PREDICT: int = int(os.getenv("NUM_PREDICT", "512"))
