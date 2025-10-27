# config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Database configuration
DB_URL = os.getenv("DB_URL", "postgresql://postgres:mypassword123@localhost:5432/ragDB")

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "schema_descriptions")

# Embedding model configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
