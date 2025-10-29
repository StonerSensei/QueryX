# config.py

# Database configuration
DB_URL = "postgresql://postgres:mypassword123@localhost:5432/ragDB"

# Qdrant configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "schema_descriptions"

# Embedding model configuration
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama configuration (add these lines)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # or whatever model you're using

# Execution settings
EXECUTE_MAX_ROWS = 100
DEFAULT_TOP_K = 3

# Logging level
LOG_LEVEL = "INFO"
