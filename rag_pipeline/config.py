# config.py

# Database configuration
DB_URL = "postgresql://postgres:mypassword123@localhost:5432/ragDB"

# Qdrant configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "schema_descriptions"

# Embedding model configuration
# During initial development/train:
#   EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# After running train_schema_embeddings.py, you can switch to:
#   EMBED_MODEL = "models/queryx-embeddings"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Path to fine-tuned FLAN-T5 SQL model
# This should match the OUTPUT_DIR in train_sql_model.py
MODEL_PATH = "models/flan_t5_sql"

# Whether to fall back to Ollama if local T5 generation fails
USE_OLLAMA_FALLBACK = False

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # or whatever model you're using

# Execution settings
EXECUTE_MAX_ROWS = 100
DEFAULT_TOP_K = 3

# Logging level
LOG_LEVEL = "INFO"
