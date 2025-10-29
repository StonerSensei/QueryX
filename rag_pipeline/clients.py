# clients.py
from sqlalchemy import create_engine
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import DB_URL, QDRANT_HOST, QDRANT_PORT, EMBED_MODEL
import logging

_engine = None
_qdrant_client = None
_model = None


def get_db_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL)
        logging.info("Database engine initialized.")
    return _engine


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logging.info("Qdrant client initialized.")
    return _qdrant_client


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
        logging.info(f"Embedding model '{EMBED_MODEL}' loaded.")
    return _model
