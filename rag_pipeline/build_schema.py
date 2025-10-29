# build_schema.py
from sqlalchemy import inspect
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
import numpy as np
import logging
import uuid
from config import DB_URL, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBED_MODEL
from clients import get_qdrant_client, get_embedding_model, get_db_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def table_to_description(table_name, inspector):
    cols = inspector.get_columns(table_name)
    pks = inspector.get_pk_constraint(table_name).get("constrained_columns", [])
    fks = inspector.get_foreign_keys(table_name)

    parts = [f"Table {table_name} contains columns:"]
    for col in cols:
        name, type_ = col["name"], str(col["type"])
        tags = []
        if name in pks:
            tags.append("Primary Key")
        for fk in fks:
            if name in fk["constrained_columns"]:
                tags.append(f"Foreign Key to {fk['referred_table']}.{fk['referred_columns'][0]}")
        tag_str = f" ({', '.join(tags)})" if tags else ""
        parts.append(f"- {name}: {type_}{tag_str}")
    return "\n".join(parts)


def build_schema():
    engine = get_db_engine()
    inspector = inspect(engine)
    qdrant = get_qdrant_client()
    model = get_embedding_model()

    table_names = inspector.get_table_names()
    if not table_names:
        logging.warning("No tables found in database.")
        return

    schema_texts, payloads = [], []
    for table in table_names:
        desc = table_to_description(table, inspector)
        schema_texts.append(desc)
        payloads.append({"table": table, "description": desc})

    embeddings = model.encode(schema_texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]

    if not qdrant.collection_exists(QDRANT_COLLECTION):
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload=payloads[i],
        )
        for i in range(len(schema_texts))
    ]

    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    logging.info(f"Uploaded {len(points)} schema entries to Qdrant collection '{QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    build_schema()
