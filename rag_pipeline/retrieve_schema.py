"""
retrieve_schema.py
------------------------------------
Retrieves the most relevant schema entries from Qdrant
for a given natural language query.

Used by:
 - generate_sql_from_query.py (for schema-aware SQL generation)
 - llm_reasoner.py (future integration with fine-tuned models)

Returns top-K schema snippets with their similarity scores.
------------------------------------
"""

import logging
from typing import List, Dict, Any
from config import (
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    EMBED_MODEL,
    DEFAULT_TOP_K,
)
from clients import get_qdrant_client, get_embedding_model

logging.basicConfig(
    filename="rag_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def retrieve_schema(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant schema information from Qdrant
    based on a natural language query.

    Args:
        query (str): User's English question.
        top_k (int): Number of top schema matches to return.

    Returns:
        List[Dict]: A list of schema entries, each with
                    { "table": str, "description": str, "score": float }
    """
    if not query:
        logging.warning("Empty query provided to retrieve_schema().")
        return []

    logging.info(f"Retrieving schema context for query: {query}")

    try:
        # Cached clients
        model = get_embedding_model()
        qdrant = get_qdrant_client()

        # Generate embedding for user query
        q_vec = (
            model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
        )
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return []

    try:
        # Qdrant similarity search
        results = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_vec,
            limit=top_k,
        )
    except Exception as e:
        logging.error(f"Qdrant search failed: {e}")
        return []

    schema_contexts = []
    for r in results:
        # Qdrant API returns either object or dict
        payload = getattr(r, "payload", None) or r.get("payload", {})
        score = getattr(r, "score", None) or r.get("score", 0.0)
        desc = payload.get("description", "").strip()
        table = payload.get("table", "unknown_table")

        schema_contexts.append(
            {
                "table": table,
                "description": desc,
                "score": float(score),
            }
        )

    # Sort descending by similarity score
    schema_contexts.sort(key=lambda x: x["score"], reverse=True)

    logging.info(f"Retrieved {len(schema_contexts)} relevant schema entries.")
    return schema_contexts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve schema info from Qdrant.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of top schema entries to fetch.")
    args = parser.parse_args()

    query = input("Enter your natural language question: ").strip()
    results = retrieve_schema(query, top_k=args.top_k)

    if not results:
        print("No schema results found.")
    else:
        print("\n--- Retrieved Schema Context ---")
        for i, r in enumerate(results, 1):
            print(f"[{i}] (score={r['score']:.4f}) table={r['table']}")
            print(f"{r['description']}\n")
