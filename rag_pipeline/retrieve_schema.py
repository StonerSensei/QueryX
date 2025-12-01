"""
retrieve_schema.py
------------------------------------
Retrieves the most relevant schema entries from Qdrant
for a given natural language query.

Used by:
 - generate_sql_from_query.py
 - llm_reasoner.py

Returns top-K schema snippets with their similarity scores.
------------------------------------
"""

from typing import List, Dict, Any

import numpy as np

from config import (
    QDRANT_COLLECTION,
    DEFAULT_TOP_K,
)
from clients import get_qdrant_client, get_embedding_model
from utils.llm_logger import logger


def _search_qdrant(
    query_vector: np.ndarray,
    top_k: int,
) -> List[Any]:
    """
    Internal helper that calls the appropriate Qdrant search API depending
    on which methods exist in the installed qdrant-client version.

    Returns a list of ScoredPoint-like objects.
    """
    qdrant = get_qdrant_client()

    # 1) New universal API (preferred in recent qdrant-client versions)
    if hasattr(qdrant, "query_points"):
        logger.info(
            "Using Qdrant.query_points for schema retrieval (top_k=%d).",
            top_k,
        )
        resp = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        # resp is a QueryResponse; actual hits are in resp.points
        return list(resp.points or [])

    if hasattr(qdrant, "search"):
        logger.info(
            "Using Qdrant.search for schema retrieval (top_k=%d).",
            top_k,
        )
        hits = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return list(hits or [])

    logger.error(
        "Qdrant client has neither 'query_points' nor 'search' methods."
    )
    return []


def retrieve_schema(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Given a natural language question, returns a list of schema contexts:
    [
      {
        "table": "mytable",
        "description": "Table mytable contains columns: ...",
        "score": 0.9321
      },
      ...
    ]
    """
    logger.info("Retrieving schema context for query: %s", question)

    model = get_embedding_model()
    query_vec = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,  # must match build_schema.py
    )[0]
    logger.info("Query embedding generated (dim=%d).", query_vec.shape[0])

    try:
        hits = _search_qdrant(query_vec, top_k)
    except Exception as e:
        logger.error("Qdrant search failed: %s", e)
        return []

    if not hits:
        logger.warning("No hits returned from Qdrant for query: %s", question)
        return []

    schema_contexts: List[Dict[str, Any]] = []
    for h in hits:
        payload = getattr(h, "payload", {}) or {}
        score = float(getattr(h, "score", 0.0))
        table = payload.get("table") or payload.get("name") or "unknown_table"
        desc = payload.get("description", "")

        schema_contexts.append(
            {
                "table": table,
                "description": desc,
                "score": score,
            }
        )

    schema_contexts.sort(key=lambda x: x["score"], reverse=True)

    logger.info(
        "Retrieved %d schema entries. Top table: %s (score=%.4f)",
        len(schema_contexts),
        schema_contexts[0]["table"],
        schema_contexts[0]["score"],
    )

    return schema_contexts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Retrieve schema info from Qdrant."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top schema entries to fetch.",
    )
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
