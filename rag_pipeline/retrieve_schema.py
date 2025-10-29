# retrieve_schema.py
import logging
from typing import List, Dict
from config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT, EMBED_MODEL
from clients import get_qdrant_client, get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def retrieve_schema(query: str, top_k: int = 3) -> List[Dict]:
    """
    Return top-k schema entries from Qdrant for the provided natural-language query.
    Each returned dict contains: { "table": str, "description": str, "score": float }
    """
    if not query:
        return []

    model = get_embedding_model()
    qdrant = get_qdrant_client()

    try:
        q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        raise

    # Qdrant search API may vary by client version; use `search` with expected args.
    try:
        results = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_vec,
            limit=top_k
        )
    except TypeError:
        # Fallback if qdrant client uses different signature
        results = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=q_vec, limit=top_k)

    out = []
    for r in results:
        payload = getattr(r, "payload", None) or r.get("payload", {})
        score = getattr(r, "score", None) or r.get("score", 0.0)
        description = payload.get("description", "")
        table = payload.get("table", None)
        out.append({"table": table, "description": description, "score": float(score)})
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()
    q = input("Enter English question: ").strip()
    res = retrieve_schema(q, top_k=args.top_k)
    print("\n--- Retrieval Results ---")
    for i, r in enumerate(res, 1):
        print(f"[{i}] score={r['score']:.4f} table={r['table']}\n{r['description']}\n")
