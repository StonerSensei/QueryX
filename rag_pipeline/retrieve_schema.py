# retrieve_only.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configuration
QDRANT_COLLECTION = "schema_descriptions"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
   
    model = SentenceTransformer(EMBED_MODEL_NAME)

   
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

   
    query = input("Enter English question: ").strip()
    if not query:
        print("No query provided.")
        return

   
    query_vector = model.encode([query])[0].tolist()

   
    results = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=3  # Top 3 similar descriptions
    )

    print("\n--- Retrieval Results ---")
    for i, r in enumerate(results):
        payload = r.payload
        score = r.score
        desc = payload.get("description", "(no description found)")
        print(f"[{i+1}] Score: {score:.4f}")
        print(f"Description: {desc}\n")

if __name__ == "__main__":
    main()
