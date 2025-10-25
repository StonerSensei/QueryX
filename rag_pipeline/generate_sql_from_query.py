import os
import requests
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Configuration
QDRANT_COLLECTION = "schema_descriptions"
OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_sql_with_ollama(question, context_texts):
    """
    Generate SQL using local Ollama model (llama3)
    No API tokens needed, runs locally
    """
    schema_context = "\n".join(context_texts)
    
    prompt = f"""You are a PostgreSQL expert. Generate ONLY the SQL query for the following question. Do not include explanations or markdown code blocks.

Database Schema:
{schema_context}

Question: {question}

Return ONLY the SQL query:"""
    
    try:
        print("Generating SQL with Ollama (llama3)...")
        
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            sql = result.get("response", "").strip()
            
            # Clean up the output
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            # Remove common prefixes that models add
            if sql.startswith("SQL query:"):
                sql = sql.replace("SQL query:", "").strip()
            if sql.startswith("Here's the SQL"):
                lines = sql.split("\n")
                sql = "\n".join(lines[1:]).strip()
            
            if sql:
                print("Successfully generated SQL with Ollama")
                return sql
            else:
                print("Model returned empty response")
                return None
        else:
            print(f"Ollama error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to Ollama at localhost:11434")
        print("Make sure Ollama is running: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print("Ollama request timed out (model is slow)")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("SQL Query Generator with Ollama\n")
    
    # Check Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("Ollama is not responding properly")
            print("Start Ollama with: ollama serve")
            return
    except requests.exceptions.ConnectionError:
        print("Cannot connect to Ollama at localhost:11434")
        print("Start Ollama with: ollama serve")
        return
    
    # Initialize clients
    try:
        print("Connecting to Qdrant...")
        qdrant = QdrantClient("localhost", port=6333)
        
        print("Loading embeddings model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    # Get user question
    question = input("\nEnter your question: ").strip()
    if not question:
        print("No question provided.")
        return
    
    print("\nSearching database schema...")
    
    try:
        # Encode question and search
        query_vector = model.encode(question).tolist()
        
        results = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=3
        ).points
        
        if not results:
            print("No schema descriptions found in Qdrant")
            return
        
        # Extract schema descriptions
        context_texts = [hit.payload["description"] for hit in results]
        
        print("\nRelevant Tables Found:")
        for i, t in enumerate(context_texts, 1):
            print(f"  {i}. {t[:100]}...")
        
        print("\nGenerating SQL with AI...")
        sql_output = generate_sql_with_ollama(question, context_texts)
        
        if sql_output:
            print("\nGenerated SQL Query:")
            print("─" * 70)
            print(sql_output)
            print("─" * 70)
            
            # Ask to execute
            execute = input("\nExecute this query? (yes/no): ").strip().lower()
            if execute in ['yes', 'y']:
                try:
                    from sqlalchemy import create_engine, text
                    
                    engine = create_engine("postgresql://postgres:mypassword123@localhost:5432/ragDB")
                    with engine.connect() as conn:
                        result = conn.execute(text(sql_output))
                        rows = result.fetchall()
                        
                        print("\nQuery Results:")
                        print("─" * 70)
                        if rows:
                            for row in rows:
                                print(row)
                        else:
                            print("(No rows returned)")
                        print("─" * 70)
                except Exception as e:
                    print(f"\nError executing query: {e}")
        else:
            print("\nFailed to generate SQL query")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

