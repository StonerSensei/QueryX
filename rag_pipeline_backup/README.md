# RAG Pipeline (rag_pipeline)

This folder contains the Retrieval-Augmented Generation (RAG) helpers used by the QueryX project. The scripts extract a PostgreSQL database schema, embed schema descriptions into Qdrant, retrieve relevant schema context for a natural-language question, and generate SQL using a local LLM (Ollama).

The pipeline enables users to convert **natural language questions** into **SQL queries** by integrating:

- A relational database schema (PostgreSQL)
- A vector database (Qdrant) for schema retrieval
- A sentence embedding model
- A local LLM (Ollama) for SQL generation

This RAG pipeline automates the full flow from database schema extraction to SQL generation and optional query execution.

---

## Folder Structure

```
rag_pipeline/
│
├── config.py                    # Centralized configuration and environment variables
├── clients.py                   # Shared clients: DB engine, Qdrant client, embedding model
├── build_schema.py              # Extracts schema, encodes it, and uploads to Qdrant
├── retrieve_schema.py           # Retrieves relevant schema info for a user question
└── generate_sql_from_query.py  # Generates SQL from English query, optionally executes it
```

---

## Tech Stack

| Component | Technology / Library | Purpose |
|-----------|---------------------|---------|
| **Database** | PostgreSQL | Stores relational data |
| **Vector Database** | Qdrant | Stores and retrieves schema embeddings |
| **Backend Framework** | SQLAlchemy | Database introspection and connection |
| **Embedding Model** | Sentence-Transformers (MiniLM-L6-v2) | Converts text into embeddings |
| **Language Model** | Ollama (Llama3 or compatible) | Generates SQL queries locally |
| **Python Libraries** | `requests`, `sqlparse`, `numpy`, `python-dotenv` | API calls, SQL validation, and configuration management |

---

## Prerequisites

Before running any scripts, ensure the following services are active:

1. **PostgreSQL**  
   - Running locally on port `5432`  
   - Accessible via a valid database URL (defined in `.env` or `config.py`)  
   - Contains at least one schema with tables.

2. **Qdrant Vector Database**  
   - Running locally on port `6333`  
   - Can be started using Docker:  
     ```bash
     docker run -p 6333:6333 qdrant/qdrant
     ```

3. **Ollama (LLM Server)**  
   - Running locally at `http://localhost:11434`  
   - Install from [https://ollama.ai](https://ollama.ai)  
   - Ensure a model is pulled, e.g.:
     ```bash
     ollama pull llama3
     ollama serve
     ```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/QueryX.git
cd QueryX/rag_pipeline
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate        # For Windows
```

### 3. Install Dependencies

```bash
pip install sqlalchemy qdrant-client sentence-transformers requests sqlparse python-dotenv numpy psycopg2-binary
```

---

## File Descriptions and Usage

### 1. config.py

Holds all configuration variables and environment settings for database, Qdrant, and models.
Automatically loads environment variables using `python-dotenv`.

### 2. clients.py

Provides shared, cached clients for:

- SQLAlchemy database engine
- Qdrant client
- SentenceTransformer model

These are reused across scripts to reduce startup time and improve efficiency.

### 3. build_schema.py

**Purpose:**  
Extracts schema information (tables, columns, primary and foreign keys) from the PostgreSQL database, generates text descriptions, embeds them using a SentenceTransformer model, and uploads them to Qdrant.

**Run:**

```bash
python build_schema.py
```

**Output Example:**

```
INFO: Uploaded 5 schema entries to Qdrant collection 'schema_descriptions'.
```

### 4. retrieve_schema.py

**Purpose:**  
Retrieves the most relevant schema descriptions from Qdrant for a given user query, providing the necessary context for SQL generation.

**Run:**

```bash
python retrieve_schema.py --top-k 3
```

**Example Interaction:**

```
Enter English question: List all employees hired after 2020
--- Retrieval Results ---
[1] score=0.9321 table=employees
Table employees contains columns: id (PK), name, hire_date...
```

### 5. generate_sql_from_query.py

**Purpose:**  
Uses the retrieved schema context and a local LLM (via Ollama) to generate a valid SQL SELECT query for a natural-language question. It can optionally execute the query and return results.

**Run (without execution):**

```bash
python generate_sql_from_query.py
```

**Run (execute generated SQL safely):**

```bash
python generate_sql_from_query.py --execute
```

**Example Output:**

```
Enter your question: Show total revenue by department
--- Generated SQL ---
SELECT department, SUM(revenue) AS total_revenue FROM sales GROUP BY department LIMIT 50;
```

---

## Execution Flow Summary

```
build_schema.py → Extract DB schema → Embed → Upload to Qdrant
retrieve_schema.py → Retrieve relevant schema for query
generate_sql_from_query.py → Combine context + LLM → Generate SQL → (Optional) Execute
```

All components share the same configuration and clients through `config.py` and `clients.py`.

---

## Example End-to-End Run

```bash
# 1. Populate schema embeddings
python build_schema.py

# 2. Test schema retrieval
python retrieve_schema.py --top-k 3

# 3. Generate SQL (and execute)
python generate_sql_from_query.py --execute
```

**Expected Output:**

```
INFO: Database engine initialized.
INFO: Qdrant client initialized.
INFO: Embedding model 'sentence-transformers/all-MiniLM-L6-v2' loaded.

Enter your question: List top 5 customers by total purchases
--- Generated SQL ---
SELECT customer_id, SUM(amount) AS total_purchases FROM orders GROUP BY customer_id ORDER BY total_purchases DESC LIMIT 5;
--- Results ---
('customer_id', 'total_purchases')
(1, 12000)
(2, 10500)
...
```

---

## License

This project is licensed under the MIT License.  
See the main repository's LICENSE file for more details.

---

## Maintainers

- Parth Tiwari (2022btech069)
- Harshit Dan (2022btech039)
- Vibhor Saxena (2022btech110)

**Project Supervisor:** Dr. Devika Kataria

---

## Acknowledgements

- **Qdrant Team** — Vector Database
- **Hugging Face** — Sentence Transformers
- **OpenAI & Meta** — LLM Research (RAG integration concepts)
- **Ollama** — Local LLM serving framework