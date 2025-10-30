"""
generate_sql_from_query.py
------------------------------------
Generates SQL queries from natural language using an LLM (via Ollama),
validates them for safety, and optionally executes them on PostgreSQL.
Integrates with:
 - Qdrant for schema retrieval
 - Ollama for LLM-based SQL generation
 - PostgreSQL via SQLAlchemy
 - sql_validator.py for pre-execution safety
------------------------------------
"""

import logging
import requests
import re
import sqlparse
from typing import List
from sqlalchemy import text
from config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    QDRANT_COLLECTION,
    EXECUTE_MAX_ROWS,
    DEFAULT_TOP_K,
)
from clients import get_db_engine, get_embedding_model
from retrieve_schema import retrieve_schema
from sql_validator import validate_sql  # ✅ import validator

# Logging setup
logging.basicConfig(
    filename="rag_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Fallback forbidden keyword regex
FORBIDDEN_SQL = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


# ---------------------------
# Validation and Formatting
# ---------------------------

def is_sql_safe(sql: str) -> bool:
    """Ensure SQL does not contain forbidden statements."""
    if not sql:
        return False
    return not FORBIDDEN_SQL.search(sql)


def ensure_limit(sql: str, max_rows: int) -> str:
    """Guarantee that a LIMIT clause is present."""
    parsed = sqlparse.parse(sql)
    if not parsed:
        return sql
    statement = parsed[0]
    sql_text = str(statement).strip().rstrip(";")
    if re.search(r"\blimit\b", sql_text, re.IGNORECASE):
        return sql_text + ";"
    return sql_text + f" LIMIT {max_rows};"


def clean_model_output(raw: str) -> str:
    """Clean up extra formatting or code fences in LLM response."""
    out = raw.strip()
    out = out.replace("```sql", "").replace("```", "")
    out = re.sub(r"(?i)^(sql\s*[:\-]*\s*)", "", out)
    return out.strip()


# ---------------------------
# Prompt and Model Interaction
# ---------------------------

def build_prompt(question: str, schema_contexts: List[dict], examples: List[dict] = None) -> str:
    """Construct the full prompt with schema + user question."""
    schema_lines = []
    for c in schema_contexts:
        table = c.get("table") or "unknown_table"
        desc = c.get("description", "")
        schema_lines.append(f"Table: {table}\n{desc}")
    schema_text = "\n\n".join(schema_lines)

    prompt_parts = [
        "You are a PostgreSQL expert. Given the database schema below, produce a single valid SQL SELECT query answering the user's question.",
        "Return only the SQL query with no explanations or markdown formatting.",
    ]
    if examples:
        for ex in examples:
            prompt_parts.append(f"Example question: {ex['q']}\nSQL: {ex['sql']}")
    prompt_parts.append("Database schema (relevant excerpts):")
    prompt_parts.append(schema_text)
    prompt_parts.append(f"User question: {question}")
    prompt_parts.append("Output only the SQL SELECT query (no backticks).")
    return "\n\n".join(prompt_parts)


def call_ollama(prompt: str, model_name: str = None, timeout: int = 120) -> str:
    """Send prompt to Ollama model and get SQL output."""
    payload = {
        "model": model_name or OLLAMA_MODEL,
        "prompt": prompt,
        "temperature": 0.1,
        "stream": False,
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if r.status_code != 200:
            logging.error(f"Ollama returned {r.status_code}: {r.text}")
            return ""
        data = r.json()
        return data.get("response", "") or data.get("text", "") or ""
    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama request failed: {e}")
        return ""


# ---------------------------
# SQL Execution
# ---------------------------

def execute_sql(sql: str):
    """Execute validated SQL on PostgreSQL."""
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            res = conn.execute(text(sql))
            rows = res.fetchall()
            cols = res.keys()
            return {"columns": cols, "rows": rows}
    except Exception as e:
        logging.error(f"SQL execution error: {e}")
        raise


# ---------------------------
# Main Logic
# ---------------------------

def generate_and_optionally_execute(question: str, execute: bool = False, top_k: int = DEFAULT_TOP_K):
    """Full flow: retrieve schema, prompt LLM, validate, and (optionally) execute."""
    logging.info(f"Received question: {question}")

    contexts = retrieve_schema(question, top_k=top_k)
    if not contexts:
        logging.warning("No schema context retrieved.")
        return None

    prompt = build_prompt(question, contexts)
    raw = call_ollama(prompt)
    if not raw:
        logging.error("No response from LLM.")
        return None

    sql = clean_model_output(raw)
    if not sql:
        logging.error("LLM returned empty SQL.")
        return None

    # ✅ Apply strict validation
    if not validate_sql(sql):
        logging.error("SQL failed validation checks.")
        return {"sql": sql, "safe": False, "executed": False, "error": "Invalid SQL structure"}

    if not is_sql_safe(sql):
        logging.error("Generated SQL contains forbidden statements. Aborting execution.")
        return {"sql": sql, "safe": False, "executed": False}

    sql_with_limit = ensure_limit(sql, EXECUTE_MAX_ROWS)
    result = {"sql": sql_with_limit, "safe": True, "executed": False, "result": None}

    if execute:
        try:
            out = execute_sql(sql_with_limit)
            result["executed"] = True
            result["result"] = out
            logging.info(f"Successfully executed SQL: {sql_with_limit}")
        except Exception as e:
            logging.error(f"Query execution failed: {e}")
            result["error"] = str(e)
    return result


# ---------------------------
# CLI Entry Point
# ---------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute the generated SQL (SELECT only)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    question = input("Enter your question: ").strip()
    result = generate_and_optionally_execute(question, execute=args.execute, top_k=args.top_k)

    if result is None:
        print("Failed to generate SQL.")
    else:
        print("\n--- Generated SQL ---")
        print(result.get("sql"))
        if result.get("executed"):
            print("\n--- Results ---")
            cols = result["result"]["columns"]
            rows = result["result"]["rows"]
            print(cols)
            for row in rows:
                print(row)
