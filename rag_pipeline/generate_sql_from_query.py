# generate_sql_from_query.py
import logging
import requests
import re
import sqlparse
from typing import List
from config import OLLAMA_URL, OLLAMA_MODEL, QDRANT_COLLECTION
from clients import get_db_engine
from retrieve_schema import retrieve_schema
from config import DB_URL
from sqlalchemy import text
from clients import get_embedding_model  # ensure model loaded once
from config import EXECUTE_MAX_ROWS, DEFAULT_TOP_K

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


FORBIDDEN_SQL = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b", re.IGNORECASE
)


def is_sql_safe(sql: str) -> bool:
    if not sql:
        return False
    if FORBIDDEN_SQL.search(sql):
        return False
    return True


def ensure_limit(sql: str, max_rows: int) -> str:
    parsed = sqlparse.parse(sql)
    if not parsed:
        return sql
    statement = parsed[0]
    sql_text = str(statement).strip().rstrip(";")
    if re.search(r"\blimit\b", sql_text, re.IGNORECASE):
        return sql_text + ";"
    return sql_text + f" LIMIT {max_rows};"


def clean_model_output(raw: str) -> str:
    out = raw.strip()
    out = out.replace("```sql", "").replace("```", "")
    out = re.sub(r"(?i)^(sql\s*[:\-]*\s*)", "", out)
    return out.strip()


def build_prompt(question: str, schema_contexts: List[dict], examples: List[dict] = None) -> str:
    schema_lines = []
    for c in schema_contexts:
        table = c.get("table") or "unknown_table"
        desc = c.get("description", "")
        schema_lines.append(f"Table: {table}\n{desc}")
    schema_text = "\n\n".join(schema_lines)

    prompt_parts = [
        "You are a PostgreSQL expert. Given the database schema below, produce a single valid SQL SELECT query answering the user's question.",
        "Return only the SQL query with no additional text or explanation."
    ]
    if examples:
        for ex in examples:
            prompt_parts.append(f"Example question: {ex['q']}\nSQL: {ex['sql']}")
    prompt_parts.append("Database schema (relevant excerpts):")
    prompt_parts.append(schema_text)
    prompt_parts.append(f"User question: {question}")
    prompt_parts.append("Only output the SQL SELECT query (no backticks).")
    return "\n\n".join(prompt_parts)


def call_ollama(prompt: str, model_name: str = None, timeout: int = 120) -> str:
    payload = {"model": model_name or OLLAMA_MODEL, "prompt": prompt, "temperature": 0.1, "stream": False}
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


def execute_sql(sql: str):
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


def generate_and_optionally_execute(question: str, execute: bool = False, top_k: int = DEFAULT_TOP_K):
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
        except Exception as e:
            result["error"] = str(e)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true", help="Execute the generated SQL (SELECT only)")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = p.parse_args()
    q = input("Enter your question: ").strip()
    r = generate_and_optionally_execute(q, execute=args.execute, top_k=args.top_k)
    if r is None:
        print("Failed to generate SQL.")
    else:
        print("\n--- Generated SQL ---")
        print(r.get("sql"))
        if r.get("executed"):
            print("\n--- Results ---")
            cols = r["result"]["columns"]
            rows = r["result"]["rows"]
            print(cols)
            for row in rows:
                print(row)
