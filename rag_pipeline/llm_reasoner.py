# rag_pipeline/llm_reasoner.py
"""
llm_reasoner.py

Orchestrates the RAG -> model -> validate -> execute loop using the fine-tuned T5 model.
Features:
 - Retrieve schema context (Qdrant)
 - Generate SQL using local fine-tuned T5 (or optionally Ollama)
 - Validate SQL with sql_validator
 - Execute SQL against PostgreSQL (safe, read-only)
 - Self-correction: if execution fails, ask model to fix SQL using DB error feedback
"""

import os
import time
from typing import List, Dict, Optional
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from retrieve_schema import retrieve_schema
from sql_validator import validate_sql
from clients import get_db_engine
from utils.llm_logger import logger
from config import (
    MODEL_PATH,
    USE_OLLAMA_FALLBACK,
    OLLAMA_URL,
    OLLAMA_MODEL,
    EXECUTE_MAX_ROWS,
    DEFAULT_TOP_K,
)
import requests
from sqlalchemy import text

# Load T5 model and tokenizer once (lazy)
_t5_tokenizer = None
_t5_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_t5_model(model_path: str = None):
    global _t5_tokenizer, _t5_model
    if _t5_model is not None and _t5_tokenizer is not None:
        return _t5_tokenizer, _t5_model
    path = model_path or MODEL_PATH
    logger.info("Loading T5 model from %s (device=%s)", path, _device)
    _t5_tokenizer = T5Tokenizer.from_pretrained(path)
    _t5_model = T5ForConditionalGeneration.from_pretrained(path).to(_device)
    _t5_model.eval()
    return _t5_tokenizer, _t5_model


def build_prompt_for_t5(question: str, schema_contexts: List[Dict], examples: Optional[List[Dict]] = None) -> str:
    schema_lines = []
    for c in schema_contexts:
        table = c.get("table") or "unknown_table"
        desc = c.get("description", "").strip()
        schema_lines.append(f"Table: {table}\n{desc}")
    schema_text = "\n\n".join(schema_lines)

    prompt_parts = [
        "Translate the following natural language question into a single valid PostgreSQL SELECT query.",
        "Do not include explanations or markdown formatting. Output only the SQL query.",
        "Relevant schema:",
        schema_text,
        f"Question: {question}"
    ]
    if examples:
        for ex in examples:
            prompt_parts.insert(0, f"Example question: {ex['q']}\nSQL: {ex['sql']}")
    return "\n\n".join(prompt_parts)


def generate_sql_with_t5(question: str, schema_contexts: List[Dict], max_length: int = 256, examples: Optional[List[Dict]] = None) -> str:
    tokenizer, model = load_t5_model()
    prompt = build_prompt_for_t5(question, schema_contexts, examples)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest", max_length=512).to(_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return sql


def call_ollama(prompt: str, timeout: int = 120) -> str:
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "temperature": 0.1, "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if r.status_code != 200:
            logger.error("Ollama returned %d: %s", r.status_code, r.text)
            return ""
        data = r.json()
        return data.get("response", "") or data.get("text", "") or ""
    except Exception as e:
        logger.error("Ollama request failed: %s", e)
        return ""


def ensure_limit_clause(sql: str, max_rows: int = EXECUTE_MAX_ROWS) -> str:
    sql_clean = sql.strip().rstrip(";")
    if "limit" in sql_clean.lower():
        return sql_clean + ";"
    return sql_clean + f" LIMIT {max_rows};"


def execute_sql_return(sql: str):
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            res = conn.execute(text(sql))
            rows = res.fetchall()
            cols = res.keys()
            return {"columns": cols, "rows": rows}
    except Exception as e:
        logger.error("Execution error: %s", e)
        raise


def self_correct_sql(original_sql: str, error_msg: str, question: str, schema_contexts: List[Dict]) -> str:
    """
    Ask the model to correct SQL using the DB error message and schema context.
    Uses T5 for correction.
    """
    prompt = (
        "A previously generated SQL query failed with the following database error:\n\n"
        f"{error_msg}\n\n"
        "Original SQL:\n"
        f"{original_sql}\n\n"
        "Given the schema below, correct the SQL so it runs without error. Output only the corrected SQL.\n\n"
    )
    # Append schema
    for c in schema_contexts:
        t = c.get("table") or "unknown_table"
        d = c.get("description", "")
        prompt += f"Table: {t}\n{d}\n\n"
    prompt += f"Question: {question}\n"

    # use T5 to correct
    tokenizer, model = load_t5_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest", max_length=512).to(_device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    logger.info("Self-correction attempt produced SQL: %s", corrected)
    return corrected


def infer_and_execute(
    question: str,
    execute: bool = True,
    top_k: int = DEFAULT_TOP_K,
    max_correction_attempts: int = 1,
    use_ollama_fallback: bool = False,
):
    logger.info("Start inference for question: %s", question)
    schema_contexts = retrieve_schema(question, top_k=top_k)
    if not schema_contexts:
        logger.warning("No schema context returned.")
        return {"error": "No schema context found."}

    # Generate SQL (prefer local fine-tuned T5)
    try:
        sql = generate_sql_with_t5(question, schema_contexts)
        logger.info("Generated SQL (T5): %s", sql)
    except Exception as e:
        logger.error("T5 generation failed: %s", e)
        sql = ""
        if use_ollama_fallback or USE_OLLAMA_FALLBACK:
            prompt = build_prompt_for_t5(question, schema_contexts)
            sql = call_ollama(prompt)
            logger.info("Generated SQL (Ollama fallback): %s", sql)

    sql = (sql or "").strip()
    if not sql:
        logger.error("No SQL produced by model.")
        return {"error": "Model returned empty SQL."}

    if not validate_sql(sql):
        logger.error("SQL validation failed for: %s", sql)
        return {"sql": sql, "valid": False, "executed": False, "error": "Validation failed."}

    if "limit" not in sql.lower():
        sql_to_run = ensure_limit_clause(sql, EXECUTE_MAX_ROWS)
    else:
        sql_to_run = sql if sql.strip().endswith(";") else sql + ";"

    result = {"sql": sql_to_run, "valid": True, "executed": False, "result": None}

    if not execute:
        logger.info("Execution disabled, returning SQL only.")
        return result

    attempts = 0
    while attempts <= max_correction_attempts:
        try:
            out = execute_sql_return(sql_to_run)
            result["executed"] = True
            result["result"] = out
            logger.info("SQL executed successfully.")
            return result
        except Exception as e:
            attempts += 1
            err_msg = str(e)
            logger.error("Execution failed (attempt %d): %s", attempts, err_msg)
            if attempts > max_correction_attempts:
                result["error"] = err_msg
                return result
            # Ask model to correct SQL using DB error
            try:
                corrected_sql = self_correct_sql(sql_to_run, err_msg, question, schema_contexts)
                if not corrected_sql:
                    logger.error("Self-correction produced empty SQL.")
                    result["error"] = "Self-correction failed."
                    return result
                if not validate_sql(corrected_sql):
                    logger.error("Corrected SQL failed validation: %s", corrected_sql)
                    result["error"] = "Corrected SQL invalid."
                    return result
                # ensure limit and set for next attempt
                sql_to_run = ensure_limit_clause(corrected_sql, EXECUTE_MAX_ROWS)
                logger.info("Retrying with corrected SQL: %s", sql_to_run)
            except Exception as inner:
                logger.error("Self-correction process failed: %s", inner)
                result["error"] = str(inner)
                return result

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Reasoner: generate and optionally execute SQL from NL.")
    parser.add_argument("--no-exec", action="store_true", help="Do not execute SQL; return SQL only.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k schema retrieval.")
    parser.add_argument("--attempts", type=int, default=1, help="Number of self-correction attempts on failure.")
    args = parser.parse_args()

    q = input("Enter your question: ").strip()
    res = infer_and_execute(q, execute=(not args.no_exec), top_k=args.top_k, max_correction_attempts=args.attempts)
    print("\n--- Result ---")
    if "sql" in res:
        print("SQL:", res["sql"])
    if res.get("executed"):
        print("Columns:", res["result"]["columns"])
        for row in res["result"]["rows"]:
            print(row)
    if res.get("error"):
        print("Error:", res["error"])
