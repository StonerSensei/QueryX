# rag_pipeline/llm_reasoner.py
"""
llm_reasoner.py

Orchestrates the RAG -> model -> validate -> execute loop using the fine-tuned T5 model.

Features:
 - Retrieve schema context (Qdrant)
 - (Optionally) re-rank schema context with SchemaAttention
 - Generate SQL using local fine-tuned T5 (or optionally Ollama)
 - Validate SQL with sql_validator
 - Execute SQL against PostgreSQL (safe, read-only)
 - Self-correction: if execution fails, ask model to fix SQL using DB error feedback
"""

import os
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import requests
from sqlalchemy import text, inspect

from retrieve_schema import retrieve_schema
from sql_validator import validate_sql, enforce_limit
from clients import get_db_engine
from utils.llm_logger import logger
from config import (
    MODEL_PATH,
    USE_OLLAMA_FALLBACK,
    OLLAMA_URL,
    OLLAMA_MODEL,
    EXECUTE_MAX_ROWS,
    DEFAULT_TOP_K,
    EMBED_MODEL,
)

# Lazy-loaded globals
_t5_tokenizer = None
_t5_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_schema_att_model = None
_schema_att_loaded = False


# ---------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------
def load_t5_model(model_path: str = None):
    """Lazy-load fine-tuned FLAN-T5 model for NL->SQL."""
    global _t5_tokenizer, _t5_model
    if _t5_model is not None and _t5_tokenizer is not None:
        return _t5_tokenizer, _t5_model

    path = model_path or MODEL_PATH
    logger.info("Loading T5 model from %s (device=%s)", path, _device)

    # ✅ use the same fast tokenizer as training
    _t5_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    _t5_model = T5ForConditionalGeneration.from_pretrained(path).to(_device)
    _t5_model.eval()
    return _t5_tokenizer, _t5_model



def get_schema_attention_model():
    """
    Lazy-load the SchemaAttention model if available.

    If models/schema_attention.pt is missing or loading fails, we skip
    re-ranking and just use Qdrant's cosine similarity ranking.
    """
    global _schema_att_model, _schema_att_loaded

    if _schema_att_loaded:
        return _schema_att_model

    _schema_att_loaded = True
    state_path = "models/schema_attention.pt"

    try:
        if not os.path.exists(state_path):
            logger.warning(
                "Schema attention weights not found at %s; "
                "skipping attention-based re-ranking.",
                state_path,
            )
            _schema_att_model = None
            return _schema_att_model

        from schema_attention import SchemaAttention  # your custom module

        logger.info("Loading SchemaAttention model from %s", state_path)
        model = SchemaAttention(embed_model_name=EMBED_MODEL)
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _schema_att_model = model
        return _schema_att_model
    except Exception as e:
        logger.error("Failed to load SchemaAttention model: %s", e)
        _schema_att_model = None
        return _schema_att_model

def build_schema_string_from_db(schema_contexts: List[Dict]) -> str:
    """
    Rebuild the schema text in the SAME format used during training, e.g.:

      "Table mytable(Age, Attrition, BusinessTravel, ...)"

    using live DB introspection via SQLAlchemy. We only use table names
    from schema_contexts; Qdrant is responsible for picking relevant tables.
    """
    engine = get_db_engine()
    inspector = inspect(engine)

    # Preserve order and uniqueness of tables from schema_contexts
    table_names = []
    for c in schema_contexts:
        t = c.get("table")
        if t and t not in table_names:
            table_names.append(t)

    parts = []
    for t in table_names:
        try:
            cols = inspector.get_columns(t)
            col_names = [col["name"] for col in cols]
            # Match dataset_generator style: Table mytable(col1, col2, ...)
            parts.append(f"Table {t}({', '.join(col_names)})")
        except Exception as e:
            logger.error("Failed to build schema string for table %s: %s", t, e)

    # If multiple tables, join them with space (you currently have only one)
    return " ".join(parts)



# ---------------------------------------------------------------------
# Prompt & Generation
# ---------------------------------------------------------------------
def build_prompt_for_t5(
    question: str,
    schema_contexts: List[Dict],
    examples: Optional[List[Dict]] = None,
) -> str:
    """
    Build input prompt for FLAN-T5.

    IMPORTANT: We keep this format consistent with the training data:
      "Schema: Table mytable(...)\nQuestion: ..."
    """
    schema_str = build_schema_string_from_db(schema_contexts)
    prompt = f"Schema: {schema_str}\nQuestion: {question}"
    return prompt



def generate_sql_with_t5(
    question: str,
    schema_contexts: List[Dict],
    max_length: int = 256,
    examples: Optional[List[Dict]] = None,
) -> str:
    """
    Generate SQL from question + schema context using fine-tuned T5.
    """
    tokenizer, model = load_t5_model()
    prompt = build_prompt_for_t5(question, schema_contexts, examples)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=512,
    ).to(_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return sql


def call_ollama(prompt: str, timeout: int = 120) -> str:
    """
    Fallback to Ollama if local T5 fails or is not desired.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "temperature": 0.1,
            "stream": False,
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if r.status_code != 200:
            logger.error("Ollama returned %d: %s", r.status_code, r.text)
            return ""
        data = r.json()
        return data.get("response", "") or data.get("text", "") or ""
    except Exception as e:
        logger.error("Ollama request failed: %s", e)
        return ""


# ---------------------------------------------------------------------
# Validation & Execution
# ---------------------------------------------------------------------
def ensure_limit_clause(sql: str, max_rows: int = EXECUTE_MAX_ROWS) -> str:
    """
    Ensure the SQL has a LIMIT and clamp it to max_rows by delegating to sql_validator.enforce_limit.
    """
    return enforce_limit(sql, max_rows)


def execute_sql_return(sql: str):
    """
    Execute SQL safely against PostgreSQL and return rows + columns.
    """
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


def self_correct_sql(
    original_sql: str,
    error_msg: str,
    question: str,
    schema_contexts: List[Dict],
) -> str:
    """
    Ask the model to correct SQL using the DB error message and schema context.
    Uses T5 for correction.
    """
    prompt = (
        "A previously generated SQL query failed with the following database error:\n\n"
        f"{error_msg}\n\n"
        "Original SQL:\n"
        f"{original_sql}\n\n"
        "Given the schema below, correct the SQL so it runs without error. "
        "The corrected query must be a single read-only SELECT/CTE query. "
        "Output only the corrected SQL.\n\n"
    )

    for c in schema_contexts:
        t = c.get("table") or "unknown_table"
        d = c.get("description", "")
        prompt += f"Table: {t}\n{d}\n\n"

    prompt += f"Question: {question}\n"

    tokenizer, model = load_t5_model()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=512,
    ).to(_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    logger.info("Self-correction attempt produced SQL: %s", corrected)
    return corrected


# ---------------------------------------------------------------------
# Schema Re-ranking
# ---------------------------------------------------------------------
def _rerank_schema_with_attention(
    question: str,
    schema_contexts: List[Dict],
    top_k: int,
) -> List[Dict]:
    """
    Optional second-stage schema ranking using SchemaAttention.

    Takes initial schema_contexts (from Qdrant cosine similarity),
    scores them with the attention model, and returns top_k.
    """
    att_model = get_schema_attention_model()
    if att_model is None or len(schema_contexts) <= 1:
        return schema_contexts[:top_k]

    try:
        schema_texts = [c.get("description", "") for c in schema_contexts]
        with torch.no_grad():
            scores = att_model(question, schema_texts)  # [N]
        scores = scores.detach().cpu().numpy().tolist()

        for c, s in zip(schema_contexts, scores):
            c["att_score"] = float(s)

        schema_contexts.sort(
            key=lambda x: x.get("att_score", 0.0),
            reverse=True,
        )
        return schema_contexts[:top_k]
    except Exception as e:
        logger.error("Schema attention ranking failed: %s", e)
        return schema_contexts[:top_k]


# ---------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------
def infer_and_execute(
    question: str,
    execute: bool = True,
    top_k: int = DEFAULT_TOP_K,
    max_correction_attempts: int = 1,
    use_ollama_fallback: bool = False,
):
    """
    Main entry point:
      1) Retrieve schema from Qdrant
      2) Re-rank with SchemaAttention (if available)
      3) Generate SQL with T5 (or Ollama fallback)
      4) Validate & execute (with self-correction loop)
    """
    logger.info("Start inference for question: %s", question)

    # 1) Coarse retrieval from Qdrant
    coarse_k = max(top_k * 2, top_k)
    schema_contexts = retrieve_schema(question, top_k=coarse_k)

    if not schema_contexts:
        logger.warning(
            "No schema context returned from Qdrant. "
            "Make sure Qdrant is running and build_schema.py has been executed."
        )
        return {"error": "No schema context found. Please rebuild schema index."}

    # 2) Fine re-ranking using SchemaAttention (if available)
    schema_contexts = _rerank_schema_with_attention(
        question, schema_contexts, top_k=top_k
    )

    # 3) Generate SQL (prefer local fine-tuned T5)
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

    # 4) Validate
    if not validate_sql(sql):
        logger.error("SQL validation failed for: %s", sql)
        return {
            "sql": sql,
            "valid": False,
            "executed": False,
            "error": "Validation failed.",
        }

    # Ensure safe LIMIT using enforce_limit wrapper
    sql_to_run = ensure_limit_clause(sql, EXECUTE_MAX_ROWS)

    result = {
        "sql": sql_to_run,
        "valid": True,
        "executed": False,
        "result": None,
    }

    if not execute:
        logger.info("Execution disabled, returning SQL only.")
        return result

    # 5) Execution + self-correction loop
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
            logger.error(
                "Execution failed (attempt %d / %d): %s",
                attempts,
                max_correction_attempts,
                err_msg,
            )

            if attempts > max_correction_attempts:
                result["error"] = err_msg
                return result

            # Ask model to correct SQL using DB error
            try:
                corrected_sql = self_correct_sql(
                    sql_to_run, err_msg, question, schema_contexts
                )
                if not corrected_sql:
                    logger.error("Self-correction produced empty SQL.")
                    result["error"] = "Self-correction failed."
                    return result

                if not validate_sql(corrected_sql):
                    logger.error(
                        "Corrected SQL failed validation: %s",
                        corrected_sql,
                    )
                    result["error"] = "Corrected SQL invalid."
                    return result

                sql_to_run = ensure_limit_clause(corrected_sql, EXECUTE_MAX_ROWS)
                logger.info("Retrying with corrected SQL: %s", sql_to_run)
            except Exception as inner:
                logger.error("Self-correction process failed: %s", inner)
                result["error"] = str(inner)
                return result

    return result


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM Reasoner: generate and optionally execute SQL from NL."
    )
    parser.add_argument(
        "--no-exec",
        action="store_true",
        help="Do not execute SQL; return SQL only.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k schema retrieval.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of self-correction attempts on failure.",
    )
    args = parser.parse_args()

    q = input("Enter your question: ").strip()
    res = infer_and_execute(
        q,
        execute=(not args.no_exec),
        top_k=args.top_k,
        max_correction_attempts=args.attempts,
    )

    print("\n--- Result ---")
    if "sql" in res:
        print("SQL:", res["sql"])
    if res.get("executed"):
        print("Columns:", res["result"]["columns"])
        for row in res["result"]["rows"]:
            print(row)
    if res.get("error"):
        print("Error:", res["error"])
