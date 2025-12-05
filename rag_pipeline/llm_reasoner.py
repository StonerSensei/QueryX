# rag_pipeline/llm_reasoner.py
"""
Orchestrates the RAG -> model -> validate -> execute loop using a fine-tuned T5 model.

Highlights:
- Retrieve schema context (Qdrant)
- (Optional) re-rank with SchemaAttention (if present)
- Generate SQL with T5; if not SQL, force SELECT/WITH prefix and retry
- If still not SQL, try instruction-hardened prompt, then Ollama fallback
- Last-resort rule-based SQL for common intents (distinct / count)
- Validate SQL, enforce LIMIT, execute safely
- Self-correction using DB error feedback
- --debug flag prints prompt, raw/forced/fallback SQL
"""

import os
import re
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

# --- Globals -----------------------------------------------------------------
_t5_tokenizer = None
_t5_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_schema_att_model = None
_schema_att_loaded = False

SELECT_PREFIX = "SELECT "
WITH_PREFIX   = "WITH "

# --- Helpers -----------------------------------------------------------------
def _looks_like_sql(s: str) -> bool:
    s0 = (s or "").lstrip().upper()
    return s0.startswith("SELECT") or s0.startswith("WITH")

def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # remove ```sql ... ``` or ``` ... ```
    s = re.sub(r"^```(?:sql)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    # remove leading “SQL:” etc.
    s = re.sub(r"^\s*sql\s*:\s*", "", s, flags=re.I)
    return s.strip()

def _force_prefix_generate(tokenizer, model, prefix_text, max_length=256):
    """
    Force the decoder to start with a prefix (SELECT/WITH) and let the model continue.
    """
    dec = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(_device)
    enc = tokenizer("", return_tensors="pt")["input_ids"].to(_device)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc,
            decoder_input_ids=dec,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def _extract_columns(schema_contexts: List[Dict]) -> List[str]:
    """Parse column names from the verbose description stored in Qdrant."""
    cols = set()
    for c in schema_contexts:
        for line in (c.get("description") or "").splitlines():
            m = re.match(r"^\s*-\s*([A-Za-z0-9_]+)\s*:\s*[A-Z]+", line)
            if m:
                cols.add(m.group(1))
    return sorted(cols)

def _best_column_for_query(query: str, columns: List[str]) -> Optional[str]:
    q = query.lower()
    # direct token match
    for col in columns:
        if col.lower() in q or q in col.lower():
            return col
    # common synonyms
    synonyms = {
        "gender": "Gender",
        "sex": "Gender",
        "income": "MonthlyIncome",
        "role": "JobRole",
        "department": "Department",
        "marital": "MaritalStatus",
        "education": "EducationField",
        "age": "Age",
    }
    for k, v in synonyms.items():
        if k in q and v in columns:
            return v
    # text-ish fallback
    textish = [c for c in columns if re.search(r"(name|role|dept|gender|status|field|title)", c, re.I)]
    return textish[0] if textish else (columns[0] if columns else None)

def _rule_based_sql(question: str, schema_contexts: List[Dict], max_rows: int) -> str:
    """
    Last-resort heuristic covering:
      - "name/list all ..." -> SELECT DISTINCT <col>
      - "how many/count ..." -> COUNT(*) with an optional equality filter
    """
    cols = _extract_columns(schema_contexts)
    if not cols:
        return ""
    table = schema_contexts[0].get("table") or "mytable"
    q = question.lower()

    # DISTINCT
    if any(k in q for k in ["name all", "list all", "unique", "distinct"]) and "count" not in q:
        col = _best_column_for_query(q, cols)
        if col:
            return f'SELECT DISTINCT "{col}" FROM {table} LIMIT {max_rows};'

    # COUNT (try to capture a value)
    if any(k in q for k in ["how many", "count", "number of"]):
        col = _best_column_for_query(q, cols)
        m = re.search(r"'([^']+)'", question)   # quoted value
        val = m.group(1) if m else None
        if not val:
            # common unquoted tokens
            m2 = re.search(r"\b(female|male|sales|research|life sciences|hr|human resources)\b", q)
            val = m2.group(1).title() if m2 else None
        if col and val:
            return f'SELECT COUNT(*) FROM {table} WHERE "{col}" = \'{val}\';'
        elif col:
            return f"SELECT COUNT(*) FROM {table};"

    # generic
    return f"SELECT * FROM {table} LIMIT {max_rows};"

# --- Model Loading ------------------------------------------------------------
def load_t5_model(model_path: str = None):
    """Lazy-load fine-tuned FLAN-T5 model for NL->SQL."""
    global _t5_tokenizer, _t5_model
    if _t5_model is not None and _t5_tokenizer is not None:
        return _t5_tokenizer, _t5_model

    path = model_path or MODEL_PATH
    logger.info("Loading T5 model from %s (device=%s)", path, _device)
    _t5_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    _t5_model = T5ForConditionalGeneration.from_pretrained(path).to(_device)
    _t5_model.eval()
    return _t5_tokenizer, _t5_model

# --- Optional Schema Attention ------------------------------------------------
def get_schema_attention_model():
    """
    Lazy-load SchemaAttention if available. Safe to skip if missing.
    """
    global _schema_att_model, _schema_att_loaded
    if _schema_att_loaded:
        return _schema_att_model
    _schema_att_loaded = True

    state_path = "models/schema_attention.pt"
    try:
        if not os.path.exists(state_path):
            logger.warning("Schema attention weights not found at %s; skipping attention re-ranking.", state_path)
            _schema_att_model = None
            return _schema_att_model

        from schema_attention import SchemaAttention  # optional module you may add/train
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

# --- Prompting & Generation ---------------------------------------------------
def build_schema_string_from_db(schema_contexts: List[Dict]) -> str:
    """
    Rebuild schema text in the SAME format as training:
        "Table mytable(col1, col2, ...)"
    """
    engine = get_db_engine()
    inspector = inspect(engine)

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
            parts.append(f"Table {t}({', '.join(col_names)})")
        except Exception as e:
            logger.error("Failed to introspect table %s: %s", t, e)
    return " ".join(parts)

def build_prompt_for_t5(question: str, schema_contexts: List[Dict], examples: Optional[List[Dict]] = None) -> str:
    """
    EXACT training format:
        "Schema: Table mytable(...)\nQuestion: ..."
    (We avoid extra instruction lines to stay close to the fine-tuning distribution.)
    """
    schema_str = build_schema_string_from_db(schema_contexts)
    return f"Schema: {schema_str}\nQuestion: {question}"

def _instruction_hardened_prompt(question: str, schema_contexts: List[Dict]) -> str:
    """
    When generation fails, try a slightly more instruction-driven prompt while keeping structure.
    """
    schema_str = build_schema_string_from_db(schema_contexts)
    return (
        f"Schema: {schema_str}\n"
        f"Question: {question}\n"
        f"Return only a single valid PostgreSQL SELECT or WITH (CTE) query. No explanations."
    )

def generate_sql_with_t5(question: str, schema_contexts: List[Dict], max_length: int = 256,
                         examples: Optional[List[Dict]] = None, debug: bool = False) -> str:
    tokenizer, model = load_t5_model()

    # 1) Plain prompt (matches training)
    prompt = build_prompt_for_t5(question, schema_contexts, examples)
    if debug:
        print("\n==== PROMPT BEGIN ====\n", prompt, "\n==== PROMPT END ====\n")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest", max_length=512).to(_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    sql = _strip_code_fences(tokenizer.decode(outputs[0], skip_special_tokens=True).strip())
    if debug:
        print(">>> RAW SQL:", sql)

    # 2) If not SQL, try forced prefixes
    if not _looks_like_sql(sql):
        logger.warning("Model output didn't start with SELECT/WITH. Trying forced-prefix decoding...")
        sql2 = _strip_code_fences(_force_prefix_generate(tokenizer, model, SELECT_PREFIX, max_length=max_length))
        if debug:
            print(">>> FORCED SELECT SQL:", sql2)
        if _looks_like_sql(sql2):
            return sql2

        sql3 = _strip_code_fences(_force_prefix_generate(tokenizer, model, WITH_PREFIX, max_length=max_length))
        if debug:
            print(">>> FORCED WITH SQL:", sql3)
        if _looks_like_sql(sql3):
            return sql3

    return sql

# --- Ollama Fallback ----------------------------------------------------------
def call_ollama(prompt: str, timeout: int = 120) -> str:
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "temperature": 0.0, "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if r.status_code != 200:
            logger.error("Ollama returned %d: %s", r.status_code, r.text)
            return ""
        data = r.json()
        # API variants differ: prefer 'response', fallback to 'text'
        return (data.get("response") or data.get("text") or "").strip()
    except Exception as e:
        logger.error("Ollama request failed: %s", e)
        return ""

# --- Validation & Execution ---------------------------------------------------
def ensure_limit_clause(sql: str, max_rows: int = EXECUTE_MAX_ROWS) -> str:
    return enforce_limit(sql, max_rows)

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

# --- Self-Correction ----------------------------------------------------------
def self_correct_sql(original_sql: str, error_msg: str, question: str, schema_contexts: List[Dict]) -> str:
    prompt = (
        "A previously generated SQL query failed with the following database error:\n\n"
        f"{error_msg}\n\n"
        "Original SQL:\n"
        f"{original_sql}\n\n"
        "Given the schema below, correct the SQL so it runs without error. "
        "The corrected query must be a single read-only PostgreSQL SELECT/CTE query. "
        "Return only the corrected SQL.\n\n"
    )
    for c in schema_contexts:
        t = c.get("table") or "unknown_table"
        d = c.get("description", "")
        prompt += f"Table: {t}\n{d}\n\n"
    prompt += f"Question: {question}\n"

    tokenizer, model = load_t5_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest", max_length=512).to(_device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
    corrected = _strip_code_fences(tokenizer.decode(outputs[0], skip_special_tokens=True).strip())
    logger.info("Self-correction attempt produced SQL: %s", corrected)
    return corrected

# --- Optional Re-ranking ------------------------------------------------------
def _rerank_schema_with_attention(question: str, schema_contexts: List[Dict], top_k: int) -> List[Dict]:
    att_model = get_schema_attention_model()
    if att_model is None or len(schema_contexts) <= 1:
        return schema_contexts[:top_k]
    try:
        with torch.no_grad():
            texts = [c.get("description", "") for c in schema_contexts]
            scores = att_model(question, texts).detach().cpu().numpy().tolist()
        for c, s in zip(schema_contexts, scores):
            c["att_score"] = float(s)
        schema_contexts.sort(key=lambda x: x.get("att_score", 0.0), reverse=True)
        return schema_contexts[:top_k]
    except Exception as e:
        logger.error("Schema attention ranking failed: %s", e)
        return schema_contexts[:top_k]

# --- Main Orchestration -------------------------------------------------------
def infer_and_execute(
    question: str,
    execute: bool = True,
    top_k: int = DEFAULT_TOP_K,
    max_correction_attempts: int = 1,
    use_ollama_fallback: bool = False,
    debug: bool = False,
):
    logger.info("Start inference for question: %s", question)

    # 1) Retrieve schema
    coarse_k = max(top_k * 2, top_k)
    schema_contexts = retrieve_schema(question, top_k=coarse_k)
    if not schema_contexts:
        logger.warning("No schema context returned from Qdrant. Make sure build_schema.py has been executed.")
        return {"error": "No schema context found. Please rebuild schema index."}

    # 2) (optional) attention re-ranking
    schema_contexts = _rerank_schema_with_attention(question, schema_contexts, top_k=top_k)

    # 3) Generate SQL (local T5)
    sql = ""
    try:
        sql = generate_sql_with_t5(question, schema_contexts, debug=debug)
    except Exception as e:
        logger.error("T5 generation failed: %s", e)
        sql = ""

    sql = _strip_code_fences((sql or "").strip())

    # 3.a) If still not SQL, try an instruction-hardened prompt
    if not _looks_like_sql(sql):
        logger.warning("Model output not SQL; trying instruction-hardened prompt.")
        tokenizer, model = load_t5_model()
        prompt2 = _instruction_hardened_prompt(question, schema_contexts)
        if debug:
            print("\n==== INSTR PROMPT BEGIN ====\n", prompt2, "\n==== INSTR PROMPT END ====\n")
        inputs2 = tokenizer(prompt2, return_tensors="pt", truncation=True, padding="longest", max_length=512).to(_device)
        with torch.no_grad():
            out2 = model.generate(**inputs2, max_length=256, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
        sql2 = _strip_code_fences(tokenizer.decode(out2[0], skip_special_tokens=True).strip())
        if debug:
            print(">>> INSTR RAW SQL:", sql2)
        if _looks_like_sql(sql2):
            sql = sql2

    # 3.b) If still not SQL, Ollama fallback (if enabled)
    if not _looks_like_sql(sql) and (use_ollama_fallback or USE_OLLAMA_FALLBACK):
        logger.warning("Falling back to Ollama...")
        ollama_prompt = (
            "Translate the question to a single valid PostgreSQL SELECT (or WITH/CTE) query.\n"
            "Return ONLY the SQL (no explanation, no formatting).\n\n"
            f"{build_prompt_for_t5(question, schema_contexts)}"
        )
        sql_ol = _strip_code_fences(call_ollama(ollama_prompt))
        if debug:
            print(">>> OLLAMA RAW:", sql_ol)
        if _looks_like_sql(sql_ol):
            sql = sql_ol

    # 3.c) Rule-based last resort
    if not _looks_like_sql(sql):
        logger.warning("Non-SQL output after all attempts. Applying rule-based fallback.")
        sql = _rule_based_sql(question, schema_contexts, EXECUTE_MAX_ROWS)
        if debug:
            print(">>> RULE-BASED SQL:", sql)

    if not sql:
        return {"error": "Model returned empty SQL."}

    # 4) Validate
    if not validate_sql(sql):
        logger.error("SQL validation failed for: %s", sql)
        return {"sql": sql, "valid": False, "executed": False, "error": "Validation failed."}

    # 5) Enforce LIMIT
    sql_to_run = ensure_limit_clause(sql, EXECUTE_MAX_ROWS)
    if debug:
        print(">>> FINAL SQL:", sql_to_run)

    result = {"sql": sql_to_run, "valid": True, "executed": False, "result": None}

    if not execute:
        logger.info("Execution disabled, returning SQL only.")
        return result

    # 6) Execute + self-correct
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
            logger.error("Execution failed (attempt %d / %d): %s", attempts, max_correction_attempts, err_msg)
            if attempts > max_correction_attempts:
                result["error"] = err_msg
                return result

            try:
                corrected_sql = self_correct_sql(sql_to_run, err_msg, question, schema_contexts)
                if not corrected_sql or not validate_sql(corrected_sql):
                    result["error"] = "Corrected SQL invalid."
                    return result
                sql_to_run = ensure_limit_clause(corrected_sql, EXECUTE_MAX_ROWS)
                if debug:
                    print(">>> RETRY with corrected SQL:", sql_to_run)
            except Exception as inner:
                logger.error("Self-correction process failed: %s", inner)
                result["error"] = str(inner)
                return result

    return result

# --- CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Reasoner: generate and optionally execute SQL from NL.")
    parser.add_argument("--no-exec", action="store_true", help="Do not execute SQL; return SQL only.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k schema retrieval.")
    parser.add_argument("--attempts", type=int, default=1, help="Self-correction attempts on failure.")
    parser.add_argument("--debug", action="store_true", help="Print prompt/raw SQL/fallbacks.")
    parser.add_argument("--ollama", action="store_true", help="Enable Ollama fallback explicitly for this run.")
    args = parser.parse_args()

    q = input("Enter your question: ").strip()
    res = infer_and_execute(
        q,
        execute=(not args.no_exec),
        top_k=args.top_k,
        max_correction_attempts=args.attempts,
        debug=args.debug,
        use_ollama_fallback=args.ollama,
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
