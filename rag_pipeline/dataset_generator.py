"""
dataset_generator.py (v2)
------------------------------------
Generate high-quality NL→SQL pairs from a live PostgreSQL DB.

Key features
- Introspects schema + FKs (joins)
- Samples real values (distinct text, numeric/date stats)
- Rich templates: filter/agg/group/order/limit/join/in/between/like/count-distinct
- Balanced coverage per table/column
- Read-only SQL validation + dedup
- Emits:  {"input": "Schema: ...\\nQuestion: ...", "output": "SELECT ...;"}

Run:
    python rag_pipeline/dataset_generator.py
"""

import os
import json
import math
import random
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

# Prefer using shared client/config if present
try:
    from clients import get_db_engine
    HAVE_CLIENTS = True
except Exception:
    HAVE_CLIENTS = False

try:
    from sql_validator import validate_sql
    HAVE_VALIDATOR = True
except Exception:
    HAVE_VALIDATOR = False

# ------------------------------ CONFIG ------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:parth%40123@localhost:5432/mytable"
)

OUTPUT_PATH = "data/fine_tune_dataset.jsonl"
os.makedirs("data", exist_ok=True)

# Target pair counts (scale up on high-end PCs)
TARGET_PER_TABLE = 500        # try 500–2000 if you have the data
MAX_DISTINCT_SAMPLES = 80     # per text column
NUMERIC_BUCKETS = [0.15, 0.5, 0.85]  # quantiles for thresholds
DATE_BUCKETS = [0.25, 0.5, 0.75]
MAX_IN_LIST = 5               # size for IN(...)
JOIN_RATE = 0.35              # ~35% of samples may include joins (if FKs exist)
COMPLEX_FILTER_RATE = 0.35    # % with 2+ WHERE conditions
LIKE_RATE = 0.30              # % of text filters that use LIKE

# Keep IDs unquoted for training consistency; quote only string literals.
# -------------------------------------------------------------------

# ------------------------------ UTILS -------------------------------
def sql_quote_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"

def is_text_col(typ: str) -> bool:
    t = (typ or "").lower()
    return any(k in t for k in ["char", "text", "varchar", "json", "uuid"])

def is_numeric_col(typ: str) -> bool:
    t = (typ or "").lower()
    return any(k in t for k in ["int", "float", "double", "numeric", "real", "decimal", "bigint", "smallint"])

def is_date_col(typ: str) -> bool:
    t = (typ or "").lower()
    return any(k in t for k in ["date", "timestamp", "time"])

def looks_like_sql(s: str) -> bool:
    s0 = (s or "").strip().upper()
    return s0.startswith("SELECT") or s0.startswith("WITH")

def ensure_select(sql: str) -> bool:
    if HAVE_VALIDATOR:
        try:
            return validate_sql(sql)
        except Exception:
            pass
    return looks_like_sql(sql)

def build_schema_str(inspector, tables: List[str]) -> str:
    parts = []
    for t in tables:
        try:
            cols = inspector.get_columns(t)
            col_names = [c["name"] for c in cols]
            parts.append(f"Table {t}({', '.join(col_names)})")
        except Exception:
            continue
    return " ".join(parts)

def run_scalar(engine, sql: str) -> Optional[Any]:
    try:
        with engine.connect() as conn:
            r = conn.execute(text(sql))
            row = r.fetchone()
            return None if row is None else row[0]
    except Exception:
        return None

def run_values(engine, sql: str, limit: int) -> List[Any]:
    try:
        with engine.connect() as conn:
            r = conn.execute(text(sql))
            vals = []
            for i, row in enumerate(r):
                vals.append(row[0])
                if i + 1 >= limit:
                    break
            return vals
    except Exception:
        return []

# -------------------------- INTROSPECTION ---------------------------
def get_engine():
    if HAVE_CLIENTS:
        logging.info("Using clients.get_db_engine()")
        return get_db_engine()
    logging.info("Using DATABASE_URL env")
    return create_engine(DATABASE_URL)

def get_schema_info(engine) -> Dict[str, Any]:
    insp = inspect(engine)
    info = {}
    for table in insp.get_table_names():
        cols = insp.get_columns(table)
        fks = insp.get_foreign_keys(table)
        info[table] = {
            "columns": [{"name": c["name"], "type": str(c["type"]).lower()} for c in cols],
            "fks": [
                {
                    "constrained": fk["constrained_columns"][0],
                    "referred_table": fk["referred_table"],
                    "referred_column": fk["referred_columns"][0],
                }
                for fk in fks if fk.get("referred_table")
            ],
            "row_count": run_scalar(engine, f"SELECT COUNT(*) FROM {table};") or 0,
        }
    return info

def sample_text_values(engine, table: str, col: str) -> List[str]:
    sql = f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL LIMIT {MAX_DISTINCT_SAMPLES};"
    vals = run_values(engine, sql, MAX_DISTINCT_SAMPLES)
    # stringify and filter empties
    vals = [str(v) for v in vals if v is not None and str(v).strip() != ""]
    return vals

def sample_numeric_stats(engine, table: str, col: str) -> Dict[str, Optional[float]]:
    # try rich stats (percentiles); fallback to min/max/avg only
    stats = {"min": None, "p15": None, "median": None, "p85": None, "max": None, "avg": None}
    try:
        sql = text(f"""
        SELECT
            MIN({col}) AS minv,
            MAX({col}) AS maxv,
            AVG({col}) AS avgv,
            percentile_cont(0.15) WITHIN GROUP (ORDER BY {col}) AS p15,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY {col}) AS p50,
            percentile_cont(0.85) WITHIN GROUP (ORDER BY {col}) AS p85
        FROM {table}
        WHERE {col} IS NOT NULL;
        """)
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()
            if row:
                stats["min"], stats["max"], stats["avg"], stats["p15"], stats["median"], stats["p85"] = row
                return stats
    except Exception:
        pass

    # fallback
    stats["min"] = run_scalar(engine, f"SELECT MIN({col}) FROM {table} WHERE {col} IS NOT NULL;")
    stats["max"] = run_scalar(engine, f"SELECT MAX({col}) FROM {table} WHERE {col} IS NOT NULL;")
    stats["avg"] = run_scalar(engine, f"SELECT AVG({col}) FROM {table} WHERE {col} IS NOT NULL;")
    if stats["min"] is not None and stats["max"] is not None:
        try:
            lo, hi = float(stats["min"]), float(stats["max"])
            stats["median"] = (lo + hi) / 2.0
            stats["p15"] = lo + 0.15 * (hi - lo)
            stats["p85"] = lo + 0.85 * (hi - lo)
        except Exception:
            pass
    return stats

def sample_date_stats(engine, table: str, col: str) -> Dict[str, Optional[str]]:
    # ISO strings
    out = {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    try:
        sql = text(f"""
        SELECT
            MIN({col})::timestamp,
            MAX({col})::timestamp,
            percentile_cont(0.25) WITHIN GROUP (ORDER BY {col})::timestamp,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY {col})::timestamp,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY {col})::timestamp
        FROM {table}
        WHERE {col} IS NOT NULL;
        """)
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()
            if row:
                mn, mx, p25, p50, p75 = row
                to_iso = lambda x: x.isoformat(sep=" ") if x else None
                out.update({"min": to_iso(mn), "max": to_iso(mx), "p25": to_iso(p25), "median": to_iso(p50), "p75": to_iso(p75)})
                return out
    except Exception:
        pass

    # fallback
    try:
        mn = run_scalar(engine, f"SELECT MIN({col}) FROM {table} WHERE {col} IS NOT NULL;")
        mx = run_scalar(engine, f"SELECT MAX({col}) FROM {table} WHERE {col} IS NOT NULL;")
        to_iso = lambda x: x.isoformat(sep=" ") if hasattr(x, "isoformat") else str(x) if x else None
        out["min"], out["max"] = to_iso(mn), to_iso(mx)
    except Exception:
        pass
    return out

# --------------------------- TEMPLATE BANK ---------------------------
# Natural language variants (keep short & varied)
BASIC_TEMPL = [
    ("List all records from {t}.", "SELECT * FROM {t};"),
    ("Show everything in {t}.", "SELECT * FROM {t};"),
    ("Return all rows from {t}.", "SELECT * FROM {t};"),
]

SELECT_ONE_TEMPL = [
    ("Show {c} from {t}.", "SELECT {c} FROM {t};"),
    ("List all values of {c} in {t}.", "SELECT {c} FROM {t};"),
    ("Give me the column {c} from {t}.", "SELECT {c} FROM {t};"),
]

ORDER_TEMPL = [
    ("List {c} from {t} ordered by {c} descending.", "SELECT {c} FROM {t} ORDER BY {c} DESC;"),
    ("Show top {n} {c} from {t} by descending {c}.", "SELECT {c} FROM {t} ORDER BY {c} DESC LIMIT {n};"),
]

COUNT_TEMPL = [
    ("How many records are in {t}?", "SELECT COUNT(*) FROM {t};"),
    ("Count rows in {t}.", "SELECT COUNT(*) FROM {t};"),
]

COUNT_DISTINCT_TEMPL = [
    ("How many unique {c} are in {t}?", "SELECT COUNT(DISTINCT {c}) FROM {t};"),
    ("Count distinct values of {c} in {t}.", "SELECT COUNT(DISTINCT {c}) FROM {t};"),
]

AGG_TEMPL = [
    ("What is the average {c} in {t}?", "SELECT AVG({c}) FROM {t};"),
    ("Find the maximum {c} in {t}.", "SELECT MAX({c}) FROM {t};"),
    ("Find the minimum {c} in {t}.", "SELECT MIN({c}) FROM {t};"),
    ("What is the sum of {c} in {t}?", "SELECT SUM({c}) FROM {t};"),
]

GROUP_TEMPL = [
    ("Show counts in {t} grouped by {c}.", "SELECT {c}, COUNT(*) FROM {t} GROUP BY {c};"),
    ("Group rows in {t} by {c} and count.", "SELECT {c}, COUNT(*) FROM {t} GROUP BY {c};"),
]

TEXT_EQ_TEMPL = [
    ("List all {t} where {c} equals {v}.", "SELECT * FROM {t} WHERE {c} = {v};"),
    ("Show rows in {t} with {c} = {v}.", "SELECT * FROM {t} WHERE {c} = {v};"),
    ("Find rows in {t} where {c} is {v}.", "SELECT * FROM {t} WHERE {c} = {v};"),
]

TEXT_LIKE_TEMPL = [
    ("Show all {t} where {c} contains {s}.", "SELECT * FROM {t} WHERE {c} LIKE {pat};"),
    ("Find rows in {t} with {c} like {s}.", "SELECT * FROM {t} WHERE {c} LIKE {pat};"),
]

IN_LIST_TEMPL = [
    ("List rows in {t} where {c} is in {vals}.", "SELECT * FROM {t} WHERE {c} IN ({vals});"),
]

NUM_CMP_TEMPL = [
    ("List all {t} where {c} is greater than {x}.", "SELECT * FROM {t} WHERE {c} > {x};"),
    ("Show {t} where {c} is less than {x}.", "SELECT * FROM {t} WHERE {c} < {x};"),
    ("Show rows in {t} where {c} is at least {x}.", "SELECT * FROM {t} WHERE {c} >= {x};"),
]

BETWEEN_TEMPL = [
    ("Show {t} where {c} is between {a} and {b}.", "SELECT * FROM {t} WHERE {c} BETWEEN {a} AND {b};"),
]

DATE_CMP_TEMPL = [
    ("List all {t} after {d}.", "SELECT * FROM {t} WHERE {c} > {d};"),
    ("List all {t} before {d}.", "SELECT * FROM {t} WHERE {c} < {d};"),
]

JOIN_TEMPL = [
    ("List rows joining {t1} with {t2}.", "SELECT * FROM {t1} JOIN {t2} ON {t1}.{k1} = {t2}.{k2};"),
    ("Count records after joining {t1} and {t2}.", "SELECT COUNT(*) FROM {t1} JOIN {t2} ON {t1}.{k1} = {t2}.{k2};"),
]

# ----------------------- EXAMPLE GENERATION -------------------------
def make_schema_input(inspector, tables: List[str]) -> str:
    return f"Schema: {build_schema_str(inspector, tables)}"

def add_example(out: List[Dict[str, str]], seen: set, schema_str: str, nl: str, sql: str):
    sql = sql.strip()
    if not sql.endswith(";"):
        sql += ";"
    key = (nl.strip(), sql)
    if key in seen:
        return
    if not ensure_select(sql):
        return
    seen.add(key)
    out.append({"input": f"{schema_str}\nQuestion: {nl}", "output": sql})

def choose(vals: List, k: int) -> List:
    if not vals:
        return []
    if len(vals) <= k:
        return list(vals)
    return random.sample(vals, k)

def value_tokens(v: str) -> List[str]:
    toks = [t for t in str(v).replace("_", " ").split() if t]
    return toks

def gen_for_table(engine, inspector, t: str, meta: Dict[str, Any], target: int, out: List[Dict], seen: set):
    cols = meta["columns"]
    row_count = meta["row_count"] or 0
    text_cols = [c for c in cols if is_text_col(c["type"])]
    num_cols  = [c for c in cols if is_numeric_col(c["type"])]
    date_cols = [c for c in cols if is_date_col(c["type"])]
    fks       = meta["fks"]

    schema_str = make_schema_input(inspector, [t])

    # Baselines always
    for nl, sql in BASIC_TEMPL:
        add_example(out, seen, schema_str, nl.format(t=t), sql.format(t=t))

    # Select individual columns
    for c in choose(cols, min(5, len(cols))):
        for nl, sql in SELECT_ONE_TEMPL:
            add_example(out, seen, schema_str, nl.format(t=t, c=c["name"]), sql.format(t=t, c=c["name"]))

    # ORDER / LIMIT
    for c in choose(num_cols, min(3, len(num_cols))):
        n = random.choice([5, 10, 20])
        for nl, sql in ORDER_TEMPL:
            add_example(out, seen, schema_str, nl.format(t=t, c=c["name"], n=n), sql.format(t=t, c=c["name"], n=n))

    # COUNT + COUNT DISTINCT
    for nl, sql in COUNT_TEMPL:
        add_example(out, seen, schema_str, nl.format(t=t), sql.format(t=t))
    for c in choose(cols, min(3, len(cols))):
        for nl, sql in COUNT_DISTINCT_TEMPL:
            add_example(out, seen, schema_str, nl.format(t=t, c=c["name"]), sql.format(t=t, c=c["name"]))

    # TEXT filters (use real values)
    for c in choose(text_cols, min(5, len(text_cols))):
        vals = sample_text_values(engine, t, c["name"])
        vals = choose(vals, min(MAX_DISTINCT_SAMPLES, 12))
        if not vals:
            continue

        # equality
        for v in choose(vals, min(6, len(vals))):
            vq = sql_quote_str(v)
            for nl, sql in TEXT_EQ_TEMPL:
                add_example(out, seen, schema_str, nl.format(t=t, c=c["name"], v=vq), sql.format(t=t, c=c["name"], v=vq))

        # IN list
        if len(vals) >= 3:
            sub = choose(vals, min(MAX_IN_LIST, len(vals)))
            in_list = ", ".join(sql_quote_str(v) for v in sub)
            for nl, sql in IN_LIST_TEMPL:
                add_example(out, seen, schema_str, nl.format(t=t, c=c["name"], vals=in_list),
                            sql.format(t=t, c=c["name"], vals=in_list))

        # LIKE
        if random.random() < LIKE_RATE:
            v = random.choice(vals)
            toks = value_tokens(v)
            if toks:
                token = random.choice(toks)
                pat = sql_quote_str(f"%{token}%")
                sdesc = sql_quote_str(token)
                for nl, sql in TEXT_LIKE_TEMPL:
                    add_example(out, seen, schema_str, nl.format(t=t, c=c["name"], s=sdesc, pat=pat),
                                sql.format(t=t, c=c["name"], pat=pat))

    # NUMERIC filters (use stats)
    for c in choose(num_cols, min(6, len(num_cols))):
        stats = sample_numeric_stats(engine, t, c["name"])
        # comparisons
        for x in [stats.get("p15"), stats.get("median"), stats.get("p85"), stats.get("avg")]:
            if x is None:
                continue
            val = str(float(x)).rstrip("0").rstrip(".")
            for nl, sql in NUM_CMP_TEMPL:
                add_example(out, seen, schema_str, nl.format(t=t, c=c["name"], x=val),
                            sql.format(t=t, c=c["name"], x=val))
        # between
        if stats.get("min") is not None and stats.get("max") is not None:
            lo = stats["min"]; hi = stats["max"]
            try:
                lo, hi = float(lo), float(hi)
                if lo < hi:
                    a = lo + 0.25*(hi-lo)
                    b = lo + 0.65*(hi-lo)
                    a_s = str(round(a, 3)).rstrip("0").rstrip(".")
                    b_s = str(round(b, 3)).rstrip("0").rstrip(".")
                    for nl, sql in BETWEEN_TEMPL:
                        add_example(out, seen, schema_str, nl.format(t=t, c=c["name"], a=a_s, b=b_s),
                                    sql.format(t=t, c=c["name"], a=a_s, b=b_s))
            except Exception:
                pass

    # DATE filters
    for c in choose(date_cols, min(3, len(date_cols))):
        ds = sample_date_stats(engine, t, c["name"])
        for dkey in ["p25", "median", "p75", "min", "max"]:
            d = ds.get(dkey)
            if not d:
                continue
            for nl, sql in DATE_CMP_TEMPL:
                add_example(out, seen, schema_str, nl.format(t=t, c=c["name"], d=sql_quote_str(d)),
                            sql.format(t=t, c=c["name"], d=sql_quote_str(d)))

    # GROUP BY
    gb_cols = choose(cols, min(3, len(cols)))
    for c in gb_cols:
        for nl, sql in GROUP_TEMPL:
            add_example(out, seen, schema_str, nl.format(t=t, c=c["name"]), sql.format(t=t, c=c["name"]))

    # Optional multi-clause filters (mix text + numeric)
    if random.random() < COMPLEX_FILTER_RATE and (text_cols or num_cols):
        qparts = []
        if text_cols:
            c = random.choice(text_cols)
            vals = sample_text_values(engine, t, c["name"])
            if vals:
                v = random.choice(vals)
                qparts.append(f"{c['name']} = {sql_quote_str(v)}")
        if num_cols:
            c = random.choice(num_cols)
            st = sample_numeric_stats(engine, t, c["name"])
            thr = st.get("median") or st.get("avg") or st.get("p85")
            if thr is not None:
                val = str(float(thr)).rstrip("0").rstrip(".")
                op = random.choice([">", "<", ">=", "<="])
                qparts.append(f"{c['name']} {op} {val}")
        if qparts:
            where = " AND ".join(qparts)
            nl = f"List rows in {t} where " + " and ".join(
                [" and ".join(qparts).replace(" =", " equals ").replace(">=", " at least ").replace("<=", " at most ")]
            )
            sql = f"SELECT * FROM {t} WHERE {where};"
            add_example(out, seen, schema_str, nl, sql)

    # JOIN examples (FKs)
    if fks and random.random() < JOIN_RATE:
        fk = random.choice(fks)
        t2 = fk["referred_table"]
        k1 = fk["constrained"]; k2 = fk["referred_column"]
        schema_str2 = make_schema_input(inspector, [t, t2])
        for nl, sql in JOIN_TEMPL:
            add_example(out, seen, schema_str2,
                        nl.format(t1=t, t2=t2),
                        sql.format(t1=t, t2=t2, k1=k1, k2=k2))

    # cap to target (shuffle for variety)
    random.shuffle(out)

def generate(engine) -> List[Dict[str, str]]:
    insp = inspect(engine)
    info = get_schema_info(engine)
    tables = list(info.keys())
    if not tables:
        logging.warning("No tables found.")
        return []

    all_examples: List[Dict[str, str]] = []
    seen = set()

    logging.info("Found %d tables: %s", len(tables), ", ".join(tables))
    for t in tables:
        logging.info("Generating for table '%s' (rows=%s)", t, info[t]["row_count"])
        before = len(all_examples)
        gen_for_table(engine, insp, t, info[t], TARGET_PER_TABLE, all_examples, seen)
        made = len(all_examples) - before
        logging.info("  -> %d examples", made)

    # Soft cap to approx TARGET_PER_TABLE * #tables
    soft_max = TARGET_PER_TABLE * max(1, len(tables))
    if len(all_examples) > soft_max:
        logging.info("Sampling down to %d (from %d)", soft_max, len(all_examples))
        all_examples = random.sample(all_examples, soft_max)

    return all_examples

# ------------------------------ MAIN --------------------------------
def main():
    logging.info("Connecting to PostgreSQL…")
    try:
        engine = get_engine()
    except SQLAlchemyError as e:
        logging.error("DB engine init failed: %s", e)
        return

    examples = generate(engine)
    if not examples:
        logging.warning("No examples generated.")
        return

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logging.info("✅ Wrote %d pairs to %s", len(examples), OUTPUT_PATH)
    logging.info("Example:\n%s", json.dumps(examples[0], indent=2))

if __name__ == "__main__":
    main()
