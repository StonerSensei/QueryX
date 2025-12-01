"""
dataset_generator.py

Automatically generates natural language (NL) and SQL query pairs
from an existing PostgreSQL database schema. These pairs are used
to fine-tune a FLAN-T5 model for text-to-SQL generation.
"""

import os
import json
import random
import logging
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Configuration and Logging
# ---------------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:parth%40123@localhost:5432/mytable"
)

OUTPUT_PATH = "data/fine_tune_dataset.jsonl"
NUM_SAMPLES_PER_TABLE = 30  # adjust based on DB size
os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# ---------------------------------------------------------------------
# SQL Query Templates
# ---------------------------------------------------------------------
BASE_TEMPLATES = {
    "basic": [
        ("List all records from the {table} table.", "SELECT * FROM {table};"),
        ("Show all {col} values from the {table} table.", "SELECT {col} FROM {table};"),
        ("Give all unique {col} values from {table}.", "SELECT DISTINCT {col} FROM {table};")
    ],
    "aggregate": [
        ("Count total records in {table}.", "SELECT COUNT(*) FROM {table};"),
        ("What is the average {col} in {table}?", "SELECT AVG({col}) FROM {table};"),
        ("Find the maximum {col} in {table}.", "SELECT MAX({col}) FROM {table};")
    ],
    "filter_numeric": [
        ("List all {table} where {col} is greater than 100.", "SELECT * FROM {table} WHERE {col} > 100;"),
        ("Show {col} from {table} where {col} is less than 50.", "SELECT {col} FROM {table} WHERE {col} < 50;")
    ],
    "filter_text": [
        ("List all {table} where {col} equals 'example'.", "SELECT * FROM {table} WHERE {col} = 'example';"),
        ("Show all {table} where {col} contains 'test'.", "SELECT * FROM {table} WHERE {col} LIKE '%test%';")
    ],
    "filter_date": [
        ("List all {table} where {col} is after 2020-01-01.", "SELECT * FROM {table} WHERE {col} > '2020-01-01';"),
        ("Show all {table} where {col} is before 2022-01-01.", "SELECT * FROM {table} WHERE {col} < '2022-01-01';")
    ],
    "group_order": [
        ("Show record count in {table} grouped by {col}.", "SELECT {col}, COUNT(*) FROM {table} GROUP BY {col};"),
        ("List {col} from {table} ordered by {col} descending.", "SELECT {col} FROM {table} ORDER BY {col} DESC;")
    ],
}

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def get_schema_info(engine):
    """Extracts tables, columns, data types, and foreign keys from the database."""
    inspector = inspect(engine)
    schema_info = {}

    for table in inspector.get_table_names():
        columns = inspector.get_columns(table)
        fks = inspector.get_foreign_keys(table)

        schema_info[table] = {
            "columns": [{"name": c["name"], "type": str(c["type"]).lower()} for c in columns],
            "foreign_keys": [
                {
                    "constrained": fk["constrained_columns"][0],
                    "referred_table": fk["referred_table"],
                    "referred_column": fk["referred_columns"][0]
                }
                for fk in fks if fk.get("referred_table")
            ]
        }
    return schema_info


def choose_template(col_type):
    """Selects an appropriate NL-SQL template based on column type."""
    if any(k in col_type for k in ["int", "float", "numeric", "real", "double"]):
        return random.choice(BASE_TEMPLATES["filter_numeric"])
    elif any(k in col_type for k in ["char", "text", "varchar"]):
        return random.choice(BASE_TEMPLATES["filter_text"])
    elif "date" in col_type:
        return random.choice(BASE_TEMPLATES["filter_date"])
    else:
        return random.choice(BASE_TEMPLATES["basic"])


def generate_samples(schema_info):
    """Generates natural language and SQL query pairs."""
    samples = []
    tables = list(schema_info.keys())

    for table, meta in schema_info.items():
        cols = meta["columns"]
        fks = meta["foreign_keys"]

        for _ in range(NUM_SAMPLES_PER_TABLE):
            # Randomly select a column and a suitable template
            if not cols:
                continue

            col_meta = random.choice(cols)
            col_name = col_meta["name"]
            col_type = col_meta["type"]

            tpl = choose_template(col_type)
            nl = tpl[0].format(table=table, col=col_name)
            sql = tpl[1].format(table=table, col=col_name)

            # Optionally add joins if foreign keys exist
            if fks and random.random() < 0.3:  # 30% chance
                fk = random.choice(fks)
                join_table = fk["referred_table"]
                join_sql = (
                    f"SELECT * FROM {table} "
                    f"JOIN {join_table} ON {table}.{fk['constrained']} = {join_table}.{fk['referred_column']};"
                )
                join_nl = f"List all records joining {table} with {join_table}."
                samples.append({
                    "input": f"Schema: Table {table}({', '.join([c['name'] for c in cols])})\nQuestion: {join_nl}",
                    "output": join_sql
                })

            # Add non-join example
            schema_context = f"Schema: Table {table}({', '.join([c['name'] for c in cols])})"
            samples.append({
                "input": f"{schema_context}\nQuestion: {nl}",
                "output": sql
            })

    return samples


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
def main():
    logging.info("Connecting to PostgreSQL database...")
    try:
        engine = create_engine(DATABASE_URL)
        schema_info = get_schema_info(engine)
        logging.info("Extracted schema for %d tables.", len(schema_info))
    except SQLAlchemyError as e:
        logging.error("Database connection failed: %s", e)
        return

    logging.info("Generating dataset samples...")
    samples = generate_samples(schema_info)
    logging.info("Generated %d NL-SQL pairs.", len(samples))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logging.info("Dataset successfully saved to '%s'.", OUTPUT_PATH)
    logging.info("Example entry: %s", json.dumps(samples[0], indent=2))


if __name__ == "__main__":
    main()
