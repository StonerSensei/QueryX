# rag_pipeline/train_schema_embeddings.py
"""
Train a custom schema-aware embedding model for QueryX.

Uses:
  - data/fine_tune_dataset.jsonl from dataset_generator.py
  - table descriptions from build_schema.table_to_description

Output:
  - models/queryx-embeddings/ (SentenceTransformer format)
"""

import json
import re
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sqlalchemy import inspect

from build_schema import table_to_description
from clients import get_db_engine
from config import EMBED_MODEL

DATA_PATH = Path("data/fine_tune_dataset.jsonl")
OUTPUT_MODEL_DIR = Path("models/queryx-embeddings")

TABLE_REGEX = re.compile(
    r"\bfrom\s+([a-zA-Z_][\w]*)|\bjoin\s+([a-zA-Z_][\w]*)",
    re.IGNORECASE,
)


def load_schema_descriptions() -> Dict[str, str]:
    """
    Introspect DB and build text description per table,
    reusing build_schema.table_to_description.
    """
    engine = get_db_engine()
    inspector = inspect(engine)

    table_desc = {}
    for table in inspector.get_table_names():
        desc = table_to_description(table, inspector)
        table_desc[table.lower()] = desc
    return table_desc


def extract_question(full_input: str) -> str:
    """
    From 'Schema: Table orders(...)\\nQuestion: ...' extract question text.
    """
    parts = full_input.split("Question:", 1)
    return parts[1].strip() if len(parts) == 2 else full_input.strip()


def extract_tables_from_sql(sql: str) -> List[str]:
    """
    Simple regex for FROM/JOIN table names in synthetic SQL.
    """
    tables = set()
    for m in TABLE_REGEX.finditer(sql):
        t1, t2 = m.groups()
        t = (t1 or t2)
        if t:
            tables.add(t.lower())
    return list(tables)


def build_training_examples(schema_desc: Dict[str, str]) -> List[InputExample]:
    """
    Build (question, table_description) pairs.
    MultipleNegativesRankingLoss will use other pairs in the batch as negatives.
    """
    examples: List[InputExample] = []
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            full_in = row["input"]
            sql = row["output"]

            question = extract_question(full_in)
            tables = extract_tables_from_sql(sql)
            if not tables:
                continue

            for t in tables:
                if t in schema_desc:
                    desc = schema_desc[t]
                    examples.append(InputExample(texts=[question, desc]))
    return examples


def main():
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading schema descriptions from DB...")
    schema_desc = load_schema_descriptions()

    print("Building training examples from NL-SQL dataset...")
    train_examples = build_training_examples(schema_desc)
    print(f"Total training pairs: {len(train_examples)}")
    if not train_examples:
        print("No training examples found; aborting.")
        return

    print(f"Loading base embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=32)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    num_epochs = 3
    warmup_steps = int(len(train_loader) * num_epochs * 0.1)

    print("Starting embedding fine-tuning...")
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=str(OUTPUT_MODEL_DIR),
        show_progress_bar=True,
    )

    print(f"Custom embeddings saved to {OUTPUT_MODEL_DIR}")
    print("Update config.EMBED_MODEL to this path and rerun build_schema.py for new indexing.")


if __name__ == "__main__":
    main()
