# rag_pipeline/train_schema_attention.py
"""
Train the SchemaAttention model.

For each example:
  - question
  - candidate schema descriptions (some used in SQL, some not)
  - labels (1 = relevant table, 0 = not used)

Output:
  - models/schema_attention.pt
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import inspect

from schema_attention import SchemaAttention
from build_schema import table_to_description
from clients import get_db_engine
from config import EMBED_MODEL

DATA_PATH = Path("data/fine_tune_dataset.jsonl")
TABLE_REGEX = re.compile(
    r"\bfrom\s+([a-zA-Z_][\w]*)|\bjoin\s+([a-zA-Z_][\w]*)",
    re.IGNORECASE,
)


def extract_question(full_input: str) -> str:
    parts = full_input.split("Question:", 1)
    return parts[1].strip() if len(parts) == 2 else full_input.strip()


def extract_tables(sql: str) -> List[str]:
    tables = set()
    for m in TABLE_REGEX.finditer(sql):
        t1, t2 = m.groups()
        t = (t1 or t2)
        if t:
            tables.add(t.lower())
    return list(tables)


def load_all_table_descriptions() -> Dict[str, str]:
    engine = get_db_engine()
    inspector = inspect(engine)
    table_desc = {}
    for t in inspector.get_table_names():
        table_desc[t.lower()] = table_to_description(t, inspector)
    return table_desc


class SchemaAttentionDataset(Dataset):
    """
    Each item:
      {
        "question": str,
        "schema_texts": List[str],
        "labels": List[int]
      }
    """

    def __init__(self, data_path: Path, table_desc: Dict[str, str], max_tables_per_example: int = 8):
        self.examples = []
        all_tables = list(table_desc.keys())

        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                question = extract_question(row["input"])
                sql = row["output"]

                used_tables = [t for t in extract_tables(sql) if t in table_desc]
                if not used_tables:
                    continue

                negatives = [t for t in all_tables if t not in used_tables]
                if not negatives:
                    continue

                # Roughly half positives, half negatives
                num_pos = min(len(used_tables), max_tables_per_example // 2)
                num_neg = max_tables_per_example - num_pos

                pos_tables = random.sample(used_tables, num_pos)
                neg_tables = random.sample(negatives, min(num_neg, len(negatives)))

                cand_tables = pos_tables + neg_tables
                labels = [1] * len(pos_tables) + [0] * len(neg_tables)

                schema_texts = [table_desc[t] for t in cand_tables]
                self.examples.append(
                    {
                        "question": question,
                        "schema_texts": schema_texts,
                        "labels": labels,
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    # Keep variable length; the model's forward loops over examples
    return batch


def train_schema_attention():
    print("Loading table descriptions...")
    table_desc = load_all_table_descriptions()

    print("Building attention training dataset...")
    dataset = SchemaAttentionDataset(DATA_PATH, table_desc)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print(f"Total attention training examples: {len(dataset)}")
    if len(dataset) == 0:
        print("No examples found for schema attention training. Aborting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchemaAttention(embed_model_name=EMBED_MODEL).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 3

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            batch_loss = 0.0

            # Each batch element has its own variable number of schema_texts
            for ex in batch:
                q = ex["question"]
                schema_texts = ex["schema_texts"]
                labels = torch.tensor(ex["labels"], dtype=torch.float32, device=device)

                scores = model(q, schema_texts).to(device)  # [Ns]
                loss = criterion(scores, labels)
                batch_loss += loss

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - loss={avg_loss:.4f}")

    out_path = "models/schema_attention.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Schema attention model saved to {out_path}")


if __name__ == "__main__":
    train_schema_attention()
