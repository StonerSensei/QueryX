"""
train_sql_model.py
------------------------------------
Fine-tunes a FLAN-T5 model on generated NL-SQL pairs
from dataset_generator.py for text-to-SQL translation.
Uses the fast tokenizer (T5TokenizerFast via AutoTokenizer)
to avoid sentencepiece issues on Windows.
------------------------------------
"""

import os
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# ---------------------------------------------------------------------
# Logging and Config
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Start with small model to make sure everything works
MODEL_NAME = "google/flan-t5-small"

DATA_PATH = "data/fine_tune_dataset.jsonl"   # relative to rag_pipeline folder

OUTPUT_DIR = "models/flan_t5_sql"
BATCH_SIZE = 4
EPOCHS = 5
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------
def load_train_dataset():
    logging.info("[STEP 1] Loading dataset from %s", DATA_PATH)
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    logging.info("[STEP 1] Dataset loaded with %d examples.", len(dataset))
    logging.info("[STEP 1] Sample keys: %s", list(dataset[0].keys()))
    return dataset


# ---------------------------------------------------------------------
# Tokenization Function
# ---------------------------------------------------------------------
def preprocess_function(examples, tokenizer):
    """
    Tokenize input (NL + schema) and output (SQL) for T5.
    """
    # Encoder input
    model_inputs = tokenizer(
        examples["input"],
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # Decoder target
    labels = tokenizer(
        text_target=examples["output"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ---------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------
def train_model():
    # 1. Load dataset
    dataset = load_train_dataset()

    # 2. Load fast tokenizer + model
    logging.info("[STEP 2] Loading fast tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    logging.info("[STEP 2] Fast tokenizer loaded: %s", type(tokenizer))
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    logging.info("[STEP 2] Model loaded OK.")

    # 3. Tokenize dataset
    logging.info("[STEP 3] Preprocessing dataset (tokenization)...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,  # keep only model inputs/labels
    )
    logging.info("[STEP 3] Tokenization complete. Example keys: %s", tokenized_dataset[0].keys())

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4. Training arguments
    logging.info("[STEP 4] Initializing Trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="no",         # no eval split for now
        learning_rate=3e-5,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        report_to="none",                # no wandb/tensorboard
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. Train + save
    logging.info("[STEP 5] Starting fine-tuning process...")
    trainer.train()

    logging.info("[STEP 6] Saving model and tokenizer to '%s'...", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info("✅ Model fine-tuned and saved to '%s'.", OUTPUT_DIR)


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("=== train_sql_model.py starting ===")
    logging.info("Using model: %s", MODEL_NAME)
    logging.info("Dataset path: %s", DATA_PATH)
    train_model()
