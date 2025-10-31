"""
train_sql_model.py
------------------------------------
Fine-tunes a FLAN-T5 model on generated NL-SQL pairs
from dataset_generator.py for text-to-SQL translation.
------------------------------------
"""

import os
import json
import logging
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# ---------------------------------------------------------------------
# Logging and Config
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

MODEL_NAME = "google/flan-t5-base"
DATA_PATH = "data/fine_tune_dataset.jsonl"
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
    logging.info("Loading dataset from %s", DATA_PATH)
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    return dataset

# ---------------------------------------------------------------------
# Tokenization Function
# ---------------------------------------------------------------------
def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["input"],
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding="max_length",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"],
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
    dataset = load_train_dataset()

    logging.info("Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    logging.info("Preprocessing dataset...")
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    logging.info("Initializing trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="no",
        learning_rate=3e-5,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logging.info("Starting fine-tuning process...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info("Model fine-tuned and saved to '%s'.", OUTPUT_DIR)

# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    train_model()
