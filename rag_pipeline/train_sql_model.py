# rag_pipeline/train_sql_model.py
"""
NL→SQL fine-tuning for FLAN-T5 with in-code config.
- Uses fast tokenizer
- Instruction prefix to nudge SELECT/WITH outputs
- Auto GPU precision (bf16 on Ampere+, else fp16 if available)
- Optional tiny eval split + exact-match metric
- Gradient checkpointing / torch.compile toggles
- Saves ONLY the final model
Run:
    python rag_pipeline/train_sql_model.py
"""

import os
import logging
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

CONFIG = {
    "model": "google/flan-t5-base",
    "data": "data/fine_tune_dataset.jsonl",
    "output_dir": "models/flan_t5_sql",

    "epochs": 12.0,
    "max_steps": 0,               # >0 overrides epochs

    "batch_size": None,           # None = auto from VRAM
    "grad_accum": 2,
    "lr": 3e-5,
    "seed": 42,

    "max_src_len": 256,
    "max_tgt_len": 128,

    # Helps the model start with valid SQL
    "instruction_prefix": "Return only a single valid PostgreSQL query starting with SELECT or WITH.\n",

    "eval_ratio": 0.1,            # set 0.0 to disable eval split
    "freeze_encoder": False,

    "gradient_checkpointing": True,
    "torch_compile": True,
}

def auto_precision():
    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    major, _ = torch.cuda.get_device_capability(0)
    return {"fp16": False, "bf16": True} if major >= 8 else {"fp16": True, "bf16": False}

def auto_batch_size(model_name: str) -> int:
    if not torch.cuda.is_available():
        return 4
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    m = model_name.lower()
    if "small" in m:  return 32 if vram_gb >= 24 else 16
    if "base"  in m:  return 16 if vram_gb >= 24 else 8
    if "large" in m:  return 8  if vram_gb >= 24 else 4
    return 8

def normalize_sql(s):
    return " ".join((s or "").strip().rstrip(";").split()).lower()

def compute_metrics_builder(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = [[t if t != -100 else tokenizer.pad_token_id for t in seq] for seq in labels]
        p = tokenizer.batch_decode(preds, skip_special_tokens=True)
        g = tokenizer.batch_decode(labels, skip_special_tokens=True)
        exact = sum(1 for x, y in zip(p, g) if normalize_sql(x) == normalize_sql(y))
        return {"exact_match": exact / max(1, len(p))}
    return compute_metrics

def prepend_prefix(ds, prefix):
    if not prefix:
        return ds
    def _map(ex):
        ex["input"] = prefix + ex["input"]
        return ex
    return ds.map(_map)

def main():
    cfg = CONFIG.copy()
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    logging.info("=== train_sql_model.py ===")
    logging.info(f"Model: {cfg['model']}")
    logging.info(f"Data:  {cfg['data']}")
    logging.info(f"Out:   {cfg['output_dir']}")

    # 1) Load data
    ds = load_dataset("json", data_files=cfg["data"], split="train")
    if len(ds) == 0 or "input" not in ds.column_names or "output" not in ds.column_names:
        raise ValueError("Dataset must contain non-empty 'input' and 'output' fields.")
    logging.info(f"[STEP 1] Loaded {len(ds)} examples.")

    # 2) Split
    if cfg["eval_ratio"] > 0:
        idx = list(range(len(ds)))
        random.shuffle(idx)
        cut = int(len(ds) * (1 - cfg["eval_ratio"]))
        ds_train = ds.select(idx[:cut])
        ds_eval  = ds.select(idx[cut:])
        logging.info(f"[STEP 1] Split -> train={len(ds_train)} eval={len(ds_eval)}")
    else:
        ds_train, ds_eval = ds, None

    # 3) Prefix
    ds_train = prepend_prefix(ds_train, cfg["instruction_prefix"])
    if ds_eval is not None:
        ds_eval = prepend_prefix(ds_eval, cfg["instruction_prefix"])

    # 4) Tokenizer & model
    logging.info(f"[STEP 2] Loading tokenizer/model: {cfg['model']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(cfg["model"])

    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if cfg["torch_compile"] and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            logging.info("[STEP 2] torch.compile enabled.")
        except Exception as e:
            logging.warning(f"[STEP 2] torch.compile skipped: {e}")

    if cfg["freeze_encoder"]:
        logging.info("[STEP 2] Freezing encoder params.")
        for n, p in model.named_parameters():
            if n.startswith("encoder."):
                p.requires_grad = False

    # 5) Tokenization
    def preprocess(examples):
        enc = tokenizer(
            examples["input"],
            max_length=cfg["max_src_len"],
            truncation=True,
            padding="max_length",
        )
        dec = tokenizer(
            text_target=examples["output"],
            max_length=cfg["max_tgt_len"],
            truncation=True,
            padding="max_length",
        )
        enc["labels"] = dec["input_ids"]
        return enc

    logging.info("[STEP 3] Tokenizing…")
    tok_train = ds_train.map(preprocess, batched=True, remove_columns=ds.column_names)
    tok_eval  = ds_eval.map(preprocess,  batched=True, remove_columns=ds.column_names) if ds_eval else None
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 6) Training args (robust to Transformers versions)
    prec = auto_precision()
    bsz = cfg["batch_size"] or auto_batch_size(cfg["model"])
    use_cuda = torch.cuda.is_available()
    max_steps = cfg["max_steps"] if cfg["max_steps"] and cfg["max_steps"] > 0 else -1
    num_epochs = 1.0 if max_steps > 0 else cfg["epochs"]

    eval_strategy = "steps" if tok_eval is not None else "no"
    logging.info(f"[STEP 4] GPU: {use_cuda}  bf16={prec['bf16']}  fp16={prec['fp16']}  batch_size={bsz}")

    # Try new arg; fall back if running an older Transformers
    common_kwargs = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        weight_decay=0.01,
        evaluation_strategy=eval_strategy,
        eval_steps=200 if eval_strategy == "steps" else None,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=cfg["seed"],
        dataloader_pin_memory=use_cuda,
        fp16=prec["fp16"],
        bf16=prec["bf16"],
        remove_unused_columns=True,
    )

    try:
        training_args = TrainingArguments(**common_kwargs, predict_with_generate=(tok_eval is not None))
    except TypeError:
        # Older Transformers: no 'predict_with_generate'
        training_args = TrainingArguments(**common_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(compute_metrics_builder(tokenizer) if tok_eval is not None else None),
    )

    # 7) Train
    logging.info("[STEP 5] Training…")
    trainer.train()

    # 8) Save final
    logging.info(f"[STEP 6] Saving to {cfg['output_dir']} …")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    logging.info("✅ Done.")

if __name__ == "__main__":
    main()
