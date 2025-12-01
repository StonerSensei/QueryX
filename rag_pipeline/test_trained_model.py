from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL_PATH = "models/flan_t5_sql"
DATA_PATH = "data/fine_tune_dataset.jsonl"

print("Loading dataset...")
ds = load_dataset("json", data_files=DATA_PATH, split="train")

print("Loading model + tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

def generate(text):
    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    out = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tok.decode(out[0], skip_special_tokens=True).strip()

for i in range(5):
    ex = ds[i]
    inp = ex["input"]
    gold = ex["output"]
    pred = generate(inp)
    print("==== EXAMPLE", i, "====")
    print("INPUT:\n", inp)
    print("GOLD SQL:\n", gold)
    print("PRED SQL:\n", pred)
    print()
