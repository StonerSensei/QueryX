from datasets import load_dataset

DATA_PATH = r"C:\Users\USER\Documents\QueryX\rag_pipeline\data\fine_tune_dataset.jsonl"
ds = load_dataset("json", data_files=DATA_PATH, split="train")
print(ds[0])
