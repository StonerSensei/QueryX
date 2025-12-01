from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "google/flan-t5-base"

print(">>> Import OK")

try:
    print(">>> Loading tokenizer...")
    tok = T5Tokenizer.from_pretrained(MODEL_NAME)
    print(">>> Tokenizer loaded OK")

    print(">>> Loading model...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    print(">>> Model loaded OK")
except Exception as e:
    print(">>> ERROR while loading model/tokenizer:")
    import traceback
    traceback.print_exc()
