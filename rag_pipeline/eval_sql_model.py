# rag_pipeline/eval_sql_model.py
"""
Evaluation & plots for NL→SQL model.

Metrics:
- Exact match (normalized)
- StartsWith SELECT/WITH
- Valid SQL (uses sql_validator if available, else heuristic)
- Char edit distance & token F1 (basic)
Artifacts:
- CSV with per-example results
- Plots (PNG): rates bar, edit-distance hist, length hist, scatter, error-type bar
"""

import os
import re
import csv
import json
import math
import argparse
from collections import Counter
from typing import List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Optional config import
try:
    from config import INSTRUCTION_PREFIX
except Exception:
    INSTRUCTION_PREFIX = "Return only a single valid PostgreSQL query starting with SELECT or WITH.\n"

# Optional validator
try:
    from sql_validator import validate_sql as _validate_sql
    def validate_sql(s: str) -> bool:
        try:
            return _validate_sql(s)
        except Exception:
            return False
except Exception:
    def validate_sql(s: str) -> bool:
        s0 = (s or "").lstrip().upper()
        return s0.startswith("SELECT") or s0.startswith("WITH")

def normalize_sql(s: str) -> str:
    return " ".join((s or "").strip().rstrip(";").split()).lower()

def starts_like_sql(s: str) -> bool:
    s0 = (s or "").lstrip().upper()
    return s0.startswith("SELECT") or s0.startswith("WITH")

def char_edit_distance(a: str, b: str) -> int:
    # simple DP Levenshtein
    a, b = a or "", b or ""
    na, nb = len(a), len(b)
    dp = list(range(nb + 1))
    for i in range(1, na + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, nb + 1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[nb]

def token_f1(pred: str, gold: str) -> float:
    pt = pred.split()
    gt = gold.split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    pc, gc = Counter(pt), Counter(gt)
    overlap = sum((pc & gc).values())
    prec = overlap / max(1, len(pt))
    rec  = overlap / max(1, len(gt))
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def classify_error(pred: str) -> str:
    s = (pred or "").strip()
    S = s.upper()
    if not s:
        return "empty"
    if not starts_like_sql(s):
        return "no_select_with"
    if " FROM " not in S:
        return "missing_FROM"
    if "WHERE" in S and re.search(r"WHERE\s*$", S):
        return "dangling_WHERE"
    if S.count("(") != S.count(")"):
        return "paren_mismatch"
    return "other"

def plot_and_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to fine-tuned model dir")
    ap.add_argument("--data", required=True, help="Path to JSONL with fields: input, output")
    ap.add_argument("--save_dir", required=True, help="Directory to save CSV/plots")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Data
    ds = load_dataset("json", data_files=args.data, split="train")
    if len(ds) == 0:
        print("No examples found.")
        return
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # 2) Model
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 3) Predict
    results = []
    for i in range(0, len(ds), args.batch_size):
        batch = ds[i:i+args.batch_size]
        inputs = [INSTRUCTION_PREFIX + ex["input"] for ex in batch]
        golds  = [ex["output"] for ex in batch]

        enc = tok(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        preds = tok.batch_decode(gen, skip_special_tokens=True)

        for x, y, p in zip(inputs, golds, preds):
            nx, ny = normalize_sql(p), normalize_sql(y)
            res = {
                "input": x,
                "gold": y,
                "pred": p,
                "exact_match": float(nx == ny),
                "starts_with_sql": float(starts_like_sql(p)),
                "valid_sql": float(validate_sql(p)),
                "edit_distance": char_edit_distance(p, y),
                "token_f1": token_f1(p, y),
                "len_pred": len(p),
                "len_gold": len(y),
                "error_type": "" if nx == ny else classify_error(p),
            }
            results.append(res)

    # 4) Save CSV
    csv_path = os.path.join(args.save_dir, "per_example_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # 5) Aggregate
    exact = np.mean([r["exact_match"] for r in results])
    starts = np.mean([r["starts_with_sql"] for r in results])
    valid  = np.mean([r["valid_sql"] for r in results])
    edists = np.array([r["edit_distance"] for r in results], dtype=float)
    f1s    = np.array([r["token_f1"] for r in results], dtype=float)
    lg     = np.array([r["len_gold"] for r in results], dtype=float)
    lp     = np.array([r["len_pred"] for r in results], dtype=float)
    errs   = Counter([r["error_type"] for r in results if r["error_type"]])

    summary = {
        "n": len(results),
        "exact_match": exact,
        "starts_with_sql": starts,
        "valid_sql": valid,
        "edit_distance_mean": float(edists.mean()),
        "edit_distance_median": float(np.median(edists)),
        "token_f1_mean": float(f1s.mean()),
    }
    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 6) Plots
    # (a) Rates bar
    fig = plt.figure(figsize=(6,4))
    xs  = ["Exact", "StartsWith", "ValidSQL"]
    ys  = [exact, starts, valid]
    plt.bar(xs, ys)
    plt.ylim(0, 1)
    plt.title("Core Rates")
    plot_and_save(fig, os.path.join(args.save_dir, "01_core_rates.png"))

    # (b) Edit distance hist
    fig = plt.figure(figsize=(6,4))
    plt.hist(edists, bins=20)
    plt.title("Edit Distance (char)")
    plt.xlabel("Levenshtein distance")
    plt.ylabel("#Examples")
    plot_and_save(fig, os.path.join(args.save_dir, "02_edit_distance_hist.png"))

    # (c) Token F1 hist
    fig = plt.figure(figsize=(6,4))
    plt.hist(f1s, bins=20)
    plt.title("Token F1")
    plt.xlabel("F1")
    plt.ylabel("#Examples")
    plot_and_save(fig, os.path.join(args.save_dir, "03_token_f1_hist.png"))

    # (d) Length histograms
    fig = plt.figure(figsize=(6,4))
    plt.hist(lg, bins=20, alpha=0.5, label="gold")
    plt.hist(lp, bins=20, alpha=0.5, label="pred")
    plt.title("Length Distributions")
    plt.xlabel("Characters")
    plt.ylabel("#Examples")
    plt.legend()
    plot_and_save(fig, os.path.join(args.save_dir, "04_length_hist.png"))

    # (e) Pred vs gold scatter
    fig = plt.figure(figsize=(5,5))
    plt.scatter(lg, lp, s=12)
    lim = max(lg.max() if len(lg)>0 else 0, lp.max() if len(lp)>0 else 0) + 5
    plt.plot([0, lim], [0, lim])
    plt.xlabel("Gold length")
    plt.ylabel("Pred length")
    plt.title("Pred vs. Gold Length")
    plot_and_save(fig, os.path.join(args.save_dir, "05_len_scatter.png"))

    # (f) Error-type bar
    if errs:
        fig = plt.figure(figsize=(7,4))
        items = sorted(errs.items(), key=lambda kv: kv[1], reverse=True)[:10]
        labels = [k for k,_ in items]
        values = [v for _,v in items]
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha="right")
        plt.title("Top Error Categories")
        plt.ylabel("#Examples")
        plot_and_save(fig, os.path.join(args.save_dir, "06_error_types.png"))

    print(f"\nSaved:\n- {csv_path}\n- {os.path.join(args.save_dir, 'summary.json')}\n- plots_01..06")
    print("\nSummary:", json.dumps(summary, indent=2))
    print("Done.")

if __name__ == "__main__":
    main()
