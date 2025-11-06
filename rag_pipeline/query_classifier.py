# rag_pipeline/query_classifier.py
"""
QueryX - Query Classifier
-------------------------
Classifies a natural-language (or raw SQL) input into an SQL operation:
  SELECT | INSERT | UPDATE | DELETE | CREATE | DROP | ALTER | TRUNCATE | UNKNOWN

Also returns:
  - risk:    low | medium | high | critical
  - confidence: float in [0,1]
  - role:    reader | writer | admin
  - rationale: short textual reason

No heavy dependencies. Safe to call synchronously before RAG/LLM.
"""

from __future__ import annotations
import re
from typing import Dict, Optional

# ---- Primary operations we recognize
OPS = ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "TRUNCATE")

# ---- Lightweight lexicons & patterns (extend as needed)
LEX = {
    "SELECT": [
        r"\b(show|list|get|fetch|display|find|count|how many|top \d+|latest|recent)\b",
        r"\bselect\b",  # raw SQL
        r"\bretrieve\b",
        r"\bgroup by\b|\border by\b|\bwhere\b",
    ],
    "INSERT": [
        r"\b(insert|add|create a new (row|record|entry)|append)\b",
        r"\binsert\s+into\b",
        r"\badd (a|an|the)?\s*(row|record|entry)\b",
    ],
    "UPDATE": [
        r"\b(update|modify|change|set .* to)\b",
        r"\bupdate\s+\w+\s+set\b",
        r"\bcorrect\b",
    ],
    "DELETE": [
        r"\b(delete|remove|erase)\b",
        r"\bdelete\s+from\b",
        r"\bpurge\b",
    ],
    "CREATE": [
        r"\b(create table|create schema|new table|define table)\b",
        r"\bcreate\b(?!\s*(view|materialized view)\b)",  # still CREATE but high risk
    ],
    "DROP": [
        r"\b(drop table|drop column|drop schema|drop database)\b",
        r"\bdrop\b",
    ],
    "ALTER": [
        r"\b(alter table|add column|drop column|rename column|rename table|change type)\b",
        r"\balter\b",
    ],
    "TRUNCATE": [
        r"\b(truncate table|truncate)\b",
    ],
}

# ---- Phrases that escalate risk if the intent suggests mass changes
RISK_ESCALATORS = [
    r"\b(all rows|everything|entire table|whole table|wipe|clear (the )?table)\b",
    r"\b(without\s+where\b|\bno\s+where\b)",
]

# ---- Quick SQL-head detection (raw SQL path)
SQL_HEAD = re.compile(r"^\s*(select|insert|update|delete|create|drop|alter|truncate)\b", re.I)


def required_role(operation: str) -> str:
    """Map operation to a DB role."""
    op = operation.upper()
    if op == "SELECT":
        return "reader"
    if op in ("INSERT", "UPDATE"):
        return "writer"
    if op in ("DELETE", "CREATE", "DROP", "ALTER", "TRUNCATE"):
        return "admin"
    return "reader"


def base_risk(operation: str) -> str:
    """Baseline risk by operation."""
    op = operation.upper()
    if op == "SELECT":
        return "low"
    if op in ("INSERT", "UPDATE", "CREATE"):
        return "medium"
    if op in ("DELETE", "DROP", "ALTER", "TRUNCATE"):
        return "high"
    return "low"


def escalate_risk(risk: str, text: str) -> str:
    """Raise risk to critical if mass-modification hints appear."""
    lowered = text.lower()
    for pat in RISK_ESCALATORS:
        if re.search(pat, lowered):
            return "critical" if risk in ("high", "medium") else "high"
    return risk


def _score_operation(text: str, op: str) -> int:
    """Count pattern matches for the operation."""
    score = 0
    lowered = text.lower()
    for pat in LEX[op]:
        if re.search(pat, lowered):
            score += 1
    return score


def classify_query(
    user_text: str,
    hinted_operation: Optional[str] = None,
) -> Dict[str, object]:
    """
    Classify NL (or SQL) into operation with risk and confidence.

    Args:
        user_text: natural language or raw SQL
        hinted_operation: if user explicitly chose an operation (e.g., from UI),
                          we trust it but still compute risk/confidence.

    Returns:
        {
          "operation": "SELECT" | "INSERT" | ... | "UNKNOWN",
          "risk": "low" | "medium" | "high" | "critical",
          "confidence": float,
          "role": "reader" | "writer" | "admin",
          "rationale": str
        }
    """
    text = user_text.strip()
    if not text:
        return {
            "operation": "UNKNOWN",
            "risk": "low",
            "confidence": 0.0,
            "role": "reader",
            "rationale": "Empty input.",
        }

    # 1) If a UI/endpoint passes a hinted operation, prefer it (trust but verify).
    if hinted_operation and hinted_operation.upper() in OPS:
        op = hinted_operation.upper()
        # Confidence is lower if text contradicts patterns badly; still keep flow simple here.
        conf = 0.9
        risk = escalate_risk(base_risk(op), text)
        return {
            "operation": op,
            "risk": risk,
            "confidence": conf,
            "role": required_role(op),
            "rationale": f"Operation forced by hint '{op}'.",
        }

    # 2) If the text starts with a SQL keyword, classify by head token (raw SQL case).
    m = SQL_HEAD.match(text)
    if m:
        op = m.group(1).upper()
        risk = escalate_risk(base_risk(op), text)
        # Confidence boosted when raw SQL is explicit
        conf = 0.95
        return {
            "operation": op,
            "risk": risk,
            "confidence": conf,
            "role": required_role(op),
            "rationale": f"Detected raw SQL starting with '{op}'.",
        }

    # 3) NL scoring: tally pattern matches per operation and pick argmax.
    scores = {op: _score_operation(text, op) for op in OPS}
    best_op = max(scores, key=lambda k: scores[k])
    best_score = scores[best_op]

    # Heuristic confidence: normalize by maximum possible hits for that op.
    # (Simple: divide by number of patterns in that op's lexicon.)
    denom = max(1, len(LEX[best_op]))
    conf = min(1.0, best_score / denom)

    # If all zero -> UNKNOWN; default to SELECT if query sounds read-only question-like
    if best_score == 0:
        # Gentle fallback: interrogatives usually imply SELECT
        if re.search(r"\b(show|list|how many|what|which|who|when|where|get|find|display)\b", text.lower()):
            best_op, conf = "SELECT", 0.5
            rationale = "No strong signal; interrogative/read intent implies SELECT."
        else:
            best_op, conf = "UNKNOWN", 0.2
            rationale = "No patterns matched any operation."
    else:
        rationale = f"Matched patterns for {best_op} ({best_score} hits)."

    risk = escalate_risk(base_risk(best_op), text)

    return {
        "operation": best_op,
        "risk": risk,
        "confidence": round(conf, 2),
        "role": required_role(best_op),
        "rationale": rationale,
    }


# ---- Simple CLI for ad-hoc testing
if __name__ == "__main__":
    import json, sys
    if sys.stdin.isatty() and len(sys.argv) > 1:
        sample = " ".join(sys.argv[1:])
    else:
        sample = sys.stdin.read()
    result = classify_query(sample)
    print(json.dumps(result, indent=2))
