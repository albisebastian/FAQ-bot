"""Train the TF-IDF model from data/faqs.csv and evaluate it on data/test_queries.csv.

Usage:  python backend/train.py
"""
from pathlib import Path

import pandas as pd

import matcher

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "faqs.csv"
TEST_PATH = ROOT / "data" / "test_queries.csv"
MODEL_PATH = ROOT / "model" / "faq_model.joblib"
THRESHOLD = matcher.REJECT_THRESHOLD


def evaluate(model: dict, test_path: Path, threshold: float) -> None:
    test = pd.read_csv(test_path)
    in_scope = test[test["expected_id"].notna()].copy()
    out_scope = test[test["expected_id"].isna()]
    in_scope["expected_id"] = in_scope["expected_id"].astype(int)

    top1 = top3 = 0
    for _, row in in_scope.iterrows():
        results = matcher.match(model, row["query"], top_k=3)
        ids = [int(r["id"]) for r in results]
        top1 += ids[0] == row["expected_id"]
        top3 += row["expected_id"] in ids

    n = len(in_scope)
    rejected = sum(matcher.match(model, q, top_k=1)[0]["score"] < threshold for q in out_scope["query"])
    print(f"Top-1 accuracy : {top1 / n:.1%} ({top1}/{n})")
    print(f"Top-3 accuracy : {top3 / n:.1%} ({top3}/{n})")
    print(f"Out-of-scope rejected at threshold {threshold}: {rejected}/{len(out_scope)}")


if __name__ == "__main__":
    df = matcher.load_faqs(DATA_PATH)
    model = matcher.train(df)
    matcher.save(model, MODEL_PATH)
    print(f"Trained on {len(df)} FAQs, vocabulary size {len(model['vectorizer'].vocabulary_)}")
    print(f"Saved model to {MODEL_PATH}\n")
    if TEST_PATH.exists():
        evaluate(model, TEST_PATH, THRESHOLD)
