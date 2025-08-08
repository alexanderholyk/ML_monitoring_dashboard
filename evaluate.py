# evaluate.py
# Send test set to FastAPI and compute accuracy/precision.

import json
import argparse
from pathlib import Path

import requests

try:
    from sklearn.metrics import accuracy_score, precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000/predict", help="FastAPI predict endpoint")
    parser.add_argument("--test", default="test_data.json", help="Path to test data JSON")
    args = parser.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"I don't know")  # strict per your instruction 4
        return

    with test_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    y_true, y_pred = [], []
    n_ok, n_fail = 0, 0

    for i, item in enumerate(items, start=1):
        text = item["text"]
        true_label = item["true_label"]

        payload = {"text": text, "true_label": true_label}
        try:
            r = requests.post(args.api, json=payload, timeout=10)
            r.raise_for_status()
            pred = r.json().get("sentiment")
            if pred is None:
                print(f"[{i}] No 'sentiment' in response: {r.text}")
                n_fail += 1
                continue
            y_true.append(true_label)
            y_pred.append(pred)
            n_ok += 1
        except Exception as e:
            print(f"[{i}] Request failed: {e}")
            n_fail += 1

    print(f"\nProcessed: {n_ok} ok, {n_fail} failed")

    # Accuracy
    if SKLEARN_AVAILABLE:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    else:
        # Minimal fallbacks
        acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
        # macro precision
        labels = sorted(set(y_true) | set(y_pred))
        prec_vals = []
        for c in labels:
            tp = sum((p == c) and (t == c) for p, t in zip(y_pred, y_true))
            fp = sum((p == c) and (t != c) for p, t in zip(y_pred, y_true))
            prec_vals.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        prec = sum(prec_vals) / len(prec_vals) if prec_vals else 0.0

    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()