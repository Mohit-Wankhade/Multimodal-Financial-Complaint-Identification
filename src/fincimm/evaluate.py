"""Evaluation helper for multilabel aspect predictions."""

from __future__ import annotations

import argparse
import pandas as pd
from sklearn.metrics import classification_report, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="CSV with columns y_true and y_pred as pipe-separated labels")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    label_names = sorted(set("|".join(df["y_true"].fillna("")).split("|")) | set("|".join(df["y_pred"].fillna("")).split("|")))
    label_names = [label for label in label_names if label]

    def encode(text):
        labels = set(str(text).split("|")) if pd.notna(text) else set()
        return [1 if label in labels else 0 for label in label_names]

    y_true = [encode(x) for x in df["y_true"]]
    y_pred = [encode(x) for x in df["y_pred"]]
    print("Macro F1:", f1_score(y_true, y_pred, average="macro", zero_division=0))
    print("Micro F1:", f1_score(y_true, y_pred, average="micro", zero_division=0))
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))


if __name__ == "__main__":
    main()
