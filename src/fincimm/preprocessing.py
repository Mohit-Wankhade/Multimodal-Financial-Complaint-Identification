"""Preprocessing utilities for FinCIMM-style multimodal complaint data."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd


ASPECT_NORMALIZATION = {
    "feedback": "review",
    "review": "review",
    "general opinion": "review",
    "miscellaneous information": "misc_info",
    "financial advertisement": "misc_info",
    "delay response": "provider_response",
    "provider response": "provider_response",
    "payment failure": "net_banking_issue",
    "payment issue": "net_banking_issue",
    "refund": "net_banking_issue",
    "banking": "net_banking_issue",
    "net banking": "net_banking_issue",
    "fraud": "consumer_safety",
    "harassment": "consumer_safety",
    "harrasment": "consumer_safety",
    "financial policy": "financial_info",
    "news": "financial_info",
    "debt collection": "financial_info",
    "credentials": "credential_error",
    "credential error": "credential_error",
    "technical issue": "technical_issue",
    "general query": "general_query",
}


def clean_text(text: object) -> str:
    """Lowercase text and remove noisy spacing/punctuation artifacts."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9@#_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def parse_web_entities(value: object) -> List[str]:
    """Parse web entity strings from spreadsheet/list/pipe-separated formats."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()
        if "|" in text:
            raw_items = text.split("|")
        else:
            try:
                parsed = ast.literal_eval(text)
                raw_items = parsed if isinstance(parsed, list) else [text]
            except Exception:
                raw_items = re.split(r",|;", text)
    return [clean_text(item) for item in raw_items if clean_text(item)]


def normalize_aspect(value: object) -> str | None:
    """Normalize noisy aspect names into stable label names."""
    if pd.isna(value):
        return None
    aspect = clean_text(value)
    if not aspect or aspect in {"nan", "none", "-"}:
        return None
    return ASPECT_NORMALIZATION.get(aspect, aspect.replace(" ", "_"))


def split_labels(value: object) -> List[str]:
    """Split pipe/comma separated labels into normalized label names."""
    if pd.isna(value):
        return []
    labels: list[str] = []
    for item in re.split(r"\||,", str(value)):
        label = normalize_aspect(item)
        if label:
            labels.append(label)
    return sorted(set(labels))


def convert_original_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the original report spreadsheet schema into the public training schema."""
    rows = []
    for idx, row in df.iterrows():
        labels = []
        for col in ["Aspects_1", "Aspect_2", "Aspect_3"]:
            if col in df.columns:
                label = normalize_aspect(row.get(col))
                if label:
                    labels.append(label)
        text = row.get("Complaint/ Opinion", "")
        rows.append(
            {
                "id": row.get("Unnamed: 0", idx),
                "domain": str(row.get("Domain", "")).strip(),
                "text": clean_text(text),
                "image_path": row.get("Image Link", ""),
                "web_entities": "|".join(parse_web_entities(row.get("Web Entity", ""))),
                "aspect_labels": "|".join(sorted(set(labels))),
                "complaint_label": int(row.get("Over-all_Complaint Label", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load CSV/XLSX and return normalized public training schema."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if "Complaint/ Opinion" in df.columns:
        df = convert_original_schema(df)
    else:
        df = df.copy()
        df["text"] = df["text"].map(clean_text)
        df["web_entities"] = df["web_entities"].fillna("").map(lambda x: "|".join(parse_web_entities(x)))
        df["aspect_labels"] = df["aspect_labels"].fillna("").map(lambda x: "|".join(split_labels(x)))
    return df


def build_label_vocabulary(label_rows: Iterable[Iterable[str]]) -> list[str]:
    labels = sorted({label for row in label_rows for label in row})
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV/XLSX dataset")
    parser.add_argument("--output", default=None, help="Optional normalized CSV output path")
    args = parser.parse_args()

    df = load_dataset(args.input)
    print(df.head())
    print(f"Rows: {len(df)}")
    print("Labels:", build_label_vocabulary(df["aspect_labels"].map(split_labels)))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved normalized dataset to {args.output}")


if __name__ == "__main__":
    main()
