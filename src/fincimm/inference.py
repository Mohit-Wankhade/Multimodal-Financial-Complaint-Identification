"""Single-sample inference for a trained multimodal complaint model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from .dataset import FinCIMMDataset
from .model import MultimodalComplaintClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--web-entities", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--text-model", default="roberta-base")
    args = parser.parse_args()

    label_names = json.loads(Path(args.label_map).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    df = pd.DataFrame(
        [
            {
                "text": args.text,
                "image_path": args.image,
                "web_entities": args.web_entities,
                "aspect_labels": "",
            }
        ]
    )
    ds = FinCIMMDataset(df, tokenizer, label_names, image_root=".")
    batch = ds[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalComplaintClassifier(len(label_names), text_model_name=args.text_model).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(
            input_ids=batch["input_ids"].unsqueeze(0).to(device),
            attention_mask=batch["attention_mask"].unsqueeze(0).to(device),
            image=batch["image"].unsqueeze(0).to(device),
        )
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    predictions = [(label, round(prob, 4)) for label, prob in zip(label_names, probs) if prob >= args.threshold]
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
