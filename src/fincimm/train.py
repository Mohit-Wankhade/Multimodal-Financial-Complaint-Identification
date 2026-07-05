"""Training entry point for the multimodal complaint classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from .dataset import FinCIMMDataset
from .model import MultimodalComplaintClassifier
from .preprocessing import build_label_vocabulary, load_dataset, split_labels


def evaluate(model, loader, device):
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
    return {
        "macro_f1": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(all_targets, all_preds, average="micro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg["data"]["train_path"])
    label_names = build_label_vocabulary(df["aspect_labels"].map(split_labels))
    (output_dir / "label_map.json").write_text(json.dumps(label_names, indent=2))

    train_df, val_df = train_test_split(
        df,
        test_size=cfg["data"].get("test_size", 0.2),
        random_state=cfg.get("seed", 42),
        shuffle=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    train_ds = FinCIMMDataset(train_df, tokenizer, label_names, cfg["data"].get("image_root", "."), cfg["training"]["max_length"])
    val_ds = FinCIMMDataset(val_df, tokenizer, label_names, cfg["data"].get("image_root", "."), cfg["training"]["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalComplaintClassifier(
        num_labels=len(label_names),
        text_model_name=cfg["model"]["text_encoder"],
        dropout=cfg["model"].get("dropout", 0.2),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = BCEWithLogitsLoss()

    best_micro_f1 = -1.0
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        print({"epoch": epoch + 1, "loss": running_loss / max(len(train_loader), 1), **metrics})
        if metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = metrics["micro_f1"]
            torch.save(model.state_dict(), output_dir / "model.pt")


if __name__ == "__main__":
    main()
