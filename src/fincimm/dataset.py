"""PyTorch dataset for multimodal financial complaint classification."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .preprocessing import parse_web_entities, split_labels


class FinCIMMDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        label_names: Sequence[str],
        image_root: str | Path = ".",
        max_length: int = 256,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_names = list(label_names)
        self.image_root = Path(image_root)
        self.max_length = max_length
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_path: str):
        path = self.image_root / str(image_path)
        if not path.exists():
            image = Image.new("RGB", (224, 224), color="white")
        else:
            image = Image.open(path).convert("RGB")
        return self.image_transform(image)

    def _encode_labels(self, label_text: str) -> torch.Tensor:
        labels = split_labels(label_text)
        target = torch.zeros(len(self.label_names), dtype=torch.float32)
        for label in labels:
            if label in self.label_names:
                target[self.label_names.index(label)] = 1.0
        return target

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        entities = " ".join(parse_web_entities(row.get("web_entities", "")))
        combined_text = f"{row.get('text', '')} [SEP] {entities}".strip()
        encoded = self.tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["image"] = self._load_image(row.get("image_path", ""))
        item["labels"] = self._encode_labels(row.get("aspect_labels", ""))
        return item
