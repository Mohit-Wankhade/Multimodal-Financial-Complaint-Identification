"""Multimodal classifier using RoBERTa text features and ResNet image features."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from transformers import AutoModel


class MultimodalComplaintClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int,
        text_model_name: str = "roberta-base",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden = self.text_encoder.config.hidden_size

        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        image_hidden = 2048

        self.fusion = nn.Sequential(
            nn.Linear(text_hidden + image_hidden, text_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(text_hidden, num_labels)

    def forward(self, input_ids, attention_mask, image, token_type_ids=None):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        image_features = self.image_encoder(image).flatten(1)
        features = torch.cat([text_features, image_features], dim=1)
        fused = self.fusion(features)
        return self.classifier(fused)
