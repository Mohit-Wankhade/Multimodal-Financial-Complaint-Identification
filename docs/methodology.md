# Methodology

## Problem

The goal is to identify financial complaints at aspect level using multimodal inputs. A single review or tweet may contain multiple financial aspects, and only some of them may express a complaint.

## Inputs

The framework uses three input sources:

1. **Text** - complaint/review text from financial social media or consumer feedback.
2. **Image** - screenshots or visual evidence associated with the post.
3. **Web Entities** - image-derived keywords/entities that provide additional context.

## Tasks

### 1. Aspect Category Identification / ACI

ACI is modeled as a multi-label classification task. Each input can be assigned one or more financial aspect categories.

### 2. Complaint Classification of Aspect Categories / CAC

CAC classifies whether each identified aspect represents a complaint or non-complaint.

## Encoders

- **Text Encoder:** RoBERTa-based contextual encoder.
- **Image Encoder:** ResNet-152 visual feature extractor.
- **Fusion:** Multimodal fusion/attention mechanism for combining text, image, and web entity features.
- **Classifier:** Dense classifier for aspect-level predictions.

## Training Setup from Project Report

- Optimizer: Adam
- Batch size: 16 in reported experiments
- Dropout: 0.2
- Learning rate: 1e-4 in the report setup table
- GPU: RTX 2080 Ti

## Notes for Reproduction

The original experiment used local/Google Drive file paths. Before reproducing training, update:

- dataset path,
- image directory path,
- output checkpoint path,
- label mapping.

The cleaned notebook is provided under `notebooks/` for reference.
