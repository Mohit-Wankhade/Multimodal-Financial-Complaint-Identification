# GitHub Upload Guide

## 1. Create repository on GitHub

Create an empty GitHub repository named:

```text
multimodal-financial-complaint-identification
```

Do not initialize it with README, license, or `.gitignore` because this folder already contains those files.

## 2. Push from local machine

Open terminal inside this folder and run:

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: multimodal financial complaint identification framework"
git remote add origin https://github.com/Mohit-Wankhade/multimodal-financial-complaint-identification.git
git push -u origin main
```

## 3. Before pushing, verify ignored files

```bash
git status
```

Make sure these are **not** staged:

- `data.xlsx`
- raw images
- full private dataset
- `.env`
- model weights such as `.pt`, `.pth`, `.bin`, `.safetensors`
- Google Drive path dumps
- logs/checkpoints

## 4. Optional: add topics on GitHub

Recommended GitHub topics:

```text
multimodal-learning, financial-nlp, complaint-identification, roberta, resnet, pytorch, aspect-based-classification
```
