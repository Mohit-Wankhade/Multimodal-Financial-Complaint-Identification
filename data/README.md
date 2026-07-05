# Data

This repository contains only a small synthetic sample dataset in `data/sample/`.

Do **not** commit the original spreadsheet, raw tweet data, private screenshots, downloaded images, or model-ready processed files.

Suggested local-only layout:

```text
data/
├── sample/       # safe public examples committed to GitHub
├── raw/          # original private dataset, ignored by Git
├── processed/    # generated processed data, ignored by Git
└── private/      # any sensitive/non-public data, ignored by Git
```

Original uploaded spreadsheet schema observed during cleanup:

- Domain
- Complaint/ Opinion
- Over-all_Complaint Label
- Aspects_1, Complaint_1, Cause_1
- Aspect_2, Complaint_2, Cause_2
- Aspect_3, Complaint_3, Cause_3
- Complaint_Cause
- Severity level
- Sentiment
- Emotion
- Image Link
- Web Entity

For GitHub, use the simplified public-safe sample format:

```csv
id,domain,text,image_path,web_entities,aspect_labels,complaint_label
```
