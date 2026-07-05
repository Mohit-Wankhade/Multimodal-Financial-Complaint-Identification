# Metrics Summary

The project report compares the proposed FAB-CI model against text-only, image-only, text+image concatenation, SOTA, and VisualBERT baselines.

## Reported highlights

| Domain | ACI Macro-F1 | ACI Micro-F1 | CAC Macro-F1 | CAC Micro-F1 |
|---|---:|---:|---:|---:|
| Transactions | 80.98 | 80.02 | 93.27 | 92.43 |
| Retail Banking | 79.27 | 79.67 | 84.38 | 84.15 |
| Loan | 81.45 | 81.79 | 93.23 | 92.65 |
| Debit Card | 84.24 | 83.94 | 90.30 | 89.01 |
| Investments | 81.45 | 81.02 | 89.17 | 90.02 |
| Economy | 81.78 | 79.45 | 86.08 | 85.21 |
| Customer Service | 84.78 | 83.56 | 85.00 | 83.16 |
| Financial Policies | 81.86 | 83.21 | 84.31 | 84.45 |
| Salary | 86.78 | 89.67 | 81.57 | 82.84 |
| Credit Card | 87.56 | 87.81 | 86.19 | 84.50 |

## Interpretation

FAB-CI improves over unimodal baselines by incorporating text, image, and web entity information. The improvement is especially visible in complaint classification of aspect categories.
