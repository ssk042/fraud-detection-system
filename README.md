# Fraud Detection System

## Problem
Credit card fraud is rare but costly. With only 0.17% of transactions being fraudulent,
a naive model that flags nothing achieves 99.8% accuracy — and is completely useless.
This project builds a system that actually catches fraud while managing false alarms.

## Dataset
- 284,807 transactions from European cardholders
- 492 fraud cases (0.17% of data)
- Features: 28 PCA-transformed variables + Time + Amount

## Approach

**Feature Engineering**
- Extracted hour of day from transaction timestamp
- Created night-hour indicator (hours 0–6)
- Log-transformed transaction amount to reduce skew
- Z-scored amount to normalize scale

**Models**
- Logistic Regression (supervised baseline, class-weighted)
- Isolation Forest (unsupervised anomaly detection)

**Evaluation**
- Prioritized Recall and Precision over accuracy
- Plotted Precision-Recall and ROC curves
- Simulated real-world business threshold

## Results

| Model | Recall | Precision | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.92 | 0.06 | 0.97 |
| Isolation Forest | 0.31 | 0.28 | — |

## Business Impact

By flagging the top 1% highest-risk transactions:
- **89.8% of fraud caught**
- Reviewers examine only 570 out of 56,962 transactions
- **99% reduction in manual review workload**

## Key Insight
Fraud spikes at 2am and 11am — when human reviewers are least active.
Fraud amounts cluster between $1–$106, suggesting deliberate evasion of large-transaction alerts.

## Tech Stack
Python, scikit-learn, pandas, NumPy, matplotlib, seaborn

## Resume Bullet
Built fraud detection system using Logistic Regression and Isolation Forest on 284K+ transactions,
achieving 92% fraud recall and 0.97 ROC-AUC through threshold optimization —
reducing manual review workload by 99% via risk-based flagging.