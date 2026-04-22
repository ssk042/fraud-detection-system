# Fraud Detection System

## Live Demo
[View Dashboard](https://fraud-detection-system-xmkd449jxsd67vcpemm3qo.streamlit.app/)

---

## Problem
Credit card fraud is rare but costly. With only 0.17% of transactions being fraudulent, a naive model that flags nothing achieves over 99% accuracy — but completely fails to detect fraud.  
This project focuses on **prioritizing high-risk transactions** to maximize fraud detection while keeping manual review workload manageable.

---

## Dataset
- 284,807 transactions from European cardholders  
- 492 fraud cases (0.17% of data)  
- Features: 28 PCA-transformed variables + Time + Amount  

---

## Approach

### Feature Engineering
- Extracted hour of day from transaction timestamp  
- Created night-hour indicator (hours 0–6)  
- Log-transformed transaction amount to reduce skew  
- Z-scored transaction amount to capture anomalous behavior  

### Models
- **Logistic Regression (supervised, class-weighted)**  
  - Learns known fraud patterns  
  - Outputs probability scores for ranking transactions  

- **Isolation Forest (unsupervised anomaly detection)**  
  - Used to detect unusual behavior  
  - Evaluated as a complementary signal rather than primary model  

### Evaluation
- Prioritized **Recall and Precision** over accuracy due to extreme class imbalance  
- Evaluated ranking performance using **ROC-AUC (~0.97)**  
- Simulated real-world fraud triage using **percentile-based thresholds (top X% highest-risk transactions)**  
- Compared Logistic Regression and Isolation Forest, as well as agreement-based signals  

---

## Results

### Logistic Regression Performance
- **Recall (Fraud Detection): ~92%**  
- **Precision: ~6%**  
- **ROC-AUC: ~0.97**  

> Strong ranking performance allows effective prioritization of fraud risk.

---

### Isolation Forest Performance
- Performed poorly in detecting fraud in this dataset  
- Fraud cases were not strong outliers in feature space  
- Demonstrates a key limitation of anomaly detection when fraud patterns are subtle  

---

## Business Impact

### Real-World Simulation (Top 1% Review)

By reviewing only the **top 1% highest-risk transactions**:

- **570 transactions reviewed (out of ~57,000)**  
- **88 fraud cases detected (~89.8% of all fraud)**  
- **482 legitimate transactions flagged (false positives)**  
- **Precision: ~15.4% (~1 in 6 flagged transactions is fraud)**  

---

### Key Takeaway

> The system reduces manual review workload by ~99% while still catching nearly 90% of fraud.

This reflects how fraud teams operate in practice:
- prioritize high-risk transactions  
- balance detection with operational constraints  

---

## Key Insights
- Fraud transactions tend to occur during off-peak hours, supporting time-based feature engineering  
- Fraud amounts are often relatively small, suggesting attempts to avoid detection thresholds  
- Fraud in this dataset is **not strongly anomalous**, making supervised learning more effective than anomaly detection  

---

## Dashboard
- Built an interactive **Streamlit dashboard**  
- Allows users to:
  - Adjust percentage of transactions flagged for review  
  - Compare Logistic Regression vs Isolation Forest vs agreement-based signals  
  - Visualize tradeoffs between **workload, recall, precision, and false alarms**  

---

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit  

---

## One-Line Summary
Built a fraud detection system that uses machine learning to **rank transactions by risk**, enabling teams to catch most fraud while reviewing only a small fraction of activity.