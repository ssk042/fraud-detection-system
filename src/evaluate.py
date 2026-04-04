import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import joblib

def save_models(lr_model, scaler, iso_model, path='../outputs/models/'):
    joblib.dump(lr_model, path + 'logistic_regression.pkl')
    joblib.dump(scaler, path + 'scaler.pkl')
    joblib.dump(iso_model, path + 'isolation_forest.pkl')
    print("Models saved.")

def print_business_summary(y_test, lr_probs):
    threshold = np.percentile(lr_probs, 99)
    flagged_mask = lr_probs >= threshold
    
    print("=" * 50)
    print("BUSINESS SUMMARY")
    print("=" * 50)
    print(f"Total flagged:       {flagged_mask.sum()}")
    print(f"Fraud caught:        {y_test[flagged_mask].sum()}")
    print(f"Catch rate:          {y_test[flagged_mask].sum() / y_test.sum():.1%}")
    print(f"Precision:           {y_test[flagged_mask].sum() / flagged_mask.sum():.1%}")