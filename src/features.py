import numpy as np
import pandas as pd

def engineer_features(df):
    df = df.copy()

    # Time-based features
    df['hour'] = (df['Time'] // 3600) % 24
    df['is_night'] = (df['hour'] <= 6).astype(int)

    # Amount-based features
    df['amount_log'] = np.log1p(df['Amount'])
    df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

    # Drop raw columns we've replaced
    df = df.drop(columns=['Time', 'Amount'])

    return df