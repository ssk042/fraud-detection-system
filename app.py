import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
sys.path.append('src')
from features import engineer_features

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("🔍 Fraud Detection System")
st.markdown("Credit card fraud detection using Logistic Regression and Isolation Forest")

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv('data/sample.csv')
    return df

@st.cache_resource
def load_models():
    lr_model = joblib.load('outputs/models/logistic_regression.pkl')
    scaler = joblib.load('outputs/models/scaler.pkl')
    return lr_model, scaler

df = load_data()
lr_model, scaler = load_models()
df_features = engineer_features(df)

# Sidebar
st.sidebar.header("Controls")
threshold_pct = st.sidebar.slider("Flag top X% highest-risk transactions", 1, 10, 1)

# Row 1 - Key metrics
st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Fraud Cases", "492")
col3.metric("Fraud Rate", "0.17%")
col4.metric("ROC-AUC", "0.97")

st.divider()

# Row 2 - Charts
st.subheader("Exploratory Analysis")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    df['hour'] = (df['Time'] // 3600) % 24
    df[df['Class'] == 0]['hour'].hist(bins=24, ax=ax, color='steelblue', alpha=0.7, label='Legit')
    df[df['Class'] == 1]['hour'].hist(bins=24, ax=ax, color='crimson', alpha=0.7, label='Fraud')
    ax.set_title('Transactions by Hour')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Count')
    ax.legend()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Class', y='Amount', data=df[df['Amount'] < 500],
                palette=['steelblue', 'crimson'], ax=ax)
    ax.set_title('Transaction Amount by Class')
    ax.set_xticklabels(['Legit', 'Fraud'])
    ax.set_xlabel('Class')
    ax.set_ylabel('Amount ($)')
    st.pyplot(fig)

st.divider()

# Row 3 - Business simulation
st.subheader("Business Simulation")

X = df_features.drop(columns=['Class'])
y = df_features['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_test_scaled = scaler.transform(X_test)
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

threshold = np.percentile(lr_probs, 100 - threshold_pct)
flagged_mask = lr_probs >= threshold
fraud_caught = y_test.values[flagged_mask].sum()
total_flagged = flagged_mask.sum()
false_alarms = total_flagged - fraud_caught

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions Flagged", f"{total_flagged:,}")
col2.metric("Fraud Caught", f"{fraud_caught}")
col3.metric("Catch Rate", f"{fraud_caught / y_test.sum():.1%}")
col4.metric("False Alarms", f"{false_alarms:,}")

st.info(f"By reviewing only the top {threshold_pct}% highest-risk transactions, the model catches {fraud_caught / y_test.sum():.1%} of all fraud — reducing manual review workload by {(1 - total_flagged/len(y_test)):.1%}.")