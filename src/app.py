import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Patient Readmission Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "assets", "sample_cleaned_data.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(df):
    X = df.drop("target_readmit_30", axis=1)
    y = df["target_readmit_30"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "cm": confusion_matrix(y_test, y_pred),
    }

    return model, metrics


st.title("🏥 Patient Readmission Risk Analysis Dashboard")
st.markdown("A lightweight deployment version of the healthcare readmission risk project.")

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load sample dataset: {e}")
    st.stop()

try:
    model, metrics = train_model(df)
except Exception as e:
    st.error(f"Could not train model from sample dataset: {e}")
    st.stop()

st.header("📊 Dataset Overview")

total_patients = len(df)
high_risk = int(df["target_readmit_30"].sum())
low_risk = total_patients - high_risk
high_risk_pct = round((high_risk / total_patients) * 100, 2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Patients", f"{total_patients:,}")
c2.metric("High Risk Patients", f"{high_risk:,}")
c3.metric("Low Risk Patients", f"{low_risk:,}")
c4.metric("High Risk %", f"{high_risk_pct}%")

st.header("🤖 Model Performance")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
m2.metric("Precision", f"{metrics['precision']:.3f}")
m3.metric("Recall", f"{metrics['recall']:.3f}")
m4.metric("F1 Score", f"{metrics['f1']:.3f}")
m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.header("📌 Readmission Distribution")
dist_df = pd.DataFrame({
    "Category": ["Low Risk (0)", "High Risk (1)"],
    "Count": [low_risk, high_risk]
}).set_index("Category")
st.bar_chart(dist_df)

st.header("🔍 Data Preview")
st.dataframe(df.head(10), use_container_width=True)

st.header("📈 Confusion Matrix")
fig, ax = plt.subplots()
cm = metrics["cm"]
ax.imshow(cm)
ax.set_title("Logistic Regression Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")
st.pyplot(fig)

st.header("🧠 Predict Readmission Risk")
st.caption("This uses a sample patient row and lets you change a few numeric values.")

input_row = df.drop(columns=["target_readmit_30"]).iloc[0].copy()

editable_fields = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
]

for field in editable_fields:
    if field in input_row.index:
        current_value = int(input_row[field])
        max_val = max(current_value + 20, 20)
        input_row[field] = st.slider(field.replace("_", " ").title(), 0, max_val, current_value)

if st.button("Predict Risk"):
    input_df = pd.DataFrame([input_row])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write(f"**Predicted Readmission Probability:** {probability:.2%}")
    if prediction == 1:
        st.error("⚠️ High Risk of Readmission within 30 days")
    else:
        st.success("✅ Low Risk of Readmission within 30 days")

st.markdown("---")
st.markdown("Built by Shahista Tamkeen")