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


# ---------- Custom CSS ----------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 45%, #f9f7ff 100%);
    color: #1f2a44;
}

/* Hide default Streamlit decoration if desired spacing looks odd */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Hero section */
.hero-box {
    background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(240,247,255,0.82));
    border: 1px solid rgba(130, 160, 220, 0.18);
    padding: 28px 30px;
    border-radius: 24px;
    box-shadow: 0 10px 30px rgba(31, 42, 68, 0.08);
    margin-bottom: 20px;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #1b2a52;
    margin-bottom: 0.4rem;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #55627a;
    margin-bottom: 0;
}

/* Section title */
.section-title {
    font-size: 1.8rem;
    font-weight: 750;
    color: #1f2a44;
    margin-top: 18px;
    margin-bottom: 14px;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(130,160,220,0.16);
    backdrop-filter: blur(10px);
    border-radius: 22px;
    padding: 20px 22px;
    box-shadow: 0 10px 26px rgba(31, 42, 68, 0.07);
    min-height: 120px;
}

.metric-label {
    font-size: 0.95rem;
    color: #60708c;
    margin-bottom: 10px;
    font-weight: 600;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1b2a52;
    line-height: 1.1;
}

/* Glass section box */
.glass-box {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(130,160,220,0.16);
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 10px 26px rgba(31, 42, 68, 0.07);
    margin-bottom: 22px;
}

/* Streamlit dataframe/frame polish */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #4f7cff, #7b61ff);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.6rem 1.3rem;
    font-weight: 700;
    box-shadow: 0 8px 18px rgba(79,124,255,0.28);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3e6fff, #6d52ff);
    color: white;
}

/* Slider spacing */
[data-testid="stSlider"] {
    padding-top: 0.2rem;
    padding-bottom: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


# ---------- Load data/model ----------
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


# ---------- Hero ----------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🏥 Patient Readmission Risk Analysis Dashboard</div>
    <p class="hero-subtitle">
        A premium healthcare analytics dashboard for exploring readmission trends,
        evaluating model performance, and simulating risk prediction.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- Overview ----------
st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)

total_patients = len(df)
high_risk = int(df["target_readmit_30"].sum())
low_risk = total_patients - high_risk
high_risk_pct = round((high_risk / total_patients) * 100, 2)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Patients</div>
        <div class="metric-value">{total_patients:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">High Risk Patients</div>
        <div class="metric-value">{high_risk:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Low Risk Patients</div>
        <div class="metric-value">{low_risk:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">High Risk %</div>
        <div class="metric-value">{high_risk_pct}%</div>
    </div>
    """, unsafe_allow_html=True)


# ---------- Model Performance ----------
st.markdown('<div class="section-title">🤖 Model Performance</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)

metric_items = [
    ("Accuracy", f"{metrics['accuracy']:.3f}"),
    ("Precision", f"{metrics['precision']:.3f}"),
    ("Recall", f"{metrics['recall']:.3f}"),
    ("F1 Score", f"{metrics['f1']:.3f}"),
    ("ROC-AUC", f"{metrics['roc_auc']:.3f}"),
]

for col, (label, value) in zip([m1, m2, m3, m4, m5], metric_items):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------- Distribution + Confusion Matrix ----------
left, right = st.columns([1.05, 1])

with left:
    st.markdown('<div class="section-title">📌 Readmission Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    dist_df = pd.DataFrame({
        "Category": ["Low Risk (0)", "High Risk (1)"],
        "Count": [low_risk, high_risk]
    }).set_index("Category")
    st.bar_chart(dist_df)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">📈 Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = metrics["cm"]
    im = ax.imshow(cm)
    ax.set_title("Logistic Regression Confusion Matrix", pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12, fontweight="bold")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------- Data Preview ----------
st.markdown('<div class="section-title">🔍 Data Preview</div>', unsafe_allow_html=True)
st.markdown('<div class="glass-box">', unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ---------- Prediction Demo ----------
st.markdown('<div class="section-title">🧠 Predict Readmission Risk</div>', unsafe_allow_html=True)
st.markdown('<div class="glass-box">', unsafe_allow_html=True)
st.caption("Adjust a few patient-related values to simulate readmission risk prediction.")

input_row = df.drop(columns=["target_readmit_30"]).iloc[0].copy()

editable_fields = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
]

slider_cols = st.columns(2)
for idx, field in enumerate(editable_fields):
    if field in input_row.index:
        current_value = int(input_row[field])
        max_val = max(current_value + 20, 20)
        with slider_cols[idx % 2]:
            input_row[field] = st.slider(
                field.replace("_", " ").title(),
                0,
                max_val,
                current_value
            )

if st.button("Predict Risk"):
    input_df = pd.DataFrame([input_row])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown(f"""
    <div class="metric-card" style="margin-top: 14px;">
        <div class="metric-label">Predicted Readmission Probability</div>
        <div class="metric-value">{probability:.2%}</div>
    </div>
    """, unsafe_allow_html=True)

    if prediction == 1:
        st.error("⚠️ High Risk of Readmission within 30 days")
    else:
        st.success("✅ Low Risk of Readmission within 30 days")

st.markdown('</div>', unsafe_allow_html=True)


# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6b778d; font-weight:600;'>Built by Shahista Tamkeen</div>",
    unsafe_allow_html=True
)