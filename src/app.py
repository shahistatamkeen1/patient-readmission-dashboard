import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Patient Readmission Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression_model.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "reports", "model_results.csv")
LR_CM_PATH = os.path.join(BASE_DIR, "outputs", "charts", "logistic_confusion_matrix.png")
RF_CM_PATH = os.path.join(BASE_DIR, "outputs", "charts", "random_forest_confusion_matrix.png")
ROC_PATH = os.path.join(BASE_DIR, "outputs", "charts", "roc_curve_comparison.png")


@st.cache_data
def load_preview_data():
    return pd.read_csv(DATA_PATH, nrows=200)


@st.cache_data
def load_full_data_for_metrics():
    df = pd.read_csv(DATA_PATH, usecols=["target_readmit_30"])
    return df


@st.cache_data
def load_model_results():
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    return None


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🏥 Patient Readmission Risk Analysis Dashboard")
st.markdown("An interactive healthcare analytics dashboard for predicting 30-day patient readmission risk.")

# Load small/fast pieces first
try:
    metrics_df = load_full_data_for_metrics()
    preview_df = load_preview_data()
    lr_model = load_model()
    results_df = load_model_results()
except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()

# Overview
st.header("📊 Dataset Overview")

total_patients = len(metrics_df)
high_risk = int(metrics_df["target_readmit_30"].sum())
low_risk = total_patients - high_risk
high_risk_pct = round((high_risk / total_patients) * 100, 2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Patients", f"{total_patients:,}")
c2.metric("High Risk Patients", f"{high_risk:,}")
c3.metric("Low Risk Patients", f"{low_risk:,}")
c4.metric("High Risk %", f"{high_risk_pct}%")

# Model performance
st.header("🤖 Model Performance Summary")
if results_df is not None:
    st.dataframe(results_df, use_container_width=True)
else:
    st.warning("model_results.csv not found.")

# Distribution
st.header("📌 Readmission Distribution")
distribution = pd.DataFrame(
    {
        "Category": ["Low Risk (0)", "High Risk (1)"],
        "Count": [low_risk, high_risk],
    }
).set_index("Category")
st.bar_chart(distribution)

# Data preview
st.header("🔍 Cleaned Data Preview")
st.caption("Showing first 200 rows only for faster loading.")
st.dataframe(preview_df.head(10), use_container_width=True)

# Evaluation charts
st.header("📈 Evaluation Charts")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Logistic Regression Confusion Matrix")
    if os.path.exists(LR_CM_PATH):
        st.image(LR_CM_PATH, use_container_width=True)
    else:
        st.info("Logistic confusion matrix image not found.")

with col2:
    st.subheader("Random Forest Confusion Matrix")
    if os.path.exists(RF_CM_PATH):
        st.image(RF_CM_PATH, use_container_width=True)
    else:
        st.info("Random forest confusion matrix image not found.")

st.subheader("ROC Curve Comparison")
if os.path.exists(ROC_PATH):
    st.image(ROC_PATH, use_container_width=True)
else:
    st.info("ROC curve image not found.")

# Prediction demo
st.header("🧠 Predict Readmission Risk")
st.caption("This demo updates a sample patient record with a few editable values.")

try:
    full_sample = pd.read_csv(DATA_PATH, nrows=1)
    input_row = full_sample.drop(columns=["target_readmit_30"]).iloc[0].copy()

    time_in_hospital = st.slider("Time in Hospital", 1, 14, int(input_row.get("time_in_hospital", 5)))
    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 150, int(input_row.get("num_lab_procedures", 40)))
    num_medications = st.slider("Number of Medications", 1, 80, int(input_row.get("num_medications", 10)))
    number_inpatient = st.slider("Previous Inpatient Visits", 0, 20, int(input_row.get("number_inpatient", 1)))
    number_emergency = st.slider("Previous Emergency Visits", 0, 20, int(input_row.get("number_emergency", 0)))
    number_outpatient = st.slider("Previous Outpatient Visits", 0, 50, int(input_row.get("number_outpatient", 0)))

    if st.button("Predict Risk"):
        input_row["time_in_hospital"] = time_in_hospital
        input_row["num_lab_procedures"] = num_lab_procedures
        input_row["num_medications"] = num_medications
        input_row["number_inpatient"] = number_inpatient
        input_row["number_emergency"] = number_emergency
        input_row["number_outpatient"] = number_outpatient

        input_df = pd.DataFrame([input_row])

        prediction = lr_model.predict(input_df)[0]
        probability = lr_model.predict_proba(input_df)[0][1]

        st.write(f"**Predicted Readmission Probability:** {probability:.2%}")

        if prediction == 1:
            st.error("⚠️ High Risk of Readmission within 30 days")
        else:
            st.success("✅ Low Risk of Readmission within 30 days")

except Exception as e:
    st.warning(f"Prediction section could not load: {e}")

st.markdown("---")
st.markdown("Built by Shahista Tamkeen")