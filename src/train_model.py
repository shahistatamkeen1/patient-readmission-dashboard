import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("Step 1: Starting model training script...")

# Create folders if they do not exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_data.csv")
print("Step 2: Cleaned data loaded")
print("Dataset Shape:", df.shape)

# Split features and target
X = df.drop("target_readmit_30", axis=1)
y = df["target_readmit_30"]

print("Step 3: Features and target prepared")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Step 4: Train-test split completed")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Helper function
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    return results

# -----------------------------
# Logistic Regression
# -----------------------------
print("Step 5: Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_model.fit(X_train, y_train)
print("Step 6: Logistic Regression training finished")

lr_results = evaluate_model("Logistic Regression", lr_model, X_test, y_test)

# Save Logistic Regression model
joblib.dump(lr_model, "models/logistic_regression_model.pkl")
print("Step 7: Logistic Regression model saved")

# -----------------------------
# Random Forest
# -----------------------------
print("Step 8: Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("Step 9: Random Forest training finished")

rf_results = evaluate_model("Random Forest", rf_model, X_test, y_test)

# Save Random Forest model
joblib.dump(rf_model, "models/random_forest_model.pkl")
print("Step 10: Random Forest model saved")

# -----------------------------
# Save Results
# -----------------------------
results_df = pd.DataFrame([lr_results, rf_results])
results_df.to_csv("outputs/reports/model_results.csv", index=False)

print("\nModel Evaluation Results:")
print(results_df)

print("\nStep 11: Results saved to outputs/reports/model_results.csv")
print("Training completed successfully!")