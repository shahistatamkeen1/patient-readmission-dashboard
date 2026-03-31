import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

print("Step 1: Starting evaluation script...")

os.makedirs("outputs/charts", exist_ok=True)

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_data.csv")
print("Step 2: Cleaned data loaded")

# Split features and target
X = df.drop("target_readmit_30", axis=1)
y = df["target_readmit_30"]

# Load trained models
lr_model = joblib.load("models/logistic_regression_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
print("Step 3: Models loaded")

# -----------------------------
# Logistic Regression Confusion Matrix
# -----------------------------
lr_pred = lr_model.predict(X)
cm_lr = confusion_matrix(y, lr_pred)

disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr)
disp_lr.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/charts/logistic_confusion_matrix.png")
plt.close()

# -----------------------------
# Random Forest Confusion Matrix
# -----------------------------
rf_pred = rf_model.predict(X)
cm_rf = confusion_matrix(y, rf_pred)

disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot()
plt.title("Random Forest Confusion Matrix")
plt.savefig("outputs/charts/random_forest_confusion_matrix.png")
plt.close()

# -----------------------------
# ROC Curves
# -----------------------------
lr_prob = lr_model.predict_proba(X)[:, 1]
rf_prob = rf_model.predict_proba(X)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y, lr_prob)
fpr_rf, tpr_rf, _ = roc_curve(y, rf_prob)

auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("outputs/charts/roc_curve_comparison.png")
plt.close()

print("Step 4: Charts saved successfully")