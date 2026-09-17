# ============================================================
# MindTrack - Dual-Modal Mental Health Risk Prediction System
# ============================================================

# Install required libraries
# Run this separately in Google Colab/Jupyter if required:
# !pip install pandas numpy scikit-learn imbalanced-learn pytorch-tabnet torch matplotlib seaborn


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import torch

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from pytorch_tabnet.tab_model import TabNetClassifier


# ============================================================
# 2. LOAD DATASETS
# ============================================================

# PHQ-9 Dataset
phq = pd.read_csv("PHQ-9 Dataset.csv")

print("PHQ shape:", phq.shape)
display(phq.head())


# Wearable Dataset
wearable = pd.read_csv("Wearable Dataset.csv")

print("Wearable shape:", wearable.shape)
display(wearable.head())


# ============================================================
# 3. PHQ-9 DATA PREPROCESSING
# ============================================================

# Map PHQ-9 responses to numerical values

mapping = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

phq = phq.replace(mapping)


# Target variable
y_phq = phq["PHQ_Severity"]


# Feature variables
X_phq = phq.drop(columns=["PHQ_Severity"])


# ============================================================
# 4. PHQ-9 TRAIN-TEST SPLIT
# ============================================================

X_train_phq, X_test_phq, y_train_phq, y_test_phq = train_test_split(
    X_phq,
    y_phq,
    test_size=0.2,
    random_state=42,
    stratify=y_phq
)


# ============================================================
# 5. PHQ-9 CLASS DISTRIBUTION
# ============================================================

train_counts = y_train_phq.value_counts().sort_index()
test_counts = y_test_phq.value_counts().sort_index()

split_table = pd.DataFrame({
    "Training (80%)": train_counts,
    "Testing (20%)": test_counts
})

print("\n=== PHQ-9 CLASS DISTRIBUTION: TRAIN vs TEST ===\n")
print(split_table)

print("\nTOTAL DATA:", len(X_phq))
print("Total Training:", train_counts.sum())
print("Total Testing:", test_counts.sum())


# ============================================================
# 6. PHQ-9 FEATURE SCALING
# ============================================================

scaler_phq = StandardScaler()

X_train_phq_scaled = scaler_phq.fit_transform(X_train_phq)
X_test_phq_scaled = scaler_phq.transform(X_test_phq)


print("\nClass distribution BEFORE SMOTE:")
print(y_train_phq.value_counts())


# ============================================================
# 7. PHQ-9 SMOTE
# ============================================================

smote_phq = SMOTE(random_state=42)

X_train_phq_bal, y_train_phq_bal = smote_phq.fit_resample(
    X_train_phq_scaled,
    y_train_phq
)

print("\nClass distribution AFTER SMOTE:")
print(pd.Series(y_train_phq_bal).value_counts())


# ============================================================
# 8. PHQ-9 TABNET MODEL
# ============================================================

X_train_np = np.array(X_train_phq_bal)
y_train_np = np.array(y_train_phq_bal)

X_test_np = np.array(X_test_phq_scaled)
y_test_np = np.array(y_test_phq)


phq_model = TabNetClassifier(
    n_d=8,
    n_a=8,
    n_steps=3,
    gamma=1.3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={
        "step_size": 10,
        "gamma": 0.9
    },
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=1
)


phq_model.fit(
    X_train_np,
    y_train_np,
    eval_set=[
        (X_test_np, y_test_np)
    ],
    max_epochs=50,
    patience=50,
    batch_size=32,
    virtual_batch_size=16
)


# ============================================================
# 9. PHQ-9 PREDICTION & EVALUATION
# ============================================================

phq_preds = phq_model.predict(X_test_np)

# Calculate accuracy
phq_acc = accuracy_score(
    y_test_np,
    phq_preds
)

precision_phq = precision_score(
    y_test_np,
    phq_preds,
    average="weighted"
)

recall_phq = recall_score(
    y_test_np,
    phq_preds,
    average="weighted"
)

f1_phq = f1_score(
    y_test_np,
    phq_preds,
    average="weighted"
)


print("\n========================================")
print("        PHQ-9 TABNET RESULTS")
print("========================================")

print("PHQ-9 Accuracy:", phq_acc)
print("PHQ-9 Precision:", precision_phq)
print("PHQ-9 Recall:", recall_phq)
print("PHQ-9 F1-score:", f1_phq)

print("\nClassification Report:\n")
print(
    classification_report(
        y_test_np,
        phq_preds
    )
)


# ============================================================
# 10. WEARABLE DATA PREPROCESSING
# ============================================================

X_wear = wearable.copy()


# ============================================================
# 11. CREATE WEARABLE RISK PROXY LABEL
# ============================================================

# The wearable dataset does not contain direct mental-health
# labels. Therefore, a proxy risk label is created using
# sleep duration and heart-rate patterns.

def create_wearable_risk(row):

    sleep = row["sleep_duration_min"]
    hr = row["heart_rate_bpm"]

    # Very High Risk
    if sleep <= 200 or hr >= 95 or hr < 55:
        return 4

    # High Risk
    elif sleep <= 300 or hr >= 90:
        return 3

    # Moderate Risk
    elif sleep <= 360 or hr >= 85:
        return 2

    # Mild Risk
    elif sleep <= 420 or hr >= 80 or hr < 65:
        return 1

    # Low Risk
    else:
        return 0


wearable["wearable_risk"] = wearable.apply(
    create_wearable_risk,
    axis=1
)


# Features
X_wear = wearable.drop(
    columns=["wearable_risk"]
)

# Target
y_wear = wearable["wearable_risk"]


print("\nWearable Risk Distribution:")
print(y_wear.value_counts().sort_index())


# ============================================================
# 12. WEARABLE TRAIN-TEST SPLIT
# ============================================================

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_wear,
    y_wear,
    test_size=0.2,
    random_state=42,
    stratify=y_wear
)


# ============================================================
# 13. WEARABLE CLASS DISTRIBUTION
# ============================================================

train_counts_w = y_train_w.value_counts().sort_index()
test_counts_w = y_test_w.value_counts().sort_index()

split_table_w = pd.DataFrame({
    "Training (80%)": train_counts_w,
    "Testing (20%)": test_counts_w
})

print("\n=== WEARABLE CLASS DISTRIBUTION: TRAIN vs TEST ===\n")
print(split_table_w)

print("\nTOTAL WEARABLE DATA:", len(X_wear))
print("Total Training:", train_counts_w.sum())
print("Total Testing:", test_counts_w.sum())


# ============================================================
# 14. WEARABLE FEATURE SCALING
# ============================================================

scaler_w = StandardScaler()

X_train_w_scaled = scaler_w.fit_transform(
    X_train_w
)

X_test_w_scaled = scaler_w.transform(
    X_test_w
)


print("\nClass distribution BEFORE SMOTE:")
print(
    pd.Series(y_train_w)
    .value_counts()
    .sort_index()
)


# ============================================================
# 15. WEARABLE SMOTE
# ============================================================

smote_w = SMOTE(random_state=42)

X_train_w_bal, y_train_w_bal = smote_w.fit_resample(
    X_train_w_scaled,
    y_train_w
)

print("\nClass distribution AFTER SMOTE:")
print(
    pd.Series(y_train_w_bal)
    .value_counts()
    .sort_index()
)


# ============================================================
# 16. WEARABLE TABNET MODEL
# ============================================================

wear_model = TabNetClassifier(
    n_d=8,
    n_a=8,
    n_steps=3,
    gamma=1.3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={
        "step_size": 10,
        "gamma": 0.9
    },
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=1
)


wear_model.fit(
    np.array(X_train_w_bal),
    np.array(y_train_w_bal),
    eval_set=[
        (
            np.array(X_test_w_scaled),
            np.array(y_test_w)
        )
    ],
    max_epochs=50,
    patience=50,
    batch_size=32,
    virtual_batch_size=16
)


# ============================================================
# 17. WEARABLE PREDICTION & EVALUATION
# ============================================================

wear_preds = wear_model.predict(
    np.array(X_test_w_scaled)
)


wear_acc = accuracy_score(
    y_test_w,
    wear_preds
)

precision_w = precision_score(
    y_test_w,
    wear_preds,
    average="weighted"
)

recall_w = recall_score(
    y_test_w,
    wear_preds,
    average="weighted"
)

f1_w = f1_score(
    y_test_w,
    wear_preds,
    average="weighted"
)


print("\n========================================")
print("       WEARABLE TABNET RESULTS")
print("========================================")

print("Wearable Accuracy:", wear_acc)
print("Wearable Precision:", precision_w)
print("Wearable Recall:", recall_w)
print("Wearable F1-score:", f1_w)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test_w,
        wear_preds
    )
)


# ============================================================
# 18. FINAL MODEL SUMMARY
# ============================================================

print("\n========================================")
print("          MINDTRACK RESULTS")
print("========================================")

print("\nPHQ-9 Model")
print("Accuracy :", round(phq_acc, 4))
print("Precision:", round(precision_phq, 4))
print("Recall   :", round(recall_phq, 4))
print("F1-score :", round(f1_phq, 4))

print("\nWearable Model")
print("Accuracy :", round(wear_acc, 4))
print("Precision:", round(precision_w, 4))
print("Recall   :", round(recall_w, 4))
print("F1-score :", round(f1_w, 4))


# ============================================================
# END OF MINDTRACK IMPLEMENTATION
# ============================================================
