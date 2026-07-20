"""
Trains the fraud detection model and saves it to disk for the API to load.

A critical thing this script does NOT do: report plain accuracy as a
success metric. With ~0.15% fraud, a model that predicts "not fraud"
every single time would score 99.85% accuracy while being completely
useless. Instead this evaluates on:
  - Precision/Recall for the fraud class specifically (how many flagged
    transactions were actually fraud, and how much real fraud did we catch)
  - ROC-AUC (how well the model ranks fraud above non-fraud generally)
  - A confusion matrix, so it's visible exactly how many frauds were
    missed vs. how many legitimate transactions got wrongly flagged

class_weight="balanced" is used specifically to counteract the severe
imbalance — without it, the model would learn that always predicting
"not fraud" is a great strategy, since it's right 99.85% of the time.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.features import engineer_features, FEATURE_COLUMNS

DATA_PATH = "data/transactions.csv"
MODEL_PATH = "models/fraud_model.joblib"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} transactions ({df['isFraud'].sum()} fraud, {df['isFraud'].mean():.3%})")

    print("\nEngineering features...")
    X = engineer_features(df)
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\nTraining RandomForestClassifier (class_weight='balanced' to handle the ~0.15% fraud rate)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("EVALUATION (on held-out test set the model never saw)")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], digits=3))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}  (1.0 = perfect ranking, 0.5 = random guessing)")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix:")
    print(f"  True Negatives  (correctly cleared):  {tn:,}")
    print(f"  False Positives (wrongly flagged):    {fp:,}")
    print(f"  False Negatives (fraud missed):       {fn:,}")
    print(f"  True Positives  (fraud caught):       {tp:,}")

    print("\nTop feature importances:")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(importances.head(8).round(4))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
