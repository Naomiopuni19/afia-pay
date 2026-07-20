"""
Loads the trained fraud detection model (trained by scripts/train_model.py)
and uses it to score real transactions coming through the API.
"""

import os
import joblib
import pandas as pd

from app.models.transaction import TransactionRequest
from app.core.features import engineer_features

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "fraud_model.joblib")

_model = None
_model_load_attempted = False


def _load_model():
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True

    if os.path.exists(_MODEL_PATH):
        try:
            _model = joblib.load(_MODEL_PATH)
            print(f"✅ Fraud detection model loaded from {_MODEL_PATH}")
        except Exception as err:
            print(f"⚠️  Failed to load model at {_MODEL_PATH}: {err}")
            _model = None
    else:
        print(f"⚠️  No trained model found at {_MODEL_PATH} — ML scoring disabled, rules-only mode.")
        _model = None

    return _model


def _to_paysim_row(txn: TransactionRequest) -> pd.DataFrame:
    """Converts an API request (our field names) into the PaySim-style
    column names the feature engineering pipeline expects — this is the
    translation layer between the API's naming and the dataset's naming."""
    return pd.DataFrame([{
        "type": txn.type.value,
        "amount": txn.amount,
        "oldbalanceOrg": txn.old_balance_orig,
        "newbalanceOrig": txn.new_balance_orig,
        "oldbalanceDest": txn.old_balance_dest,
        "newbalanceDest": txn.new_balance_dest,
    }])


def get_ml_score(txn: TransactionRequest) -> float | None:
    """
    Returns a fraud probability (0-1) from the trained ML model, or None
    if no model has been trained/loaded yet (Phase 1 fallback — the rest
    of the system still works fine on rules alone in that case).
    """
    model = _load_model()
    if model is None:
        return None

    row = _to_paysim_row(txn)
    features = engineer_features(row)
    proba = model.predict_proba(features)[0][1]  # probability of class "1" (fraud)
    return round(float(proba), 4)
