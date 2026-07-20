"""
Turns a raw transaction (the same shape as what the API receives) into a
numeric feature vector the model can actually learn from.

Kept as its own module (not buried inside training or inference code)
because these EXACT same transformations have to happen both at training
time and at prediction time — if they ever drift apart, the model would
be scored on features it wasn't trained on. Both `train_model.py` and
`app/services/ml_model.py` import from here.
"""

import pandas as pd

TYPE_COLUMNS = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]

FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_delta_orig",
    "balance_delta_dest",
    "drain_ratio",
    "amount_to_balance_ratio",
    "dest_balance_unchanged",
    "orig_balance_zeroed",
] + TYPE_COLUMNS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a DataFrame with raw PaySim-style columns (type, amount,
    oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest) and
    returns a new DataFrame of purely numeric features ready for a model.
    """
    out = pd.DataFrame(index=df.index)

    out["amount"] = df["amount"]
    out["oldbalanceOrg"] = df["oldbalanceOrg"]
    out["newbalanceOrig"] = df["newbalanceOrig"]
    out["oldbalanceDest"] = df["oldbalanceDest"]
    out["newbalanceDest"] = df["newbalanceDest"]

    # How much the sender's/recipient's balance actually moved, vs. what
    # the transaction amount claims — a mismatch is itself a fraud signal
    out["balance_delta_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    out["balance_delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # What fraction of the sender's balance this transaction represents —
    # the single strongest signal for account-takeover-style fraud
    out["drain_ratio"] = (
        (df["oldbalanceOrg"] - df["newbalanceOrig"]) / df["oldbalanceOrg"].replace(0, 1)
    ).clip(0, 1)

    out["amount_to_balance_ratio"] = (
        df["amount"] / df["oldbalanceOrg"].replace(0, 1)
    ).clip(0, 10)

    # Binary flags for the two structural fraud patterns rules also check —
    # letting the model learn how much weight to give these vs. everything else
    out["dest_balance_unchanged"] = (
        (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)
    ).astype(int)
    out["orig_balance_zeroed"] = (df["newbalanceOrig"] <= 0.01).astype(int)

    # One-hot encode transaction type
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    for col in TYPE_COLUMNS:
        out[col] = type_dummies[col] if col in type_dummies.columns else 0

    return out[FEATURE_COLUMNS]
