"""
Generates a synthetic mobile-money transaction dataset that mirrors the
statistical structure of the real PaySim dataset (Lopez-Rojas et al.),
without using the actual downloaded file.

Why these specific numbers aren't arbitrary — they're taken from PaySim's
published characteristics:
  - Transaction type mix: CASH_OUT ~35%, PAYMENT ~34%, CASH_IN ~22%,
    TRANSFER ~8%, DEBIT ~1%
  - Fraud ONLY occurs in TRANSFER and CASH_OUT transactions in the real
    dataset — this mirrors a real fraud pattern (move money out via
    transfer, then cash it out through an agent/ATM before it's noticed)
  - Fraud rate is tiny: ~0.1-0.2% of all transactions. This is realistic
    and important — it's *why* naive accuracy is a useless metric for
    fraud detection (a model that never predicts fraud would still be
    99.8% "accurate"), and why the training script below evaluates on
    precision/recall/ROC-AUC instead.

Fraud is injected in two flavors on purpose:
  - "Obvious" fraud: full balance drain, large amount — these should get
    caught by BOTH the rule engine and the ML model, showing the two
    layers agree on the easy cases.
  - "Subtle" fraud: partial drain, moderate amount, doesn't trip the
    hard-coded rule thresholds — these exist specifically to demonstrate
    what the ML layer catches that rules alone would miss.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)

N_TRANSACTIONS = 150_000
N_STEPS = 744  # 31 days simulated hourly, matching real PaySim's run length

TYPE_PROBS = {
    "CASH_OUT": 0.352,
    "PAYMENT": 0.338,
    "CASH_IN": 0.220,
    "TRANSFER": 0.084,
    "DEBIT": 0.006,
}

FRAUD_RATE = 0.0015  # ~0.15% — deliberately tiny, matches real-world fraud prevalence
FRAUD_ELIGIBLE_TYPES = ("TRANSFER", "CASH_OUT")


def random_account_id(prefix):
    return f"{prefix}{RNG.integers(10_000_000, 99_999_999)}"


def generate_amount(txn_type):
    """Log-normal amounts, roughly matching real transaction scale by type."""
    if txn_type == "PAYMENT":
        return round(float(RNG.lognormal(mean=6.5, sigma=1.0)), 2)
    if txn_type == "DEBIT":
        return round(float(RNG.lognormal(mean=6.0, sigma=0.8)), 2)
    return round(float(RNG.lognormal(mean=9.5, sigma=1.6)), 2)  # CASH_IN/OUT/TRANSFER


def generate_legit_transaction(step):
    txn_type = RNG.choice(list(TYPE_PROBS.keys()), p=list(TYPE_PROBS.values()))
    amount = generate_amount(txn_type)

    old_balance_orig = round(float(RNG.lognormal(mean=9.0, sigma=1.5)), 2)
    old_balance_orig = max(old_balance_orig, amount)  # sender can actually afford it
    new_balance_orig = round(old_balance_orig - amount, 2)

    is_merchant_dest = txn_type == "PAYMENT"
    old_balance_dest = 0.0 if is_merchant_dest else round(float(RNG.lognormal(mean=8.0, sigma=1.5)), 2)
    new_balance_dest = 0.0 if is_merchant_dest else round(old_balance_dest + amount, 2)

    return {
        "step": step,
        "type": txn_type,
        "amount": amount,
        "nameOrig": random_account_id("C"),
        "oldbalanceOrg": old_balance_orig,
        "newbalanceOrig": new_balance_orig,
        "nameDest": random_account_id("M" if is_merchant_dest else "C"),
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "isFraud": 0,
    }


def generate_fraud_transaction(step, obvious):
    txn_type = RNG.choice(FRAUD_ELIGIBLE_TYPES)

    old_balance_orig = round(float(RNG.lognormal(mean=10.5 if obvious else 9.0, sigma=1.2)), 2)

    if obvious:
        # Classic account-takeover: drain 99-100% of the balance
        drain_ratio = RNG.uniform(0.99, 1.0)
    else:
        # Subtler: partial drain that won't trip the full_balance_drain rule,
        # but is still an unusual, ML-detectable pattern (large fraction of
        # a large balance moved in one shot, to an account with no history)
        drain_ratio = RNG.uniform(0.55, 0.85)

    amount = round(old_balance_orig * drain_ratio, 2)
    new_balance_orig = round(old_balance_orig - amount, 2)

    # Fraudulent CASH_OUT destinations are frequently unrecorded
    # (agent/ATM) — balance doesn't reflect the incoming funds
    if obvious:
        old_balance_dest = 0.0
        new_balance_dest = 0.0
    else:
        old_balance_dest = round(float(RNG.lognormal(mean=6.0, sigma=1.0)), 2)
        new_balance_dest = round(old_balance_dest + amount, 2)

    return {
        "step": step,
        "type": txn_type,
        "amount": amount,
        "nameOrig": random_account_id("C"),
        "oldbalanceOrg": old_balance_orig,
        "newbalanceOrig": new_balance_orig,
        "nameDest": random_account_id("C"),
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "isFraud": 1,
    }


def generate_dataset():
    n_fraud = int(N_TRANSACTIONS * FRAUD_RATE)
    n_legit = N_TRANSACTIONS - n_fraud

    rows = []

    for _ in range(n_legit):
        step = int(RNG.integers(1, N_STEPS + 1))
        rows.append(generate_legit_transaction(step))

    # Half of fraud cases obvious (rule-catchable), half subtle (ML-only)
    n_obvious = n_fraud // 2
    for i in range(n_fraud):
        step = int(RNG.integers(1, N_STEPS + 1))
        rows.append(generate_fraud_transaction(step, obvious=(i < n_obvious)))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = generate_dataset()
    output_path = "data/transactions.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df):,} transactions -> {output_path}")
    print(f"\nType distribution:\n{df['type'].value_counts(normalize=True).round(3)}")
    print(f"\nFraud count: {df['isFraud'].sum():,} ({df['isFraud'].mean():.3%})")
    print(f"\nFraud by type:\n{df[df['isFraud'] == 1]['type'].value_counts()}")
