"""
Rule-based fraud detection.

These rules encode patterns that are well-known red flags in mobile money
fraud (account takeover, SIM-swap fraud, cash-out mule accounts, balance
manipulation). They're intentionally simple and explainable — each one
should be describable to a non-technical person in one sentence, which is
exactly why real fraud systems keep a rules layer even after adding ML:
rules are auditable and instant, while ML catches what rules miss.
"""

from app.models.transaction import TransactionRequest, TransactionType, RuleFlag


# Thresholds are deliberately named constants, not magic numbers, so they
# can be tuned later (or made configurable per-deployment) without hunting
# through the logic itself.
LARGE_TRANSACTION_THRESHOLD = 200_000.0
FULL_BALANCE_DRAIN_RATIO = 0.99  # sending out >=99% of account balance
DEST_BALANCE_UNCHANGED_TOLERANCE = 0.01


def check_full_balance_drain(txn: TransactionRequest) -> RuleFlag | None:
    """
    Flags transactions that empty (or nearly empty) the sender's account.

    Legitimate users rarely drain an account to zero in one transaction;
    this is a classic signature of account takeover, where a fraudster
    moves the entire balance out before the real owner notices.
    """
    if txn.old_balance_orig <= 0:
        return None

    drained_ratio = (txn.old_balance_orig - txn.new_balance_orig) / txn.old_balance_orig

    if drained_ratio >= FULL_BALANCE_DRAIN_RATIO and txn.type in (
        TransactionType.CASH_OUT,
        TransactionType.TRANSFER,
    ):
        return RuleFlag(
            rule="full_balance_drain",
            description=(
                f"Transaction empties {drained_ratio:.0%} of the sender's balance "
                f"in a single {txn.type.value.lower()} — a common account-takeover pattern."
            ),
            severity="high",
        )
    return None


def check_balance_inconsistency(txn: TransactionRequest) -> RuleFlag | None:
    """
    Flags transactions where the sender's reported before/after balances
    don't match the transaction amount.

    In a legitimate system these numbers are always internally consistent
    (old_balance - amount = new_balance). A mismatch usually means either
    a client bug, or — more concerning — an attempt to manipulate balance
    figures to hide the true size of a fraudulent transfer.
    """
    expected_new_balance = txn.old_balance_orig - txn.amount
    discrepancy = abs(expected_new_balance - txn.new_balance_orig)

    # Allow a small tolerance for floating point / fee rounding
    if discrepancy > max(1.0, txn.amount * 0.01):
        return RuleFlag(
            rule="balance_inconsistency",
            description=(
                f"Sender's reported balance doesn't match the transaction amount "
                f"(expected new balance ≈ {expected_new_balance:,.2f}, "
                f"got {txn.new_balance_orig:,.2f})."
            ),
            severity="medium",
        )
    return None


def check_dest_balance_unchanged(txn: TransactionRequest) -> RuleFlag | None:
    """
    Flags CASH_OUT/TRANSFER transactions where the recipient's balance
    doesn't move at all.

    This is a known PaySim/mobile-money fraud signature: fraudulent
    CASH_OUT transactions are often routed through "merchant" accounts
    that don't actually reflect the incoming funds in their tracked
    balance, because the money is being withdrawn through an ATM or agent
    network rather than credited to a real recipient wallet.
    """
    if txn.type not in (TransactionType.CASH_OUT, TransactionType.TRANSFER):
        return None

    if txn.old_balance_dest == 0 and txn.new_balance_dest == 0 and txn.amount > 0:
        return RuleFlag(
            rule="dest_balance_unchanged",
            description=(
                "Recipient's balance shows no change despite a nonzero transfer amount — "
                "consistent with cash-out through an unrecorded agent/ATM channel."
            ),
            severity="medium",
        )
    return None


def check_large_amount(txn: TransactionRequest) -> RuleFlag | None:
    """
    Flags unusually large transactions for extra scrutiny.

    This isn't proof of fraud on its own (plenty of large transactions are
    legitimate), but it's a standard trigger for step-up verification in
    real systems — hence "low" severity rather than "high".
    """
    if txn.amount >= LARGE_TRANSACTION_THRESHOLD:
        return RuleFlag(
            rule="large_amount",
            description=(
                f"Transaction amount ({txn.amount:,.2f}) exceeds the large-transaction "
                f"threshold ({LARGE_TRANSACTION_THRESHOLD:,.2f}) and warrants extra scrutiny."
            ),
            severity="low",
        )
    return None


def check_self_transfer(txn: TransactionRequest) -> RuleFlag | None:
    """
    Flags transactions where sender and recipient are the same account.

    Self-transfers are sometimes used to test whether a compromised
    account is usable before attempting a larger fraudulent transaction,
    or to artificially inflate transaction history.
    """
    if txn.name_orig == txn.name_dest:
        return RuleFlag(
            rule="self_transfer",
            description="Sender and recipient are the same account.",
            severity="low",
        )
    return None


# All rules run for every transaction, in this order. Order matters only
# for readability of the resulting flag list, not for scoring.
ALL_RULES = [
    check_full_balance_drain,
    check_balance_inconsistency,
    check_dest_balance_unchanged,
    check_large_amount,
    check_self_transfer,
]


def run_rules(txn: TransactionRequest) -> list[RuleFlag]:
    """Runs every rule against a transaction and returns the flags that fired."""
    flags = []
    for rule_fn in ALL_RULES:
        result = rule_fn(txn)
        if result is not None:
            flags.append(result)
    return flags


# Severity -> numeric weight, used to combine multiple rule flags into a
# single 0-1 risk contribution from the rules layer.
SEVERITY_WEIGHT = {"low": 0.15, "medium": 0.35, "high": 0.6}


def rule_based_score(flags: list[RuleFlag]) -> float:
    """
    Combines fired rule flags into a single risk score between 0 and 1.

    Uses a "noisy-OR" style combination rather than a simple sum, so that
    multiple flags push the score up but it never exceeds 1 — this avoids
    the score becoming meaningless when many low-severity rules fire at once.
    """
    if not flags:
        return 0.0

    survival_probability = 1.0
    for flag in flags:
        weight = SEVERITY_WEIGHT.get(flag.severity, 0.15)
        survival_probability *= (1 - weight)

    return round(1 - survival_probability, 4)
