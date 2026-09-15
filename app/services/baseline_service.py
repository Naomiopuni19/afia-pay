from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models.db_models import Transaction, UserBaseline


def get_or_create_baseline(db: Session, user_id: int) -> UserBaseline:
    baseline = db.query(UserBaseline).filter(UserBaseline.user_id == user_id).first()
    if baseline is None:
        baseline = UserBaseline(user_id=user_id)
        db.add(baseline)
        db.commit()
        db.refresh(baseline)
    return baseline


def check_against_baseline(
    db: Session, user_id: int, recipient: str, amount: float, now: datetime
) -> list[dict]:
    """Compares one transaction against the user's own learned history.

    Returns a list of personal flags, empty if it looks like the user's
    usual behavior. Needs at least a few past transactions before it
    trusts the baseline enough to flag anything on amount or hour, so
    early transactions all pass through clean to build up history.
    """
    baseline = get_or_create_baseline(db, user_id)
    flags = []

    known_recipients = set(
        r for r in (baseline.known_recipients or "").split("|") if r
    )
    is_new_recipient = recipient not in known_recipients

    if baseline.txn_count >= 3:
        if amount > baseline.avg_amount * 3:
            flags.append({
                "rule": "personal_amount_anomaly",
                "description": f"Amount is {amount / baseline.avg_amount:.1f}x your typical transaction",
                "severity": "high" if amount > baseline.avg_amount * 5 else "medium",
            })

        if is_new_recipient and amount > baseline.avg_amount * 1.5:
            flags.append({
                "rule": "personal_new_recipient_high_amount",
                "description": "First-ever transfer to this recipient, above your usual amount",
                "severity": "medium",
            })

    if baseline.txn_count >= 5:
        hour = now.hour
        if hour < baseline.typical_hour_start or hour > baseline.typical_hour_end:
            flags.append({
                "rule": "personal_odd_hour",
                "description": f"Sent at {hour}:00, outside your usual {baseline.typical_hour_start}:00 to {baseline.typical_hour_end}:00 window",
                "severity": "medium",
            })

    recent_cutoff = now - timedelta(minutes=10)
    recent_count = (
        db.query(Transaction)
        .filter(Transaction.user_id == user_id, Transaction.timestamp >= recent_cutoff)
        .count()
    )
    if recent_count >= 4:
        flags.append({
            "rule": "personal_velocity",
            "description": f"{recent_count} transactions in the last 10 minutes, well above your normal pace",
            "severity": "high",
        })

    same_recipient_recent = (
        db.query(Transaction)
        .filter(
            Transaction.user_id == user_id,
            Transaction.recipient == recipient,
            Transaction.timestamp >= recent_cutoff,
        )
        .count()
    )
    if same_recipient_recent >= 5:
        flags.append({
            "rule": "personal_same_recipient_velocity",
            "description": f"{same_recipient_recent} transfers to {recipient} in the last 10 minutes",
            "severity": "high",
        })

    insufficient_cutoff = now - timedelta(minutes=15)
    repeated_insufficient = (
        db.query(Transaction)
        .filter(
            Transaction.user_id == user_id,
            Transaction.recipient == recipient,
            Transaction.insufficient_funds == True,
            Transaction.timestamp >= insufficient_cutoff,
        )
        .count()
    )
    if repeated_insufficient >= 5:
        flags.append({
            "rule": "personal_repeated_insufficient_attempts",
            "description": f"{repeated_insufficient} failed insufficient-balance attempts to {recipient} in the last 15 minutes, looks like probing",
            "severity": "high",
        })

    return flags


def update_baseline(db: Session, user_id: int, recipient: str, amount: float, now: datetime) -> None:
    """Rolls one more transaction into the user's baseline, whether or not
    it was flagged. A blocked attempt still tells us about their intent,
    and normal accepted transactions are what shapes the running average."""
    baseline = get_or_create_baseline(db, user_id)

    known_recipients = set(
        r for r in (baseline.known_recipients or "").split("|") if r
    )
    known_recipients.add(recipient)

    total_before = baseline.avg_amount * baseline.txn_count
    baseline.txn_count += 1
    baseline.avg_amount = (total_before + amount) / baseline.txn_count
    baseline.max_amount = max(baseline.max_amount, amount)
    baseline.known_recipients = "|".join(known_recipients)

    hour = now.hour
    if baseline.txn_count == 1:
        baseline.typical_hour_start = max(0, hour - 2)
        baseline.typical_hour_end = min(23, hour + 2)
    else:
        baseline.typical_hour_start = min(baseline.typical_hour_start, max(0, hour - 1))
        baseline.typical_hour_end = max(baseline.typical_hour_end, min(23, hour + 1))

    baseline.updated_at = now
    db.commit()