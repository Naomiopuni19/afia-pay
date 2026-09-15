from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.core.deps import get_current_user
from app.models.db_models import User, Transaction, Wallet
from app.models.wallet import SendMoneyRequest, SendMoneyResponse, TransactionOut, BaselineOut
from app.models.transaction import TransactionRequest, TransactionType
from app.services.screening_service import screen_transaction
from app.services.baseline_service import check_against_baseline, update_baseline

router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])

DECISION_BY_LEVEL = {"high": "block", "medium": "review", "low": "allow"}
SEVERITY_WEIGHT = {"high": 0.35, "medium": 0.2, "low": 0.1}


def _personal_score(flags: list[dict]) -> float:
    return min(1.0, sum(SEVERITY_WEIGHT.get(f["severity"], 0.1) for f in flags))


@router.post("/send", response_model=SendMoneyResponse)
def send_money(
    payload: SendMoneyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    wallet = db.query(Wallet).filter(Wallet.user_id == current_user.id).first()
    if wallet is None:
        raise HTTPException(status_code=400, detail="No wallet found for this account")

    now = datetime.utcnow()

    if payload.amount > wallet.balance:
        # Log the failed attempt itself, since a burst of these to the same
        # recipient is a probing pattern worth catching, not just a normal
        # user error to silently ignore.
        failed_txn = Transaction(
            user_id=current_user.id,
            recipient=payload.recipient,
            amount=payload.amount,
            timestamp=now,
            risk_score=None,
            risk_level=None,
            is_flagged=False,
            explanation="Rejected: insufficient balance for this amount.",
            insufficient_funds=True,
        )
        db.add(failed_txn)
        db.commit()
        raise HTTPException(status_code=400, detail="Insufficient balance")

    global_request = TransactionRequest(
        step=int(now.timestamp() // 3600),
        type=TransactionType.TRANSFER,
        amount=payload.amount,
        name_orig=f"user_{current_user.id}",
        old_balance_orig=wallet.balance,
        new_balance_orig=wallet.balance - payload.amount,
        name_dest=payload.recipient,
        old_balance_dest=0,
        new_balance_dest=payload.amount,
    )
    global_result = screen_transaction(global_request)

    personal_flags = check_against_baseline(
        db, current_user.id, payload.recipient, payload.amount, now
    )
    personal_score = _personal_score(personal_flags)

    final_score = max(global_result.risk_score, personal_score)
    if final_score >= 0.7:
        risk_level = "high"
    elif final_score >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"
    decision = DECISION_BY_LEVEL[risk_level]
    is_flagged = decision != "allow"

    explanation_parts = [global_result.explanation]
    if personal_flags:
        summary = "; ".join(f"{f['rule']} ({f['severity']})" for f in personal_flags)
        explanation_parts.append(f"Personal baseline flags: {summary}.")
    else:
        explanation_parts.append("No deviation from your usual behavior.")
    explanation = " ".join(explanation_parts)

    txn = Transaction(
        user_id=current_user.id,
        recipient=payload.recipient,
        amount=payload.amount,
        timestamp=now,
        risk_score=round(final_score, 4),
        risk_level=risk_level,
        is_flagged=is_flagged,
        explanation=explanation,
        insufficient_funds=False,
    )
    db.add(txn)

    if decision != "block":
        wallet.balance -= payload.amount

    update_baseline(db, current_user.id, payload.recipient, payload.amount, now)

    db.commit()
    db.refresh(txn)
    db.refresh(wallet)

    return SendMoneyResponse(
        transaction=TransactionOut.model_validate(txn),
        wallet_balance=wallet.balance,
        decision=decision,
    )


@router.get("/history", response_model=list[TransactionOut])
def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    txns = (
        db.query(Transaction)
        .filter(Transaction.user_id == current_user.id)
        .order_by(Transaction.timestamp.desc())
        .all()
    )
    return [TransactionOut.model_validate(t) for t in txns]

@router.get("/baseline", response_model=BaselineOut)
def get_baseline(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from app.services.baseline_service import get_or_create_baseline
    baseline = get_or_create_baseline(db, current_user.id)
    known = [r for r in (baseline.known_recipients or "").split("|") if r]
    return BaselineOut(
        txn_count=baseline.txn_count,
        avg_amount=round(baseline.avg_amount, 2),
        known_recipient_count=len(known),
        typical_hour_start=baseline.typical_hour_start,
        typical_hour_end=baseline.typical_hour_end,
    )
