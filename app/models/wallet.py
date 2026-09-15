from datetime import datetime
from pydantic import BaseModel, Field


class SendMoneyRequest(BaseModel):
    recipient: str = Field(..., min_length=1, description="Recipient name or number")
    amount: float = Field(..., gt=0)


class TransactionOut(BaseModel):
    id: int
    recipient: str
    amount: float
    timestamp: datetime
    risk_score: float | None
    risk_level: str | None
    is_flagged: bool
    explanation: str | None

    class Config:
        from_attributes = True


class SendMoneyResponse(BaseModel):
    transaction: TransactionOut
    wallet_balance: float
    decision: str  # "allow" | "review" | "block"

class BaselineOut(BaseModel):
    txn_count: int
    avg_amount: float
    known_recipient_count: int
    typical_hour_start: int
    typical_hour_end: int
