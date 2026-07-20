from fastapi import APIRouter

from app.models.transaction import TransactionRequest, FraudScreeningResult
from app.services.screening_service import screen_transaction

router = APIRouter(prefix="/api/v1", tags=["screening"])


@router.post("/screen", response_model=FraudScreeningResult)
def screen(transaction: TransactionRequest) -> FraudScreeningResult:
    """
    Screens a single transaction for fraud indicators.

    Runs the transaction through the rule engine (and the ML model, once
    trained) and returns a combined risk assessment.
    """
    return screen_transaction(transaction)
