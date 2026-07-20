"""
Data models for transactions flowing through the fraud detection engine.

These mirror the shape of the PaySim mobile-money dataset, since that's
what the ML model is trained on — so a real transaction sent to this API
and a row from that dataset look the same to the model.

A note on naming: the Python code below uses snake_case internally
(the standard, expected convention for Python — see PEP 8), but every
field is given a camelCase `alias` so the actual JSON API — what shows
up in /docs, what the frontend sends and receives — uses camelCase
throughout. This is the standard pattern for a Python backend serving a
JavaScript-friendly API contract: idiomatic Python inside, camelCase
JSON outside.
"""

from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic.alias_generators import to_camel


class TransactionType(str, Enum):
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    TRANSFER = "TRANSFER"
    PAYMENT = "PAYMENT"
    DEBIT = "DEBIT"


class CamelModel(BaseModel):
    """Base class: any model built on this automatically exposes camelCase
    field names in its JSON, while Python code still uses snake_case
    attributes. `populate_by_name=True` means the API also accepts
    snake_case on the way in, so nothing breaks if a client sends either."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class TransactionRequest(CamelModel):
    """A single transaction submitted for fraud screening."""

    step: int = Field(
        ..., ge=0,
        description="Time step in hours since simulation start (matches PaySim convention). "
                    "In a real system this would just be derived from the timestamp."
    )
    type: TransactionType
    amount: float = Field(..., gt=0, description="Transaction amount")

    name_orig: str = Field(..., description="Sender account identifier")
    old_balance_orig: float = Field(..., ge=0, description="Sender balance before transaction")
    new_balance_orig: float = Field(..., ge=0, description="Sender balance after transaction")

    name_dest: str = Field(..., description="Recipient account identifier")
    old_balance_dest: float = Field(0, ge=0, description="Recipient balance before transaction")
    new_balance_dest: float = Field(0, ge=0, description="Recipient balance after transaction")

    @field_validator("name_orig", "name_dest")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Account identifier cannot be blank")
        return v.strip()


class RuleFlag(CamelModel):
    """A single rule that fired during rule-based screening."""
    rule: str
    description: str
    severity: str  # "low" | "medium" | "high"


class FraudScreeningResult(CamelModel):
    """The full response returned for a screened transaction."""

    is_flagged: bool
    risk_score: float = Field(..., ge=0, le=1, description="0 = safe, 1 = certain fraud")
    risk_level: str  # "low" | "medium" | "high"

    rule_flags: list[RuleFlag]
    ml_score: float | None = Field(
        None, description="Model-predicted fraud probability, if the ML model is loaded"
    )

    explanation: str
