"""
Orchestrates fraud screening: runs the rule engine, and (once Phase 2 adds
the trained model) blends in the ML score too.

Kept deliberately separate from the rule engine itself so that swapping in
the ML model later doesn't require touching rule logic at all.
"""

from app.models.transaction import TransactionRequest, FraudScreeningResult
from app.services import rule_engine
from app.services.ml_model import get_ml_score  # Phase 2 — safe no-op until trained


RISK_LEVEL_THRESHOLDS = [
    (0.7, "high"),
    (0.4, "medium"),
    (0.0, "low"),
]


def _risk_level(score: float) -> str:
    for threshold, level in RISK_LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return "low"


def _combine_scores(rule_score: float, ml_score: float | None) -> float:
    """
    Blends the rule-based score with the ML score, when available.

    Weighted toward the ML model once it's trained (it's generally the
    stronger signal), but rules still contribute — a transaction that
    trips an obvious rule shouldn't score low just because the model
    disagrees, since rules encode hard business knowledge the model
    might not have learned from historical data alone.
    """
    if ml_score is None:
        return rule_score
    return round((0.4 * rule_score) + (0.6 * ml_score), 4)


def _build_explanation(flags, risk_level: str, ml_score: float | None) -> str:
    if not flags and ml_score is None:
        return "No rule violations detected and no ML model loaded yet — transaction appears normal based on available checks."

    if not flags and ml_score is not None:
        return f"No rules triggered, but the ML model assigned a fraud probability of {ml_score:.1%}."

    rule_summary = "; ".join(f"{f.rule} ({f.severity})" for f in flags)
    return f"Flagged by {len(flags)} rule(s): {rule_summary}. Overall risk assessed as {risk_level}."


def screen_transaction(txn: TransactionRequest) -> FraudScreeningResult:
    """Runs a transaction through the full fraud screening pipeline."""

    flags = rule_engine.run_rules(txn)
    rule_score = rule_engine.rule_based_score(flags)

    ml_score = get_ml_score(txn)  # Returns None until Phase 2's model is trained/loaded

    final_score = _combine_scores(rule_score, ml_score)
    risk_level = _risk_level(final_score)

    return FraudScreeningResult(
        is_flagged=final_score >= 0.4,
        risk_score=final_score,
        risk_level=risk_level,
        rule_flags=flags,
        ml_score=ml_score,
        explanation=_build_explanation(flags, risk_level, ml_score),
    )
