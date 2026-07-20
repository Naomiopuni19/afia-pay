import pytest
from app.models.transaction import TransactionRequest, TransactionType
from app.services import rule_engine


def make_txn(**overrides):
    defaults = dict(
        step=1,
        type=TransactionType.TRANSFER,
        amount=1000.0,
        name_orig="C111",
        old_balance_orig=5000.0,
        new_balance_orig=4000.0,
        name_dest="C222",
        old_balance_dest=0.0,
        new_balance_dest=0.0,
    )
    defaults.update(overrides)
    return TransactionRequest(**defaults)


def test_normal_transaction_flags_nothing():
    txn = make_txn(
        old_balance_orig=5000, new_balance_orig=4000,
        old_balance_dest=1000, new_balance_dest=2000,
    )
    flags = rule_engine.run_rules(txn)
    # dest_balance_unchanged shouldn't fire since dest balance did change
    rule_names = [f.rule for f in flags]
    assert "dest_balance_unchanged" not in rule_names
    assert "balance_inconsistency" not in rule_names


def test_full_balance_drain_detected():
    txn = make_txn(
        type=TransactionType.CASH_OUT,
        amount=4999,
        old_balance_orig=5000,
        new_balance_orig=1,  # 99.98% drained
    )
    flags = rule_engine.run_rules(txn)
    assert any(f.rule == "full_balance_drain" for f in flags)


def test_balance_inconsistency_detected():
    txn = make_txn(
        amount=1000,
        old_balance_orig=5000,
        new_balance_orig=4900,  # should be 4000, off by 900
    )
    flags = rule_engine.run_rules(txn)
    assert any(f.rule == "balance_inconsistency" for f in flags)


def test_dest_balance_unchanged_detected():
    txn = make_txn(
        type=TransactionType.CASH_OUT,
        old_balance_dest=0,
        new_balance_dest=0,
    )
    flags = rule_engine.run_rules(txn)
    assert any(f.rule == "dest_balance_unchanged" for f in flags)


def test_large_amount_detected():
    txn = make_txn(
        amount=250_000,
        old_balance_orig=500_000,
        new_balance_orig=250_000,
    )
    flags = rule_engine.run_rules(txn)
    assert any(f.rule == "large_amount" for f in flags)


def test_self_transfer_detected():
    txn = make_txn(name_orig="C111", name_dest="C111")
    flags = rule_engine.run_rules(txn)
    assert any(f.rule == "self_transfer" for f in flags)


def test_rule_based_score_empty_flags_is_zero():
    assert rule_engine.rule_based_score([]) == 0.0


def test_rule_based_score_never_exceeds_one():
    txn = make_txn(
        type=TransactionType.CASH_OUT,
        amount=500_000,
        old_balance_orig=500_000,
        new_balance_orig=0,
        old_balance_dest=0,
        new_balance_dest=0,
        name_orig="C111",
        name_dest="C111",
    )
    flags = rule_engine.run_rules(txn)
    score = rule_engine.rule_based_score(flags)
    assert 0 <= score <= 1
