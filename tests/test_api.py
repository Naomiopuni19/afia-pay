from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_screen_normal_transaction():
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 500,
        "nameOrig": "C111",
        "oldBalanceOrig": 5000,
        "newBalanceOrig": 4500,
        "nameDest": "M999",
        "oldBalanceDest": 1000,
        "newBalanceDest": 1500,
    }
    response = client.post("/api/v1/screen", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "riskScore" in data
    assert "isFlagged" in data
    assert data["riskLevel"] in ("low", "medium", "high")


def test_screen_suspicious_transaction_flags_high_risk():
    payload = {
        "step": 1,
        "type": "CASH_OUT",
        "amount": 499_999,
        "nameOrig": "C111",
        "oldBalanceOrig": 500_000,
        "newBalanceOrig": 1,
        "nameDest": "C222",
        "oldBalanceDest": 0,
        "newBalanceDest": 0,
    }
    response = client.post("/api/v1/screen", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["isFlagged"] is True
    assert data["riskScore"] > 0.4
    assert len(data["ruleFlags"]) >= 2  # should trip full_balance_drain + dest_balance_unchanged + large_amount


def test_screen_rejects_invalid_amount():
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": -100,  # invalid — amount must be > 0
        "nameOrig": "C111",
        "oldBalanceOrig": 5000,
        "newBalanceOrig": 4500,
        "nameDest": "M999",
    }
    response = client.post("/api/v1/screen", json=payload)
    assert response.status_code == 422  # validation error


def test_screen_rejects_missing_fields():
    response = client.post("/api/v1/screen", json={"amount": 500})
    assert response.status_code == 422


def test_screen_still_accepts_snake_case_for_compatibility():
    """The API should accept EITHER camelCase or snake_case on input
    (populate_by_name=True), even though it always outputs camelCase."""
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 500,
        "name_orig": "C111",
        "old_balance_orig": 5000,
        "new_balance_orig": 4500,
        "name_dest": "M999",
        "old_balance_dest": 1000,
        "new_balance_dest": 1500,
    }
    response = client.post("/api/v1/screen", json=payload)
    assert response.status_code == 200
    assert "riskScore" in response.json()  # output is still camelCase either way
