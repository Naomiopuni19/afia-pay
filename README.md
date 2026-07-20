# Afia Pay : Mobile Money Fraud Detection API

A FastAPI service that screens mobile money transactions for fraud, combining
**rule-based detection** (fast, explainable, auditable) with a **trained
machine learning model** trained on the [PaySim
dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)  a synthetic
dataset modeled specifically on mobile money systems like M-Pesa.

Built as a portfolio project demonstrating fintech fraud detection concepts:
why real systems layer rules and ML together, how to design an explainable
API, and how to structure a production-shaped FastAPI service.

## API design: camelCase JSON, idiomatic Python underneath

The Python code follows standard Python convention (snake_case ; see
[PEP 8](https://peps.python.org/pep-0008/)) throughout the codebase, but
the actual JSON API request bodies, responses, everything in `/docs` 
uses camelCase (`nameOrig`, `riskScore`, etc.), matching typical JavaScript/
frontend API conventions. This is handled with Pydantic's `alias_generator`
(see `app/models/transaction.py`), not by renaming Python variables  a
Python reviewer sees proper snake_case, an API consumer sees camelCase.
The API also still accepts snake_case input for backwards compatibility.

## Why rules *and* ML?

Rules catch known fraud patterns instantly and can be explained to a
compliance officer in one sentence ("flag any transaction that drains 99%+
of an account's balance"). ML catches subtler patterns a human wouldn't
think to write a rule for, but is harder to audit. Real fraud systems use
both rules as a fast, transparent first line, ML as a second layer for
what rules miss. This project deliberately keeps the two systems as
separate, swappable pieces (see `app/services/rule_engine.py` and
`app/services/ml_model.py`) that get combined in `screening_service.py`.

## Project status

- ✅ **Phase 1** — FastAPI skeleton, 5 rule-based fraud checks, full test suite
- ✅ **Phase 2** — Trained ML model (RandomForest), wired into the API alongside the rules
- ✅ **Phase 3** — Visual demo frontend (see below)
- ✅ **Phase 4** — Deployed to Render (see below)

## Deployment (Render)

This API is deployed on [Render](https://render.com)'s free tier 
straightforward for a Python/FastAPI service, unlike serverless platforms
like Vercel which don't fit a service carrying a trained ML model well.

**To deploy your own copy:**

1. Push this project to its own GitHub repository (separate from any other
   project  Afia Pay isn't part of the bakery site's repo)
2. Go to [render.com](https://render.com) → sign up with GitHub
3. **New** → **Web Service** → connect the repository
4. Render should auto-detect the `render.yaml` blueprint in this repo and
   fill in the build/start commands automatically. If not, set manually:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service** first deploy takes a few minutes
6. Once live, your API is reachable at `https://your-service-name.onrender.com`
7. Visit `https://your-service-name.onrender.com/docs` to confirm it's working
8. **Update the demo frontend** — open `frontend/index.html`, find the
   `apiUrl` constant near the top of the `<script>` section, and change it
   from `http://127.0.0.1:8000/api/v1/screen` to your live Render URL

**Note on Render's free tier:** the service spins down after 15 minutes of
inactivity and takes ~30-60 seconds to wake back up on the next request 
completely normal for a free tier, just means the very first request after
a period of inactivity will feel slow. Fine for a portfolio piece; a paid
tier removes this if it's ever needed for something more serious.

## Trying the visual demo

1. Start the API server (see "Running locally" below) and leave it running
2. Open `frontend/index.html` directly in your browser (just double-click it)
3. Use the "Normal" / "Obvious Fraud" / "Subtle Fraud" preset buttons to fill in
   a realistic example, or type in your own transaction
4. Click "Scan Transaction" and watch the security seal stamp the verdict 
   green "CLEARED" or red "FLAGGED," with the exact rule flags and ML score
   that fired

This is a single static HTML file (no build step, no separate frontend
server) that calls your locally-running API directly  the API's CORS
settings already allow this.

### A note on the training data

The model is trained on a **synthetic dataset generated to mirror the real
PaySim dataset's statistical structure** (transaction type mix, fraud rate,
fraud-only-in-TRANSFER/CASH_OUT pattern)  not the actual downloaded PaySim
file. See `scripts/generate_dataset.py` for exactly how and why each
statistic was chosen to match the real dataset's published characteristics.

### Model results (on a held-out test set, ~0.15% fraud rate)

| Metric | Score | What it means |
|---|---|---|
| ROC-AUC | 0.996 | Near-perfect at ranking fraud above legitimate transactions |
| Recall (fraud) | 82% | Catches the large majority of real fraud in the test set |
| Precision (fraud) | 14% | Most flags turn out to be false alarms  see below |

That precision number is not a flaw  it's the realistic trade-off for
extremely rare-event detection. With fraud this uncommon, prioritizing
recall (catching real fraud) necessarily means accepting more false
positives, since missing fraud is typically far costlier than a false
alarm that gets manually reviewed and cleared. A model tuned for high
precision instead would catch almost no fraud at all.

**Concrete proof the ML layer adds value beyond the rules:** a transaction
that drains 70% of an account's balance (below the rules' 99%
`full_balance_drain` threshold, so **zero rules fire**) still gets flagged
by the model with a 77% fraud probability a pattern the rule engine
alone would have completely missed.

## Running locally

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Then visit `http://localhost:8000/docs` for interactive API documentation
(auto-generated by FastAPI).

## Running tests

```bash
python -m pytest tests/ -v
```

## Regenerating the dataset / retraining the model

The trained model is included in `models/fraud_model.joblib`, so you
don't need to do this to run the API  but if you want to regenerate
the data or retrain from scratch:

```bash
python scripts/generate_dataset.py   # -> data/transactions.csv
python scripts/train_model.py        # -> models/fraud_model.joblib
```

## Example request

```bash
curl -X POST http://localhost:8000/api/v1/screen \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1, "type": "CASH_OUT", "amount": 499999,
    "name_orig": "C111", "old_balance_orig": 500000, "new_balance_orig": 1,
    "name_dest": "C222", "old_balance_dest": 0, "new_balance_dest": 0
  }'
```

## Project structure

```
app/
  main.py                    FastAPI app entry point
  models/transaction.py      Pydantic request/response schemas
  routers/screening.py       API routes
  core/features.py           Feature engineering (shared by training + live scoring)
  services/
    rule_engine.py           5 explainable fraud detection rules
    ml_model.py               Loads the trained model, scores real requests
    screening_service.py     Combines rules + ML into a final risk score
scripts/
  generate_dataset.py        Builds the synthetic PaySim-style dataset
  train_model.py              Trains and evaluates the fraud model
data/
  transactions.csv           Generated training data
models/
  fraud_model.joblib         The trained, saved model
tests/
  test_rule_engine.py        Unit tests for each rule
  test_api.py                Integration tests for the API
```

## The five rules (Phase 1)

| Rule | What it catches |
|---|---|
| `full_balance_drain` | Sender empties 99%+ of their balance in one transaction  classic account takeover |
| `balance_inconsistency` | Reported balances don't match the transaction amount |
| `dest_balance_unchanged` | Recipient balance doesn't move despite a nonzero transfer  cash-out through an unrecorded channel |
| `large_amount` | Transaction exceeds a configurable large-amount threshold |
| `self_transfer` | Sender and recipient are the same account |
