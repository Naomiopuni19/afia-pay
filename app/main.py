from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import screening

app = FastAPI(
    title="Afia Pay Fraud Detection API",
    description=(
        "A mobile money fraud screening API combining rule-based detection "
        "with a machine learning model trained on transaction patterns. "
        "Built as a portfolio project demonstrating fintech fraud detection concepts."
    ),
    version="0.1.0",
)

# Permissive CORS for portfolio/demo purposes — in a real production
# deployment this would be locked down to specific known origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(screening.router)


@app.get("/", tags=["health"])
def root():
    return {
        "service": "Afia Pay Fraud Detection API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}
