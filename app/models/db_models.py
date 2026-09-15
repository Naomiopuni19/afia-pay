from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    wallet = relationship("Wallet", back_populates="owner", uselist=False)
    transactions = relationship("Transaction", back_populates="owner")


class Wallet(Base):
    __tablename__ = "wallets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    balance = Column(Float, default=5000.0)

    owner = relationship("User", back_populates="wallet")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    recipient = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    risk_score = Column(Float, nullable=True)
    risk_level = Column(String, nullable=True)
    is_flagged = Column(Boolean, default=False)
    explanation = Column(String, nullable=True)
    insufficient_funds = Column(Boolean, default=False)

    owner = relationship("User", back_populates="transactions")


class UserBaseline(Base):
    __tablename__ = "user_baselines"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    avg_amount = Column(Float, default=0.0)
    max_amount = Column(Float, default=0.0)
    txn_count = Column(Integer, default=0)
    known_recipients = Column(String, default="")
    typical_hour_start = Column(Integer, default=6)
    typical_hour_end = Column(Integer, default=22)
    updated_at = Column(DateTime, default=datetime.utcnow)
