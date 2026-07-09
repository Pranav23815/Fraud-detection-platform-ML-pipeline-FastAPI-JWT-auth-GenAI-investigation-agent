import uuid
from datetime import datetime
from sqlalchemy import String, Float, DateTime, ForeignKey, Boolean, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Transaction(Base):
    """
    Raw transaction record — mirrors the PaySim schema so ingestion
    is a near 1:1 mapping from the CSV, with a few production-style
    additions (id, ingested_at) that the raw dataset wouldn't have.
    """
    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    step: Mapped[int] = mapped_column(Integer)  # PaySim's simulated time unit (1 step = 1 hour)
    type: Mapped[str] = mapped_column(String(20))  # CASH_OUT, TRANSFER, PAYMENT, etc.
    amount: Mapped[float] = mapped_column(Float)
    name_orig: Mapped[str] = mapped_column(String(50), index=True)
    oldbalance_org: Mapped[float] = mapped_column(Float)
    newbalance_orig: Mapped[float] = mapped_column(Float)
    name_dest: Mapped[str] = mapped_column(String(50), index=True)
    oldbalance_dest: Mapped[float] = mapped_column(Float)
    newbalance_dest: Mapped[float] = mapped_column(Float)
    is_fraud_label: Mapped[bool] = mapped_column(Boolean, default=False)  # ground truth, if known
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prediction: Mapped["Prediction"] = relationship(back_populates="transaction", uselist=False)


class Prediction(Base):
    """
    Model output for a transaction, stored separately from the raw
    transaction so we can version predictions (re-score old transactions
    with a new model) without mutating source data.
    """
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("transactions.id"), unique=True)
    fraud_probability: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(50))
    shap_values: Mapped[dict] = mapped_column(JSONB, nullable=True)  # top feature contributions
    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    transaction: Mapped["Transaction"] = relationship(back_populates="prediction")
    report: Mapped["InvestigationReport"] = relationship(back_populates="prediction", uselist=False)


class InvestigationReport(Base):
    """
    GenAI-generated human-readable summary of why a transaction was
    flagged. Kept as a separate table so the LLM layer can be swapped
    or re-run independently of the ML scoring layer.
    """
    __tablename__ = "investigation_reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("predictions.id"), unique=True)
    summary: Mapped[str] = mapped_column(Text)
    risk_factors: Mapped[dict] = mapped_column(JSONB, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prediction: Mapped["Prediction"] = relationship(back_populates="report")
