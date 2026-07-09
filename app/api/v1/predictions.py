from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from app.db.database import get_db
from app.core.dependencies import get_current_user
from app.schemas.prediction import TransactionInput, PredictionResponse
from app.services.fraud_scoring import fraud_scoring_service
from app.agents.investigation_agent import investigation_report_agent
from app.models.models import Transaction, Prediction, InvestigationReport, User

router = APIRouter(prefix="/api/v1", tags=["fraud-detection"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    transaction: TransactionInput,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Scores a transaction for fraud risk in real time.

    Flow: receive transaction -> engineer features -> model inference ->
    SHAP explanation -> persist transaction + prediction -> if flagged,
    generate a GenAI investigation report -> return score + explanation.
    """
    txn_dict = transaction.model_dump()
    result = fraud_scoring_service.predict(txn_dict)

    db_transaction = Transaction(
        step=transaction.step,
        type=transaction.type,
        amount=transaction.amount,
        name_orig=transaction.name_orig,
        oldbalance_org=transaction.oldbalance_org,
        newbalance_orig=transaction.oldbalance_org,
        name_dest=transaction.name_dest,
        oldbalance_dest=transaction.oldbalance_dest,
        newbalance_dest=transaction.oldbalance_dest,
        is_fraud_label=False,
    )
    db.add(db_transaction)
    await db.flush()

    db_prediction = Prediction(
        transaction_id=db_transaction.id,
        fraud_probability=result["fraud_probability"],
        model_version=result["model_version"],
        shap_values={
            f["feature"]: f["shap_contribution"] for f in result["top_contributing_features"]
        },
    )
    db.add(db_prediction)
    await db.flush()

    # Only spend an LLM call on transactions actually flagged.
    investigation_report_data = None
    if result["is_flagged"]:
        report = investigation_report_agent.generate_report(txn_dict, result)
        db_report = InvestigationReport(
            prediction_id=db_prediction.id,
            summary=report["summary"],
            risk_factors={
                "risk_level": report["risk_level"],
                "recommended_action": report["recommended_action"],
            },
        )
        db.add(db_report)
        investigation_report_data = report

    await db.commit()

    return PredictionResponse(
        prediction_id=str(db_prediction.id),
        investigation_report=investigation_report_data,
        **result,
    )


@router.get("/predictions/{prediction_id}/report")
async def get_investigation_report(
    prediction_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fetches the GenAI-generated investigation report for a given prediction,
    if one exists (only flagged transactions get a report generated).
    """
    result = await db.execute(
        select(InvestigationReport).where(
            InvestigationReport.prediction_id == uuid.UUID(prediction_id)
        )
    )
    report = result.scalar_one_or_none()

    if report is None:
        return {"detail": "No investigation report exists for this prediction (not flagged, or not yet generated)."}

    return {
        "summary": report.summary,
        "risk_level": report.risk_factors.get("risk_level"),
        "recommended_action": report.risk_factors.get("recommended_action"),
        "generated_at": report.generated_at,
    }