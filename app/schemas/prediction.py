from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    step: int = Field(..., description="Simulated time unit (1 step = 1 hour)")
    type: str = Field(..., description="Transaction type: PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT")
    amount: float
    name_orig: str
    oldbalance_org: float
    name_dest: str
    oldbalance_dest: float

    class Config:
        json_schema_extra = {
            "example": {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "name_orig": "C1231006815",
                "oldbalance_org": 181.0,
                "name_dest": "C1666544295",
                "oldbalance_dest": 0.0,
            }
        }


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class PredictionResponse(BaseModel):
    prediction_id: str
    fraud_probability: float
    is_flagged: bool
    model_version: str
    top_contributing_features: list[FeatureContribution]
    investigation_report: dict | None = None