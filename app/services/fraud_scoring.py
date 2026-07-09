from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "fraud_model_v1.json"
MODEL_VERSION = "fraud_model_v1"

FEATURE_COLUMNS = [
    "step", "amount", "log_amount", "oldbalance_org", "oldbalance_dest",
    "amount_to_balance_ratio", "dest_is_merchant",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]

FLAG_THRESHOLD = 0.5


class FraudScoringService:
    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(MODEL_PATH))
        self.explainer = shap.TreeExplainer(self.model)

    def _engineer_features(self, transaction: dict) -> pd.DataFrame:
        amount = transaction["amount"]
        oldbalance_org = transaction["oldbalance_org"]
        oldbalance_dest = transaction["oldbalance_dest"]
        txn_type = transaction["type"]

        row = {
            "step": transaction["step"],
            "amount": amount,
            "log_amount": np.log1p(amount),
            "oldbalance_org": oldbalance_org,
            "oldbalance_dest": oldbalance_dest,
            "amount_to_balance_ratio": amount / (oldbalance_org + 1),
            "dest_is_merchant": int(transaction["name_dest"].startswith("M")),
            "type_CASH_IN": int(txn_type == "CASH_IN"),
            "type_CASH_OUT": int(txn_type == "CASH_OUT"),
            "type_DEBIT": int(txn_type == "DEBIT"),
            "type_PAYMENT": int(txn_type == "PAYMENT"),
            "type_TRANSFER": int(txn_type == "TRANSFER"),
        }
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict(self, transaction: dict) -> dict:
        features_df = self._engineer_features(transaction)

        fraud_probability = float(self.model.predict_proba(features_df)[0][1])
        is_flagged = fraud_probability >= FLAG_THRESHOLD

        shap_values = self.explainer.shap_values(features_df)[0]

        contributions = sorted(
            [
                {
                    "feature": col,
                    "value": float(features_df.iloc[0][col]),
                    "shap_contribution": float(shap_values[i]),
                }
                for i, col in enumerate(FEATURE_COLUMNS)
            ],
            key=lambda x: abs(x["shap_contribution"]),
            reverse=True,
        )[:5]

        return {
            "fraud_probability": fraud_probability,
            "is_flagged": is_flagged,
            "model_version": MODEL_VERSION,
            "top_contributing_features": contributions,
        }


fraud_scoring_service = FraudScoringService()