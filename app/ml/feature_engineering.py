"""
Feature engineering pipeline for fraud detection.
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sqlalchemy import select
from app.db.database import AsyncSessionLocal
from app.models.models import Transaction

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "features.parquet"


async def load_transactions() -> pd.DataFrame:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Transaction))
        rows = result.scalars().all()
        data = [
            {
                "step": r.step,
                "type": r.type,
                "amount": r.amount,
                "name_orig": r.name_orig,
                "oldbalance_org": r.oldbalance_org,
                "newbalance_orig": r.newbalance_orig,
                "name_dest": r.name_dest,
                "oldbalance_dest": r.oldbalance_dest,
                "newbalance_dest": r.newbalance_dest,
                "is_fraud_label": r.is_fraud_label,
            }
            for r in rows
        ]
    return pd.DataFrame(data)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["orig_balance_error"] = df["oldbalance_org"] - df["newbalance_orig"] - df["amount"]
    df["dest_balance_error"] = df["newbalance_dest"] - df["oldbalance_dest"] - df["amount"]

    df["orig_emptied"] = ((df["oldbalance_org"] > 0) & (df["newbalance_orig"] == 0)).astype(int)
    df["dest_started_zero"] = (df["oldbalance_dest"] == 0).astype(int)

    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalance_org"] + 1)

    type_dummies = pd.get_dummies(df["type"], prefix="type")
    df = pd.concat([df, type_dummies], axis=1)

    df["dest_is_merchant"] = df["name_dest"].str.startswith("M").astype(int)

    df["log_amount"] = np.log1p(df["amount"])

    feature_cols = [
        "step", "amount", "log_amount",
        "oldbalance_org", "newbalance_orig", "oldbalance_dest", "newbalance_dest",
        "orig_balance_error", "dest_balance_error",
        "orig_emptied", "dest_started_zero",
        "amount_to_balance_ratio", "dest_is_merchant",
    ] + list(type_dummies.columns)

    result = df[feature_cols + ["is_fraud_label"]].copy()
    return result


if __name__ == "__main__":
    print("Loading transactions from Postgres...")
    raw_df = asyncio.run(load_transactions())
    print(f"Loaded {len(raw_df):,} rows. Engineering features...")

    features_df = engineer_features(raw_df)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    features_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nEngineered {len(features_df.columns) - 1} features.")
    print(f"Saved to {OUTPUT_PATH}")
    print("\nFeature columns:")
    for col in features_df.columns:
        print(f"  - {col}")