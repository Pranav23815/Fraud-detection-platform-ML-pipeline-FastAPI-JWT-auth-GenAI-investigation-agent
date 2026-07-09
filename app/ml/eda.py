"""
Exploratory Data Analysis on the ingested transactions.
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sqlalchemy import select
from app.db.database import AsyncSessionLocal
from app.models.models import Transaction


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


def run_eda(df: pd.DataFrame):
    print("=" * 60)
    print(f"TOTAL TRANSACTIONS: {len(df):,}")
    print("=" * 60)

    fraud_count = df["is_fraud_label"].sum()
    fraud_pct = fraud_count / len(df) * 100
    print(f"\nFraud cases: {fraud_count:,} ({fraud_pct:.4f}% of all transactions)")
    print(f"Legit cases: {len(df) - fraud_count:,} ({100 - fraud_pct:.4f}%)")
    print(
        "\n-> This is a SEVERE class imbalance. Accuracy is meaningless here: "
        "a model that always predicts 'not fraud' would score >99% accuracy "
        "while catching zero fraud. We'll optimize for recall/precision/AUC-PR instead."
    )

    print("\n--- Transaction Type Distribution ---")
    print(df["type"].value_counts())

    print("\n--- Fraud Rate by Transaction Type ---")
    fraud_by_type = df.groupby("type")["is_fraud_label"].agg(["sum", "count", "mean"])
    fraud_by_type.columns = ["fraud_count", "total_count", "fraud_rate"]
    print(fraud_by_type.sort_values("fraud_rate", ascending=False))
    print(
        "\n-> If fraud is concentrated in specific transaction types, that's a "
        "critical feature — and something we can call out explicitly as a "
        "modeling decision in interviews."
    )

    print("\n--- Amount Statistics: Fraud vs Legit ---")
    print(df.groupby("is_fraud_label")["amount"].describe())

    df["orig_balance_diff"] = df["oldbalance_org"] - df["newbalance_orig"] - df["amount"]
    inconsistent = (df["orig_balance_diff"].abs() > 0.01).sum()
    print(
        f"\nTransactions where balances don't reconcile with amount: {inconsistent:,} "
        f"({inconsistent / len(df) * 100:.2f}%)"
    )
    print(
        "-> This balance-reconciliation mismatch is a genuinely useful engineered "
        "feature — it's not in the raw columns, we derived it from domain logic."
    )

    print("\n" + "=" * 60)
    print("EDA complete. See app/ml/feature_engineering.py for the next step.")
    print("=" * 60)


if __name__ == "__main__":
    df = asyncio.run(load_transactions())
    run_eda(df)