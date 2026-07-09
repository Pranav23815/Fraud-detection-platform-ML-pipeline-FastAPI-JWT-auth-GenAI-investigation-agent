import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import insert

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.db.database import AsyncSessionLocal
from app.models.models import Transaction

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "PS_20174392719_1491204439457_log.csv"
BATCH_SIZE = 5000


async def ingest(limit):
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}")
        print("Download it from Kaggle (ealaxi/paysim1) and place it in the data/ folder.")
        return

    print(f"Reading CSV from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, nrows=limit)
    print(f"Loaded {len(df):,} rows into memory. Starting DB insert...")

    df = df.rename(columns={
        "nameOrig": "name_orig",
        "oldbalanceOrg": "oldbalance_org",
        "newbalanceOrig": "newbalance_orig",
        "nameDest": "name_dest",
        "oldbalanceDest": "oldbalance_dest",
        "newbalanceDest": "newbalance_dest",
        "isFraud": "is_fraud_label",
    })
    df["is_fraud_label"] = df["is_fraud_label"].astype(bool)

    columns = [
        "step", "type", "amount", "name_orig", "oldbalance_org",
        "newbalance_orig", "name_dest", "oldbalance_dest",
        "newbalance_dest", "is_fraud_label",
    ]
    records = df[columns].to_dict(orient="records")

    async with AsyncSessionLocal() as session:
        total = len(records)
        for i in range(0, total, BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            await session.execute(insert(Transaction), batch)
            await session.commit()
            print(f"  Inserted {min(i + BATCH_SIZE, total):,} / {total:,}")

    print("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100_000)
    args = parser.parse_args()
    limit = None if args.limit == 0 else args.limit
    asyncio.run(ingest(limit))