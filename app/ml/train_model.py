"""
Trains an XGBoost fraud classifier on the engineered features.

Key modeling decisions (be ready to explain these in interviews):

1. XGBoost over logistic regression / random forest: gradient boosting
   handles non-linear feature interactions well, and natively supports
   class-weighting via scale_pos_weight without needing SMOTE.

2. scale_pos_weight instead of SMOTE: SMOTE synthesizes fake minority
   examples by interpolation, risking unrealistic points in a financial
   feature space. scale_pos_weight instead penalizes missing fraud more
   heavily in the loss function, using only real data.

3. Stratified K-Fold CV: with ~116 fraud cases in 100k rows, a single
   train/test split risks an unlucky/lucky fold. Stratified CV guarantees
   every fold preserves the true fraud ratio.

4. AUC-PR as primary metric, not AUC-ROC or accuracy: under severe
   imbalance, AUC-ROC is misleadingly optimistic; AUC-PR specifically
   measures ranking quality for the rare positive class.

5. Leakage guards: PaySim's fraud-generation rule deterministically empties
   the origin account and follows a fixed post-transaction balance pattern.
   Any feature derived from POST-transaction state on the origin/destination
   side is a near-direct proxy for the label, not real fraud signal. We only
   keep PRE-transaction state (oldbalance_org, oldbalance_dest), the
   transaction itself (amount, type, step), and ratios built from
   pre-transaction data — signal a real-time fraud system would actually
   have available at decision time, before the transaction completes.

Usage:
    python -m app.ml.train_model
"""

import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "features.parquet"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_features():
    df = pd.read_parquet(DATA_PATH)

    # LEAKAGE GUARD:
    # PaySim's fraud-generation rule deterministically sets post-transaction
    # balances on both the origin and destination side. Any feature computed
    # from newbalance_orig, newbalance_dest, or an error term involving them
    # lets the model read the answer off the label-generation rule itself,
    # rather than learning genuine fraud behavior. A real-time fraud system
    # scores a transaction using only what's known BEFORE/AT the moment of
    # the transaction (pre-transaction balances, amount, type) — not the
    # after-the-fact outcome. We restrict features to that realistic set.
    leaky_cols = [
        "orig_balance_error",
        "orig_emptied",
        "newbalance_orig",
        "dest_balance_error",
        "newbalance_dest",
        "dest_started_zero",
    ]
    df = df.drop(columns=leaky_cols)
    print(f"Dropped leaky (post-transaction-derived) features: {leaky_cols}")

    X = df.drop(columns=["is_fraud_label"])
    y = df["is_fraud_label"].astype(int)
    return X, y


def train():
    X, y = load_features()
    print(f"Loaded {len(X):,} rows, {X.shape[1]} features.")
    print(f"Features used: {list(X.columns)}")
    print(f"Fraud rate: {y.mean() * 100:.4f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.1f}")

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="xgboost_no_leakage_v2"):
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "aucpr",
            "random_state": 42,
        }
        mlflow.log_params(params)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucpr_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr)

            val_probs = model.predict_proba(X_val)[:, 1]
            fold_aucpr = average_precision_score(y_val, val_probs)
            cv_aucpr_scores.append(fold_aucpr)
            print(f"  Fold {fold + 1}: AUC-PR = {fold_aucpr:.4f}")

        mean_cv_aucpr = np.mean(cv_aucpr_scores)
        std_cv_aucpr = np.std(cv_aucpr_scores)
        print(f"\nCV AUC-PR: {mean_cv_aucpr:.4f} (+/- {std_cv_aucpr:.4f})")
        mlflow.log_metric("cv_aucpr_mean", mean_cv_aucpr)
        mlflow.log_metric("cv_aucpr_std", std_cv_aucpr)

        final_model = xgb.XGBClassifier(**params)
        final_model.fit(X_train, y_train)

        test_probs = final_model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)

        test_aucpr = average_precision_score(y_test, test_probs)
        test_auroc = roc_auc_score(y_test, test_probs)

        print(f"\n--- Held-out Test Set Results ---")
        print(f"AUC-PR:  {test_aucpr:.4f}")
        print(f"AUC-ROC: {test_auroc:.4f}")
        print("\n" + classification_report(y_test, test_preds, target_names=["Legit", "Fraud"]))

        mlflow.log_metric("test_aucpr", test_aucpr)
        mlflow.log_metric("test_auroc", test_auroc)

        importance = pd.Series(
            final_model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        print("--- Feature Importances ---")
        print(importance)

        mlflow.xgboost.log_model(final_model, "model")

        model_path = MODEL_DIR / "fraud_model_v1.json"
        final_model.save_model(str(model_path))
        print(f"\nModel saved to {model_path}")

    print("\nRun 'mlflow ui' and open http://localhost:5000 to see this experiment.")


if __name__ == "__main__":
    train()