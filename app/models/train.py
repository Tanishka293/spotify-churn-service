import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def generate_spotify_data(num_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic Spotify user engagement data with churn ground truth.
    """
    np.random.seed(random_state)
    user_ids = [f"SPOT-{np.random.randint(10000, 99999)}" for _ in range(num_samples)]
    listening_hours = np.random.gamma(shape=2, scale=5, size=num_samples)
    skip_rate = np.clip(np.random.normal(loc=0.4, scale=0.15, size=num_samples), 0.0, 1.0)
    playlists_created = np.random.poisson(lam=4, size=num_samples)
    days_since_login = np.random.exponential(scale=5, size=num_samples).astype(int)
    auto_renewal = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
    subscription_tier = np.random.choice(["Individual", "Student", "Family"], size=num_samples, p=[0.6, 0.25, 0.15])

    churn_score = (
        (skip_rate * 2.0)
        + (days_since_login * 0.15)
        - (listening_hours * 0.05)
        - (auto_renewal * 1.5)
        + np.random.normal(0, 0.5, num_samples)
    )
    probabilities = 1 / (1 + np.exp(-churn_score))
    renewed_next_month = (probabilities < 0.5).astype(int)

    return pd.DataFrame({
        "user_id": user_ids,
        "listening_hours_per_week": listening_hours,
        "skip_rate": skip_rate,
        "playlists_created": playlists_created,
        "days_since_last_login": days_since_login,
        "auto_renewal": auto_renewal,
        "subscription_tier": subscription_tier,
        "renewed_next_month": renewed_next_month,
    })


def build_pipeline() -> Pipeline:
    """
    Build an end-to-end production ML pipeline that encapsulates preprocessing and modeling.
    """
    numeric_features = [
        "listening_hours_per_week",
        "skip_rate",
        "playlists_created",
        "days_since_last_login",
        "auto_renewal",
    ]
    categorical_features = ["subscription_tier"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    classifier = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        eval_metric="logloss",
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_model() -> None:
    print("[1/4] Generating synthetic Spotify dataset...")
    df = generate_spotify_data()

    data_dir = BASE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "spotify_churn_data.csv"
    df.to_csv(dataset_path, index=False)
    print(f"      Saved {len(df)} records to {dataset_path}")

    # Features and Target
    X = df.drop(columns=["user_id", "renewed_next_month"])
    y = df["renewed_next_month"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[2/4] Training Scikit-Learn + XGBoost Pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("[3/4] Evaluating Model Performance...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
        "precision": float(round(precision_score(y_test, y_pred), 4)),
        "recall": float(round(recall_score(y_test, y_pred), 4)),
        "f1_score": float(round(f1_score(y_test, y_pred), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, y_prob), 4)),
        "test_samples": len(y_test),
    }

    print("\n========== Model Evaluation Metrics ==========")
    for k, v in metrics.items():
        print(f"  {k:15}: {v}")
    print("===============================================\n")

    print("[4/4] Serializing Artifacts & Metadata...")
    artifacts_dir = BASE_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "spotify_xgb_model.joblib"
    metadata_path = artifacts_dir / "model_metadata.json"

    joblib.dump(pipeline, model_path)
    with open(metadata_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"      Production pipeline saved to: {model_path}")
    print(f"      Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    train_model()
