import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from olist_returns import config


def load_latest_model():
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{config.MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    threshold = float(client.get_run(latest.run_id).data.params["threshold"])
    model = mlflow.lightgbm.load_model(f"models:/{config.MODEL_NAME}/{latest.version}")
    return model, threshold


def load_exported_model(model_dir):
    model_dir = Path(model_dir)
    model = mlflow.lightgbm.load_model(str(model_dir))
    threshold = json.loads((model_dir / "threshold.json").read_text())["threshold"]
    return model, threshold


def load_model():
    """Use MODEL_DIR if set (Docker/cloud); otherwise the local MLflow model registry."""
    model_dir = os.getenv("MODEL_DIR")
    return load_exported_model(model_dir) if model_dir else load_latest_model()


def predict(df, model=None, threshold=None):
    if model is None:
        model, threshold = load_latest_model()
    X = df[config.FEATURES].copy()
    for c in config.CAT_COLS:
        X[c] = X[c].astype("category")
    p = model.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "order_id": df["order_id"].values,
            "probability": p,
            "flag": (p >= threshold).astype(int),
        }
    )
