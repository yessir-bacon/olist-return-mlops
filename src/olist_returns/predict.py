import json
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from olist_returns import config


def load_latest_model():
    """Load the newest registered model from the local MLflow registry."""
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{config.MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    threshold = float(client.get_run(latest.run_id).data.params["threshold"])
    model = mlflow.lightgbm.load_model(f"models:/{config.MODEL_NAME}/{latest.version}")
    return model, threshold


def _download_gcs_dir(uri: str) -> Path:
    """Download every file under a gs://bucket/prefix to a local temp folder."""
    from google.cloud import storage

    bucket_name, _, prefix = uri.removeprefix("gs://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    local_dir = Path(tempfile.mkdtemp())

    for blob in storage.Client().list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        dest = local_dir / blob.name[len(prefix) :]
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))

    if not any(local_dir.iterdir()):
        raise FileNotFoundError(f"No model files found at {uri}")
    return local_dir


def load_exported_model(model_dir: str):
    """Load an exported model from a local folder or a gs:// URI."""
    if model_dir.startswith("gs://"):
        local_dir = _download_gcs_dir(model_dir)
    else:
        local_dir = Path(model_dir)
    model = mlflow.lightgbm.load_model(str(local_dir))
    threshold = json.loads((local_dir / "threshold.json").read_text())["threshold"]
    return model, threshold


def load_model():
    """Use MODEL_DIR if set (Docker/cloud); otherwise the local MLflow registry."""
    model_dir = os.getenv("MODEL_DIR")
    return load_exported_model(model_dir) if model_dir else load_latest_model()


def predict(df: pd.DataFrame, model=None, threshold=None) -> pd.DataFrame:
    if model is None:
        model, threshold = load_model()
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
