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
