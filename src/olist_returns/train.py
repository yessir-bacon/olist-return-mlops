import json
import shutil

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from olist_returns import config


def setup_mlflow():
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.EXPERIMENT)


def to_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in config.CAT_COLS:
        df[c] = df[c].astype("category")
    return df


def load_splits(path=config.MODEL_TABLE) -> dict[str, pd.DataFrame]:
    df = to_category(pd.read_parquet(path))
    return {s: df[df["split"] == s] for s in ["train", "valid", "test"]}


def choose_threshold(y_true, p) -> float:
    """Pick the threshold that maximizes F1."""
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[int(np.argmax(f1[:-1]))])


def evaluate(model, threshold: float, df: pd.DataFrame) -> dict:
    y = df["label"]
    p = model.predict_proba(df[config.FEATURES])[:, 1]
    pred = (p >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "base_rate": float(y.mean()),
        "n_orders": len(df),
    }


def export_model(model, threshold: float):
    """Save the model and threshold to models/latest for Docker."""
    export_dir = config.ROOT / "models" / "latest"
    shutil.rmtree(export_dir, ignore_errors=True)
    mlflow.lightgbm.save_model(model, str(export_dir))
    (export_dir / "threshold.json").write_text(json.dumps({"threshold": threshold}))


def train_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    run_name="lgbm_train",
    register=False,
    export=False,
    tags=None,
):
    X_tr, y_tr = train[config.FEATURES], train["label"]
    X_va, y_va = valid[config.FEATURES], valid["label"]

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            eval_X=X_va,
            eval_y=y_va,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        p_va = model.predict_proba(X_va)[:, 1]
        threshold = choose_threshold(y_va, p_va)
        metrics = {
            "valid_pr_auc": average_precision_score(y_va, p_va),
            "valid_roc_auc": roc_auc_score(y_va, p_va),
        }

        mlflow.log_params(
            {
                **config.LGBM_PARAMS,
                "best_iteration": model.best_iteration_,
                "threshold": threshold,
            }
        )
        mlflow.log_metrics(metrics)
        if register:
            mlflow.lightgbm.log_model(
                lgb_model=model, name="model", registered_model_name=config.MODEL_NAME
            )

    if export:
        export_model(model, threshold)
    return model, threshold, metrics, run.info.run_id


def main():
    setup_mlflow()
    splits = load_splits()
    _, threshold, metrics, _ = train_model(
        splits["train"], splits["valid"], register=True, export=True
    )
    print(f"Valid PR-AUC: {metrics['valid_pr_auc']:.4f} | threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
