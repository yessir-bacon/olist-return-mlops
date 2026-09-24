import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from olist_returns import config


def load_splits(path=config.MODEL_TABLE):
    df = pd.read_parquet(path)
    for c in config.CAT_COLS:
        df[c] = df[c].astype("category")
    return {s: df[df["split"] == s] for s in ["train", "valid", "test"]}


def choose_threshold(y_true, p):
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[int(np.argmax(f1[:-1]))])


def main():
    splits = load_splits()
    X_tr, y_tr = splits["train"][config.FEATURES], splits["train"]["label"]
    X_va, y_va = splits["valid"][config.FEATURES], splits["valid"]["label"]

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.EXPERIMENT)

    with mlflow.start_run(run_name="lgbm_train"):
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

        mlflow.log_params(
            {
                **config.LGBM_PARAMS,
                "best_iteration": model.best_iteration_,
                "threshold": threshold,
            }
        )
        mlflow.log_metrics(
            {
                "valid_pr_auc": average_precision_score(y_va, p_va),
                "valid_roc_auc": roc_auc_score(y_va, p_va),
            }
        )
        mlflow.lightgbm.log_model(
            lgb_model=model, name="model", registered_model_name=config.MODEL_NAME
        )
    print(
        f"Valid PR-AUC: {average_precision_score(y_va, p_va):.4f} | "
        f"threshold: {threshold:.4f}"
    )


if __name__ == "__main__":
    main()
