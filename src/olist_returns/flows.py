import sys

import mlflow
import pandas as pd
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from olist_returns import config
from olist_returns.features import assign_split, build_base_table
from olist_returns.ingest import load_raw
from olist_returns.train import evaluate, setup_mlflow, to_category, train_model


@task(retries=2, retry_delay_seconds=10, cache_policy=NO_CACHE)
def ingest():
    return load_raw()


@task(cache_policy=NO_CACHE)
def build_features(raw):
    return to_category(build_base_table(raw))


@task(cache_policy=NO_CACHE)
def train_and_register(train, valid):
    return train_model(train, valid, register=True, export=True)


@task(cache_policy=NO_CACHE)
def split_for_cutoff(df, cutoff):
    """Train on data older than 2 months, validate on the 2 months before
    the cutoff, and evaluate on the month after it."""
    ts = df["order_purchase_timestamp"]
    valid_start = cutoff - pd.DateOffset(months=2)
    next_end = cutoff + pd.DateOffset(months=1)
    return (
        df[ts < valid_start],
        df[(ts >= valid_start) & (ts < cutoff)],
        df[(ts >= cutoff) & (ts < next_end)],
    )


@task(cache_policy=NO_CACHE)
def retrain_and_evaluate(train, valid, next_month, cutoff):
    model, threshold, _, run_id = train_model(
        train,
        valid,
        run_name=f"replay_{cutoff:%Y-%m}",
        tags={"flow": "monthly_replay", "cutoff": f"{cutoff:%Y-%m-%d}"},
    )
    metrics = evaluate(model, threshold, next_month)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"next_month_{k}": v for k, v in metrics.items()})
    return {"month": f"{cutoff:%Y-%m}", **metrics}


@flow(name="olist-training-pipeline", log_prints=True)
def training_pipeline():
    setup_mlflow()
    df = assign_split(build_features(ingest()))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.MODEL_TABLE, index=False)

    _, threshold, metrics, _ = train_and_register(
        df[df["split"] == "train"], df[df["split"] == "valid"]
    )
    print(f"Valid PR-AUC: {metrics['valid_pr_auc']:.4f} | threshold: {threshold:.4f}")


@flow(name="olist-monthly-replay", log_prints=True)
def monthly_replay(start="2018-01-01", end="2018-08-01"):
    setup_mlflow()
    df = build_features(ingest())
    results = []
    for cutoff in pd.date_range(start, end, freq="MS"):
        train, valid, next_month = split_for_cutoff(df, cutoff)
        results.append(retrain_and_evaluate(train, valid, next_month, cutoff))

    summary = pd.DataFrame(results)
    print("\n" + summary.round(3).to_string(index=False))
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        monthly_replay()
    else:
        training_pipeline()
