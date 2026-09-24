from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_TABLE = PROCESSED_DIR / "orders_with_features.parquet"

MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "olist-bad-review"
MODEL_NAME = "olist-bad-review"

CAT_COLS = ["main_category", "payment_type", "customer_state"]
NUM_COLS = [
    "delivery_days",
    "estimated_days",
    "delayed_days",
    "is_late",
    "n_items",
    "n_sellers",
    "total_price",
    "total_freight",
    "freight_ratio",
    "avg_weight_g",
    "max_installments",
]
FEATURES = CAT_COLS + NUM_COLS

DATA_START, DATA_END = "2017-01-01", "2018-09-01"
VALID_START, TEST_START = "2018-04-01", "2018-06-01"

LGBM_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    metric="average_precision",
    random_state=42,
    verbose=-1,
)
