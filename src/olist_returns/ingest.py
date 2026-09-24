import pandas as pd

from olist_returns import config

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
}
DATE_COLS = {
    "orders": [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "reviews": ["review_creation_date"],
}


def load_raw(raw_dir=config.RAW_DIR) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(raw_dir / fname, parse_dates=DATE_COLS.get(name))
        for name, fname in FILES.items()
    }
