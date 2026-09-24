from olist_returns import config
from olist_returns.ingest import load_raw


def build_label(orders, reviews):
    latest = reviews.sort_values("review_creation_date").drop_duplicates(
        "order_id", keep="last"
    )
    df = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
    ].copy()
    df = df.merge(latest[["order_id", "review_score"]], on="order_id", how="inner")
    df["label"] = (df["review_score"] <= 2).astype(int)
    return df


def add_delivery_features(df):
    df = df.copy()
    purchase = df["order_purchase_timestamp"]
    df["delivery_days"] = (df["order_delivered_customer_date"] - purchase).dt.days
    df["estimated_days"] = (df["order_estimated_delivery_date"] - purchase).dt.days
    df["delayed_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days
    df["is_late"] = (df["delayed_days"] > 0).astype(int)
    return df


def add_item_features(df, items, products):
    feats = (
        items.merge(products, on="product_id", how="left")
        .groupby("order_id")
        .agg(
            n_items=("order_item_id", "count"),
            n_sellers=("seller_id", "nunique"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            avg_weight_g=("product_weight_g", "mean"),
            main_category=("product_category_name", "first"),
        )
        .reset_index()
    )
    df = df.merge(feats, on="order_id", how="left")
    df["freight_ratio"] = df["total_freight"] / df["total_price"]
    return df


def add_payment_features(df, payments):
    feats = (
        payments.groupby("order_id")
        .agg(
            payment_type=("payment_type", "first"),
            max_installments=("payment_installments", "max"),
        )
        .reset_index()
    )
    return df.merge(feats, on="order_id", how="left")


def assign_split(df):
    ts = df["order_purchase_timestamp"]
    df = df[(ts >= config.DATA_START) & (ts < config.DATA_END)].copy()
    df["split"] = "train"
    df.loc[df["order_purchase_timestamp"] >= config.VALID_START, "split"] = "valid"
    df.loc[df["order_purchase_timestamp"] >= config.TEST_START, "split"] = "test"
    return df


def build_model_table(raw):
    df = build_label(raw["orders"], raw["reviews"])
    df = add_delivery_features(df)
    df = add_item_features(df, raw["items"], raw["products"])
    df = add_payment_features(df, raw["payments"])
    df = df.merge(
        raw["customers"][["customer_id", "customer_state"]],
        on="customer_id",
        how="left",
    )
    df = assign_split(df)
    base_cols = ["order_id", "order_purchase_timestamp", "split", "label"]
    return df[base_cols + config.FEATURES]


def main():
    table = build_model_table(load_raw())
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(config.MODEL_TABLE, index=False)
    print(f"Saved {len(table):,} rows to {config.MODEL_TABLE}")


if __name__ == "__main__":
    main()
