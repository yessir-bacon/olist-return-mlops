import pandas as pd

from olist_returns.features import add_delivery_features, assign_split


def test_delivery_features():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": pd.to_datetime(["2018-01-01"]),
            "order_delivered_customer_date": pd.to_datetime(["2018-01-11"]),
            "order_estimated_delivery_date": pd.to_datetime(["2018-01-08"]),
        }
    )
    out = add_delivery_features(df).iloc[0]
    assert out["delivery_days"] == 10
    assert out["estimated_days"] == 7
    assert out["delayed_days"] == 3
    assert out["is_late"] == 1


def test_assign_split():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": pd.to_datetime(
                ["2016-12-31", "2017-06-01", "2018-04-15", "2018-07-01", "2018-09-05"]
            )
        }
    )
    out = assign_split(df)
    assert list(out["split"]) == ["train", "valid", "test"]  # edges dropped
