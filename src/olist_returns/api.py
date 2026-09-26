from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from olist_returns.predict import load_model, predict


class Order(BaseModel):
    order_id: str
    main_category: str | None = None
    payment_type: str | None = None
    customer_state: str | None = None
    delivery_days: float
    estimated_days: float
    delayed_days: float
    is_late: int
    n_items: int
    n_sellers: int
    total_price: float
    total_freight: float
    freight_ratio: float | None = None
    avg_weight_g: float | None = None
    max_installments: int | None = None


class Prediction(BaseModel):
    order_id: str
    probability: float
    flag: int


state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request
    state["model"], state["threshold"] = load_model()
    yield


app = FastAPI(title="Olist Bad Review API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": state["threshold"]}


@app.post("/predict", response_model=list[Prediction])
def predict_orders(orders: list[Order]):
    df = pd.DataFrame([o.model_dump() for o in orders])
    return predict(df, state["model"], state["threshold"]).to_dict(orient="records")
