# Olist Bad Review Prediction: End-to-End MLOps

[![CI](https://github.com/yessir-bacon/olist-return-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/yessir-bacon/olist-return-mlops/actions/workflows/ci.yml)
[![Deploy](https://github.com/yessir-bacon/olist-return-mlops/actions/workflows/deploy.yml/badge.svg)](https://github.com/yessir-bacon/olist-return-mlops/actions/workflows/deploy.yml)

Predicts which e-commerce orders will receive a negative review (1–2 stars) using the
[Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
Negative reviews serve as a proxy for customer dissatisfaction and returns, so flagged orders
can be targeted with proactive outreach.

The project takes a model from notebook to a live cloud API, with experiment tracking, a model
registry, orchestrated retraining, drift monitoring, and tested pipeline code.

**Live API:** [https://olist-api-xwf7nkgo2q-uc.a.run.app/docs](https://olist-api-xwf7nkgo2q-uc.a.run.app/docs).
The first request may take 10–20 seconds while the service starts from zero.

## Architecture

```mermaid
flowchart LR
    A[Raw CSVs] --> B[ingest.py]
    B --> C[features.py]
    C --> D[Modeling table]
    D --> E[train.py]
    E --> F[(MLflow tracking + registry)]
    E --> G[Exported model]
    G --> S[(GCP Cloud Storage)]
    S --> H[FastAPI on GCP Cloud Run]
    D --> M[drift.py / Evidently]
    M -.triggers retrain.-> E
    P[Prefect flows] -.orchestrates.-> B
    P -.-> C
    P -.-> E
    P -.-> M
```

## Results

| Model | Valid PR-AUC | Valid ROC-AUC |
|---|---|---|
| Logistic regression (baseline) | 0.357 | 0.739 |
| **LightGBM** | **0.400** | **0.739** |

Final LightGBM performance on the held-out test set (Jun–Aug 2018):

| Metric | Value |
|---|---|
| PR-AUC | 0.309 (base rate 0.097) |
| ROC-AUC | 0.684 |
| Precision @ threshold 0.22 | 0.451 |
| Recall @ threshold 0.22 | 0.274 |

The decision threshold was chosen on validation by maximizing F1.

## Key findings

- **Delivery delay is the dominant driver** of bad reviews, followed by the number of items
  in an order. Multi-item orders often ship separately and arrive incomplete.
- **The label rate declines over time** (14.0% train → 11.3% valid → 9.7% test), mainly because
  late deliveries fell by half in 2018. Late orders in later periods also produced fewer bad
  reviews, possibly because very late orders were still undelivered when the data was pulled.
- **The time-based split was kept as-is** to reflect realistic production performance. The drop
  from validation to test PR-AUC is an expected result of this shift, not overfitting.

## Monthly retraining replay

A Prefect flow simulates production by retraining each month on the data available so far
and scoring the following month.

| Month | PR-AUC | Base rate | Lift |
|---|---|---|---|
| 2018-01 | 0.511 | 0.137 | 3.7x |
| 2018-02 | 0.641 | 0.194 | 3.3x |
| 2018-03 | 0.672 | 0.212 | 3.2x |
| 2018-04 | 0.435 | 0.118 | 3.7x |
| 2018-05 | 0.364 | 0.108 | 3.4x |
| 2018-06 | 0.280 | 0.100 | 2.8x |
| 2018-07 | 0.283 | 0.097 | 2.9x |
| 2018-08 | 0.375 | 0.095 | 3.9x |

Raw PR-AUC swings with the share of late deliveries, peaking during the Feb–Mar 2018 delivery
spike, but lift over the base rate stays between 2.8x and 3.9x. The model is stable; the
amount of predictable, delivery-driven dissatisfaction is what changes.

## Drift monitoring

A second Prefect flow uses [Evidently](https://github.com/evidentlyai/evidently) to compare
each month's orders against the current model's training data, and retrains only when the
share of drifted features exceeds a threshold.

| Strategy | Retrains | Avg next-month PR-AUC |
|---|---|---|
| Always retrain monthly | 8 | 0.445 |
| Drift threshold 0.3 | 7 | 0.446 |
| **Drift threshold 0.4** | **3** | **0.443** |

A 0.4 threshold cut retraining by over 60% for a 0.002 drop in average PR-AUC.

Overall drift share turned out to be a noisy signal: 4–6 of 14 features drift every month,
because Evidently's tests flag even small seasonal shifts at these sample sizes, and drift
share doesn't track performance closely. A natural next step would be triggering on drift in
the key delivery features or on observed performance once labels arrive.

## Design decisions

- **No leakage:** features use only information available by delivery time. Review text and
  review dates are excluded.
- **Time-based split:** train < Apr 2018, valid Apr–May 2018, test Jun–Aug 2018.
- **PR-AUC over accuracy:** the classes are imbalanced, and base rates differ across splits.
- **Test set evaluated once**, after all model and threshold decisions were made on validation.
- **Model stored outside the image:** the API loads the model from Cloud Storage at startup,
  so a new model can be published without rebuilding the container.

## Project structure

```
src/olist_returns/
├── config.py      # paths, features, split dates, params
├── ingest.py      # load raw CSVs
├── features.py    # build the modeling table
├── train.py       # train, choose threshold, log, register, and export model
├── predict.py     # load model (local registry or Cloud Storage) and score orders
├── drift.py       # Evidently drift reports and drift share
├── api.py         # FastAPI service
└── flows.py       # Prefect training, replay, and drift monitoring flows
notebooks/         # EDA and model development
tests/             # unit tests
Dockerfile
```

## How to run

**Setup**
```bash
conda create -n olist-returns python=3.13 -y
conda activate olist-returns
pip install -r requirements-dev.txt
pip install -e .
```
Download the dataset from Kaggle and unzip the CSVs into `data/raw/`.

**Pipeline**
```bash
python -m olist_returns.features   # build the modeling table
python -m olist_returns.train      # train, register, and export the model
pytest                             # run tests
```

**Orchestrated with Prefect**
```bash
python -m olist_returns.flows          # full training pipeline
python -m olist_returns.flows replay   # monthly retraining replay
python -m olist_returns.flows drift    # drift-triggered retraining (reports saved to reports/)
```

**API with Docker (local)**

Train first so `models/latest/` exists, then mount it into the container:
```bash
docker build -t olist-api .
docker run -p 8080:8080 -v "$(pwd)/models/latest:/models" -e MODEL_DIR=/models olist-api
```
Then open `http://localhost:8080/docs`.

**Deploy to GCP Cloud Run**
```bash
# Publish the model
gcloud storage cp -r models/latest gs://olist-mlops-models/

# Build, push, and deploy the image
docker build -t us-central1-docker.pkg.dev/olist-mlops/olist/olist-api:v2 .
docker push us-central1-docker.pkg.dev/olist-mlops/olist/olist-api:v2
gcloud run deploy olist-api \
  --image us-central1-docker.pkg.dev/olist-mlops/olist/olist-api:v2 \
  --region us-central1 --port 8080 --memory 1Gi \
  --min-instances 0 --max-instances 1 --allow-unauthenticated \
  --set-env-vars MODEL_DIR=gs://olist-mlops-models/latest
```

## Roadmap

- [x] EDA, leak-free features, time-based split
- [x] Baseline and LightGBM models with MLflow tracking
- [x] Modular pipeline with unit tests
- [x] FastAPI prediction service
- [x] Docker containerization
- [x] Prefect orchestration with monthly retraining
- [x] Deployment to GCP Cloud Run, with the model served from Cloud Storage
- [x] Drift monitoring with Evidently and drift-triggered retraining
- [x] CI/CD with GitHub Actions (lint and tests on PRs, auto-deploy to Cloud Run on merge)

## Author

Mason Fuller · [LinkedIn](https://linkedin.com/in/mason1fuller)