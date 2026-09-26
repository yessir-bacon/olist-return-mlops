FROM python:3.11-slim

# LightGBM needs OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/latest/ models/latest/
ENV MODEL_DIR=/app/models/latest

EXPOSE 8080
CMD ["uvicorn", "olist_returns.api:app", "--host", "0.0.0.0", "--port", "8080"]