FROM python:3.13-slim

# LightGBM needs OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8080
CMD ["uvicorn", "olist_returns.api:app", "--host", "0.0.0.0", "--port", "8080"]