# Fraud scoring service. Build from the project root after Steps 5, 8 and 9 have created the models/ files:
#   python -m fraud.pin_requirements
#   docker build -t fraud-api --build-arg PYTHON_VERSION=3.14 .      (use the version pin_requirements prints)
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRAUD_ROOT=/app
WORKDIR /app

# lightgbm needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY fraud ./fraud
COPY models ./models

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "fraud.api:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
