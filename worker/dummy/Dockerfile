FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
