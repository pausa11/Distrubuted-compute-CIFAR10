FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requisitos de la raíz
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Script de la raíz
COPY train.py /app/train.py

# Usuario no-root (opcional)
RUN useradd -m runner && chown -R runner:runner /app
USER runner

VOLUME ["/data", "/app/checkpoints"]

CMD ["python", "train.py", "--epochs", "2", "--batch-size", "128", "--cpu", "--data", "/data", "--checkpoints", "/app/checkpoints"]
