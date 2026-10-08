FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies required by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .


RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 -r requirements.txt

COPY . .

# Default process--> overridden by Docker Compose
CMD ["python", "-m", "streaming.consumer"]
