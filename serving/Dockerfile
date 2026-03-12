FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY serving/app /app/app

ENV PYTHONPATH=/app/src
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

RUN useradd -m -u 1000 myuser
RUN mkdir -p /app/logs && chown -R myuser:myuser /app

USER myuser

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}"]
