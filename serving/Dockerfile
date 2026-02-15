FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only
RUN pip install torch torchvision --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and app
COPY src /app/src
COPY serving/app /app/app

# Set python path to include src
ENV PYTHONPATH=/app/src

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Create non-root user early so we can chown files
RUN useradd -m -u 1000 myuser

# Ensure logs directory exists and is writable by the non-root user
RUN mkdir -p /app/logs \
    && chown -R myuser:myuser /app

USER myuser

# Expose port
EXPOSE 8000

# Run
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
