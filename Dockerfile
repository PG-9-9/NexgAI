# --- Builder Stage: Install & cache dependencies ---
FROM python:3.10-slim AS builder
WORKDIR /tmp

# 1. System tools for pip installs & git-based deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# 2. Copy & sanitize requirements
COPY requirements.txt .
# Remove all CUDA/GPU-specific wheels
RUN grep -Ev "^(torch|torchvision|torchaudio|nvidia-|cudnn|cusparse)" requirements.txt > clean-reqs.txt

# 3. Install CPU-only torch globally
RUN pip install --no-cache-dir torch==2.7.1

# 4. Install the rest of your deps globally
RUN pip install --no-cache-dir -r clean-reqs.txt

# 5. Pre-download the SentenceTransformer model into HF cache
ENV HF_HOME=/model_cache
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-large-en-v1.5')
EOF


# --- Final Stage: Copy only runtime artifacts ---
FROM python:3.10-slim
WORKDIR /app

# Copy installed Python packages & CLI scripts
COPY --from=builder /usr/local /usr/local

# Copy the HF model cache so startup is instant
COPY --from=builder /model_cache /root/.cache/huggingface

# Copy the application source
COPY . .

# Expose and launch
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
