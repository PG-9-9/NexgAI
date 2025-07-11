FROM python:3.10-slim

WORKDIR /app

# Install system build deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

# Copy and sanitize requirements
COPY requirements.txt .

# Remove any lines for CUDA/GPU packages (torch+cu, nvidia-, cudnn, cusparse, etc.)
RUN grep -Ev "^(torch|torchvision|torchaudio|nvidia-|cudnn|cusparse)" requirements.txt > clean-reqs.txt

# Install a CPU-only torch (match your code’s torch version)
RUN pip install --no-cache-dir torch==2.7.1

# Install the rest of your dependencies
RUN pip install --no-cache-dir -r clean-reqs.txt

# Copy application code
COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
