FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install system dependencies and AWS CLI (optional)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends awscli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_new2.txt

# Expose port 8080
EXPOSE 8080

# Run Flask app
CMD ["python3", "app.py"]