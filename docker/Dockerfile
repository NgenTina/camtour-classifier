# Use a compatible Python version
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY api/requirements-api.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Install transformers and torch with CUDA support
RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    accelerate