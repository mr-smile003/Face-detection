FROM python:3.10-slim

# Install system dependencies for dlib and Python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libx11-dev \
    libboost-python-dev \
    libboost-thread-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip and upgrade setuptools, wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better layer caching
COPY requirements.txt .

# Use prebuilt binary for dlib if possible
RUN pip install --no-cache-dir --only-binary=:all: dlib || pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default port
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
