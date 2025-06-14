# Use official Python slim image
FROM python:3.10-slim as base

# Set work directory
WORKDIR /app

# Copy requirements.txt first for better caching
COPY requirements.txt .

# Install system dependencies, build tools, and Python dependencies in one layer, then clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the code
COPY . .

# Set permissions for the startup script
RUN chmod +x start.sh

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Set the default command to run both servers
CMD ["/bin/bash", "start.sh"]
