# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies and clean up
RUN apt-get update && apt-get install -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better caching
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only the necessary code
COPY . .

# Set permissions for the startup script
RUN chmod +x start.sh

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Set the default command to run both servers
CMD ["/bin/bash", "/start.sh"]
