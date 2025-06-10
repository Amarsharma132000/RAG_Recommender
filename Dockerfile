# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential

# Copy requirements and install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your code
COPY . .

# Copy and set permissions for the startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Set the default command to run both servers
CMD ["/bin/bash", "/start.sh"]
