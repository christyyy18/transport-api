FROM python:3.10-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow and other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app
WORKDIR /app

# Expose the required port
EXPOSE 8080

# Command to run the application
CMD ["python", "transport_model.py"]