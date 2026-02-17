# Dockerfile
# ----------
# Containerize the AAVAIL Revenue Forecasting API

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p models logs data

# Expose the API port
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"]
