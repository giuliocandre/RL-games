# Use Python 3.11 as specified in .python-version
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir flask

# Copy application files
COPY app.py .
COPY perudo.py .
COPY perudo_policy.pkl .
COPY html/ ./html/

# Expose port 5000 (Flask default)
EXPOSE 5000

# Set environment variable to ensure Flask runs in production mode
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run the Flask application
CMD ["python", "app.py"]
