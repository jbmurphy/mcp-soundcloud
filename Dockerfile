FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for scdl (ffmpeg for audio conversion)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install scdl separately (SoundCloud downloader)
RUN pip install --no-cache-dir scdl

# Copy application code
COPY src/ ./src/

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 3040

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3040/health || exit 1

# Run the server
CMD ["python", "src/http_server.py"]
