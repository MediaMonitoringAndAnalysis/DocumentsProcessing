# Use multi-stage build for better security and smaller image size
FROM python:3.10-slim-bullseye as builder

# Set working directory
WORKDIR /app

# Install system dependencies required for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Second stage: Runtime
FROM python:3.10-slim-bullseye as runtime

# Set working directory
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Prevent Python from writing pyc files
    PYTHONHASHSEED=random \
    # Enable better Python security features
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Copy the application code
COPY . .

# Create necessary directories with appropriate permissions
RUN mkdir -p /app/data/extraction/pdf_files /app/data/extraction/figures \
    && chown -R 1000:1000 /app

# Create and switch to non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Download YOLOv10 weights (as non-root user)
RUN mkdir -p models \
    && wget -O models/yolov10x_best.pt https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt

# Set secure file permissions
RUN find /app -type d -exec chmod 755 {} \; \
    && find /app -type f -exec chmod 644 {} \;

# Expose any necessary ports (if needed)
# EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["python"]

# Default command (can be overridden)
CMD ["main_pdf_extraction.py"]