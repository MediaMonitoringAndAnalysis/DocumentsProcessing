# Stage 1: Builder — install Python dependencies
FROM python:3.10-slim-bullseye AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files first to leverage layer caching
COPY setup.py pyproject.toml ./
COPY documents_processing/ ./documents_processing/

# Install into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.10-slim-bullseye AS runtime

WORKDIR /app

# System dependencies:
#   - poppler-utils   → pdf2image (PDF → PNG conversion)
#   - tesseract-ocr   → pytesseract (OCR for handwritten/scanned PDFs)
#   - libgl1-mesa-glx + libglib2.0-0 → OpenCV
#   - wget            → YOLO weight download
#   - libreoffice     → DOCX/PPTX → PDF conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    wget \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Copy application code
COPY . .

# Create data directories and non-root user
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data/original_docs /app/data/figures /app/models \
    && chown -R appuser:appuser /app

USER appuser

# Download YOLOv10 document-layout weights
RUN wget -q -O /app/models/yolov10x_best.pt \
    https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt

EXPOSE 5000

# Default: run the Flask API. Override CMD to use the CLI instead.
CMD ["python", "main_documents_extraction.py"]
