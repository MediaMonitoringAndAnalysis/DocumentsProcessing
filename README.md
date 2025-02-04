# DocumentsProcessing

A Python library for extracting and processing information from PDF documents, including text, figures, tables, and metadata. The library uses advanced vision-language models for image understanding and text extraction.

## Features

- PDF text extraction with intelligent text cleaning and formatting
- Automatic figure and table detection using YOLOv10
- Face detection and removal for privacy protection
- Metadata extraction (publication date, authors, title)
- Support for multiple inference pipelines (VLM and OpenAI)
- Structured output formatting

## Installation

### Dependencies

1. Install the base requirements:
```bash
pip install pdf2image numpy opencv-python pillow torch transformers requests PyMuPDF pandas tqdm nltk
```

2. Install YOLOv10 for document layout analysis:
```bash
pip install -q git+https://github.com/THU-MIG/yolov10.git
pip install -q supervision
```

3. Download the YOLOv10 pre-trained weights:
```bash
wget https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt
mv yolov10x_best.pt models/yolov10x_best.pt
```

## Usage

### Basic Usage

```python
from src.pdf_extraction import PDFExtractor

# Initialize the extractor
extractor = PDFExtractor(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    inference_pipeline_name="VLM"
)

# Process a PDF file
results = extractor(
    pdf_file_name="document.pdf",
    pdf_doc_folder_path="path/to/pdf/folder",
    extract_figures_bool=True,
    extract_metadata_bool=True
)
```

### Using OpenAI Models

```python
extractor = PDFExtractor(
    model_name="gpt-4o",
    inference_pipeline_name="OpenAI",
    api_key="your-api-key"
)
```

## Output Format

The extractor returns a pandas DataFrame with the following columns:
- `text`: Extracted text content
- `Entry Type`: Type of entry (PDF Text, PDF Picture, PDF Table)
- `entry_fig_path`: Path to extracted figures/tables
- `Document Publishing Date`: Publication date (if metadata extraction enabled)
- `Document Source`: Author organizations (if metadata extraction enabled)
- `Document Title`: Document title (if metadata extraction enabled)

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.