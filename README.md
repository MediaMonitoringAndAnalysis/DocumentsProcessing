# DocumentsProcessing
A Python library for extracting and processing information from PDF documents using advanced vision-language models.

## Features
- Extract text content from PDF documents
- Extract figures and images with captions
- Extract metadata (title, authors, publication date, etc.)
- Process documents using various LLM backends (OpenAI, Ollama)
- Command-line interface for easy document processing
- Flask API for integration with web applications

## Installation

### Dependencies

<!-- 1. Install the base requirements:
```bash
pip install pdf2image numpy opencv-python pillow torch transformers requests PyMuPDF pandas tqdm nltk
``` -->

<!-- 2. Install YOLOv10 for document layout analysis:
```bash
pip install -q git+https://github.com/THU-MIG/yolov10.git
pip install -q supervision
```

3. Download the YOLOv10 pre-trained weights:
```bash
wget https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt
mv yolov10x_best.pt models/yolov10x_best.pt
``` -->

1. Download [Libre Office](https://www.libreoffice.org) to convert word and pptx files to pdf.

2. Install the project:
* Install the project from the repository:
```bash
git clone https://github.com/MediaMonitoringAndAnalysis/DocumentsProcessing.git
cd DocumentsProcessing
pip install -e .
```
* Install with pip:
```bash
pip install git+https://github.com/MediaMonitoringAndAnalysis/DocumentsProcessing.git
```

## Usage

```python
import os
import argparse
from documents_processing import DocumentsDataExtractor

inference_pipeline_name = Literal["Ollama", "OpenAI"]

documents_data_extractor = DocumentsDataExtractor(
    inference_pipeline_name
)

# Define paths
base_path = "data"
pdf_folder = os.path.join(base_path, "original_docs")
figures_folder = os.path.join(base_path, "figures")

# Create necessary directories
os.makedirs(pdf_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)

# Test PDF file
pdf_filename = "test.pdf"

# Extract information
results_df = documents_data_extractor(
    file_name=pdf_filename,
    doc_folder_path=pdf_folder,
    extract_metadata_bool=True,
    extract_figures_bool=True
)

# Save results
output_path = "test_output.csv"
results_df.to_csv(output_path, index=False)
```

### Using OpenAI Models

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

1. Fork the repository
2. Create your feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add some amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

## Acknowledgements

This project uses various open-source libraries and models
Special thanks to the contributors and maintainers of the dependencies

## TODO
- generate poetry file for the project (medium priority).
- test local api for pdf processing (medium priority).
- test dockerfile (medium priority).
- create docker-compose file (low priority).