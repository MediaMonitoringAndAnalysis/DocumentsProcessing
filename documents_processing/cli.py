#!/usr/bin/env python
import os
import argparse
from documents_processing import DocumentsDataExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract information from PDF documents")
    parser.add_argument("--file", required=True, help="Document file to process (PDF, DOCX, PPTX)")
    parser.add_argument("--output", default="output.csv", help="Output CSV file path")
    parser.add_argument(
        "--inference",
        default="OpenAI",
        choices=["OpenAI", "Ollama"],
        help="Inference pipeline to use",
    )
    parser.add_argument("--extract-figures", action="store_true", help="Extract figures and tables from document")
    parser.add_argument(
        "--metadata-type",
        default="none",
        choices=["document", "interview", "none"],
        help="Type of metadata to extract",
    )
    parser.add_argument("--data-dir", default="data", help="Base directory to store data")
    parser.add_argument("--doc-url", default=None, help="URL to download the document from if not found locally")

    args = parser.parse_args()

    pdf_folder = os.path.join(args.data_dir, "original_docs")
    figures_folder = os.path.join(args.data_dir, "figures")
    os.makedirs(pdf_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)

    extractor = DocumentsDataExtractor(args.inference)

    results = extractor(
        file_name=args.file,
        doc_folder_path=pdf_folder,
        figures_saving_path=figures_folder,
        doc_url=args.doc_url,
        metadata_extraction_type=args.metadata_type,
        extract_figures_bool=args.extract_figures,
    )

    results.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
