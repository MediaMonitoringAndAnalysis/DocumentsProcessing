#!/usr/bin/env python
import os
import argparse
import sys
from documents_processing import DocumentsDataExtractor

def main():
    parser = argparse.ArgumentParser(description="Extract information from PDF documents")
    parser.add_argument("--file", required=True, help="PDF file to process")
    parser.add_argument("--output", default="output.csv", help="Output CSV file")
    parser.add_argument("--inference", default="OpenAI", choices=["OpenAI", "Ollama"], 
                        help="Inference pipeline to use")
    parser.add_argument("--extract-figures", action="store_true", help="Extract figures from PDF")
    parser.add_argument("--extract-metadata", action="store_true", help="Extract metadata from PDF")
    parser.add_argument("--data-dir", default="data", help="Directory to store data")
    
    args = parser.parse_args()
    
    # Create data directories
    pdf_folder = os.path.join(args.data_dir, "original_docs")
    figures_folder = os.path.join(args.data_dir, "figures")
    os.makedirs(pdf_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)
    
    # Initialize extractor
    extractor = DocumentsDataExtractor(args.inference)
    
    # Process document
    results = extractor(
        file_name=args.file,
        doc_folder_path=pdf_folder,
        extract_metadata_bool=args.extract_metadata,
        extract_figures_bool=args.extract_figures
    )
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()