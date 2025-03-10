import os
import argparse
from documents_processing.documents_data_extraction import DocumentsDataExtractor


def test_pdf_extraction(inference_pipeline_name: str):
    # Initialize the extractor
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
    
    # Print results
    print("\nExtracted Information:")
    print("----------------------")
    print(f"Document Title: {results_df['Document Title'].iloc[0]}")
    print(f"Document Date: {results_df['Document Publishing Date'].iloc[0]}")
    print(f"Document Source: {results_df['Document Source'].iloc[0]}")
    
    print("\nExtracted Text Entries:")
    for idx, row in results_df.iterrows():
        print(f"\nEntry Type: {row['Entry Type']}")
        print(f"Text: {row['text'][:200]}...")  # Print first 200 chars
        
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_pipeline", type=str, default="OpenAI", choices=["OpenAI", "Ollama"])
    args = parser.parse_args()
    
    test_pdf_extraction(args.inference_pipeline) 