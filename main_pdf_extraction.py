from documents_processing.documents_data_extraction import DocumentsDataExtractor
from flask import Flask, request, jsonify
import os


app = Flask(__name__)

# Initialize with default inference pipeline


@app.route('/extract-doc', methods=['POST'])
def extract_pdf():
    try:
        data = request.json
        
        # Get parameters from request
        doc_folder_path = data.get('doc_folder_path', os.path.join("data", "original_docs"))
        doc_filename = data.get('doc_filename')
        figures_path = data.get('figures_path', os.path.join("data", "figures"))
        extract_figures = data.get('extract_figures', True)
        extract_metadata = data.get('extract_metadata', True)
        output_csv = data.get('output_csv', "test_output.csv")
        inference_pipeline = data.get('inference_pipeline', "OpenAI")
        
        documents_data_extractor = DocumentsDataExtractor(inference_pipeline)
        
        # Validate required parameters
        if not doc_filename:
            return jsonify({"error": "doc_filename is required"}), 400
            
        # Create necessary directories
        os.makedirs(doc_folder_path, exist_ok=True)
        os.makedirs(figures_path, exist_ok=True)
        
        # Process PDF
        pdf_output = documents_data_extractor(
            file_name=doc_filename,
            doc_folder_path=doc_folder_path,
            figures_path=figures_path,
            extract_figures_bool=extract_figures,
            extract_metadata_bool=extract_metadata,
        )
        
        # Save to CSV if specified
        if output_csv:
            pdf_output.to_csv(output_csv, index=False)
        
        # Convert DataFrame to dictionary for JSON response
        result = pdf_output.to_dict(orient='records')
        
        # Extract key information for the response
        document_info = {}
        if len(pdf_output) > 0:
            document_info = {
                "Document Title": pdf_output['Document Title'].iloc[0] if 'Document Title' in pdf_output.columns else None,
                "Document Date": pdf_output['Document Publishing Date'].iloc[0] if 'Document Publishing Date' in pdf_output.columns else None,
                "Document Source": pdf_output['Document Source'].iloc[0] if 'Document Source' in pdf_output.columns else None
            }
        
        return jsonify({
            "status": "success",
            "document_info": document_info,
            "data": result,
            "csv_path": output_csv if output_csv else None
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)