from src.pdf_extraction import PDFExtractor
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
pdf_extractor = PDFExtractor()

@app.route('/extract-doc', methods=['POST'])
def extract_pdf():
    try:
        data = request.json
        
        # Get parameters from request
        pdf_doc_folder_path = data.get('pdf_folder_path', os.path.join("data", "extraction", "pdf_files"))
        pdf_filename = data.get('pdf_filename')
        figures_path = data.get('figures_path', os.path.join(pdf_doc_folder_path, "..", "figures"))
        extract_figures = data.get('extract_figures', True)
        extract_metadata = data.get('extract_metadata', True)
        output_csv = data.get('output_csv', "test_pdf_output.csv")
        
        # Validate required parameters
        if not pdf_filename:
            return jsonify({"error": "pdf_filename is required"}), 400
            
        # Create figures directory if it doesn't exist
        os.makedirs(figures_path, exist_ok=True)
        
        # Process PDF
        pdf_output = pdf_extractor(
            pdf_filename,
            pdf_doc_folder_path,
            figures_path,
            extract_figures_bool=extract_figures,
            extract_metadata_bool=extract_metadata,
        )
        
        # Save to CSV if specified
        if output_csv:
            pdf_output.to_csv(output_csv, index=False)
        
        # Convert DataFrame to dictionary for JSON response
        result = pdf_output.to_dict(orient='records')
        
        return jsonify({
            "status": "success",
            "data": result,
            "csv_path": output_csv if output_csv else None
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)