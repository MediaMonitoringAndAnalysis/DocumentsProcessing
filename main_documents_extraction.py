from documents_processing.documents_data_extraction import DocumentsDataExtractor
from flask import Flask, request, jsonify
import os


app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/extract-doc", methods=["POST"])
def extract_pdf():
    try:
        data = request.json

        doc_folder_path = data.get("doc_folder_path", os.path.join("data", "original_docs"))
        doc_filename = data.get("doc_filename")
        figures_saving_path = data.get("figures_saving_path", os.path.join("data", "figures"))
        extract_figures = data.get("extract_figures", False)
        metadata_extraction_type = data.get("metadata_extraction_type", "none")
        output_csv = data.get("output_csv", None)
        inference_pipeline = data.get("inference_pipeline", "OpenAI")
        doc_url = data.get("doc_url", None)
        relevant_pages = data.get("relevant_pages_for_metadata_extraction", None)

        if not doc_filename:
            return jsonify({"error": "doc_filename is required"}), 400

        if metadata_extraction_type not in ("document", "interview", "none"):
            return jsonify({"error": "metadata_extraction_type must be 'document', 'interview', or 'none'"}), 400

        os.makedirs(doc_folder_path, exist_ok=True)
        os.makedirs(figures_saving_path, exist_ok=True)

        documents_data_extractor = DocumentsDataExtractor(inference_pipeline)

        pdf_output = documents_data_extractor(
            file_name=doc_filename,
            doc_folder_path=doc_folder_path,
            figures_saving_path=figures_saving_path,
            doc_url=doc_url,
            metadata_extraction_type=metadata_extraction_type,
            extract_figures_bool=extract_figures,
            relevant_pages_for_metadata_extraction=relevant_pages,
        )

        if output_csv:
            pdf_output.to_csv(output_csv, index=False)

        result = pdf_output.to_dict(orient="records")

        document_info = {}
        if len(pdf_output) > 0:
            document_info = {
                "Document Title": pdf_output["Document Title"].iloc[0] if "Document Title" in pdf_output.columns else None,
                "Document Date": pdf_output["Document Publishing Date"].iloc[0] if "Document Publishing Date" in pdf_output.columns else None,
                "Document Source": pdf_output["Document Source"].iloc[0] if "Document Source" in pdf_output.columns else None,
                "Number of Pages": pdf_output["Number of Pages"].iloc[0] if "Number of Pages" in pdf_output.columns else None,
            }

        return jsonify({
            "status": "success",
            "document_info": document_info,
            "data": result,
            "csv_path": output_csv,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
