from pdf_extraction import PDFExtractor
import os

pdf_doc_folder_path = os.path.join("data", "extraction", "pdf_files")
pdf_files = [
    os.path.join(pdf_doc_folder_path, f)
    for f in os.listdir(pdf_doc_folder_path)
    if f.endswith(".pdf")
]

figures_path = os.path.join(pdf_doc_folder_path, "..", "figures")
os.makedirs(figures_path, exist_ok=True)

pdf_extractor = PDFExtractor()

one_pdf_output = pdf_extractor(
    pdf_files[0].split("/")[-1],
    pdf_doc_folder_path,
    figures_path,
    extract_figures_bool=True,
    extract_metadata_bool=True,
)
one_pdf_output.to_csv("test_pdf_output.csv", index=False)
print(one_pdf_output)