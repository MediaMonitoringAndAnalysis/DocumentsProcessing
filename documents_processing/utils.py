import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import requests
import base64   

additional_supported_file_extensions = [".docx", ".doc", ".pptx"]
supported_file_extensions = [".pdf"] + additional_supported_file_extensions

class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def convert_to_pdf(
    input_file: str, output_file: str, libreoffice_path: str = None
) -> None:
    """
    Convert a docx/doc/pptx to PDF using LibreOffice

    Args:
        input_file: Path to input document
        output_file: Path where to save PDF
        libreoffice_path: Optional path to LibreOffice executable. If not provided,
                         will attempt to detect based on OS.
    """
    # Try to find LibreOffice path if not provided, depending on OS
    if not libreoffice_path:
        if sys.platform == "darwin":  # macOS
            libreoffice_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
        elif sys.platform == "win32":  # Windows
            possible_paths = [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                os.path.expanduser(
                    "~\\AppData\\Programs\\LibreOffice\\program\\soffice.exe"
                ),
            ]
            libreoffice_path = next(
                (path for path in possible_paths if os.path.exists(path)), None
            )
        else:  # Linux
            try:
                libreoffice_path = (
                    subprocess.check_output(["which", "soffice"]).decode().strip()
                )
            except subprocess.CalledProcessError:
                libreoffice_path = None

    if not libreoffice_path or not os.path.exists(libreoffice_path):
        raise RuntimeError(
            "LibreOffice not found. Please install LibreOffice to convert documents to PDF."
        )

    file_ext = Path(input_file).suffix.lower()

    # Convert if not pdf
    if file_ext == ".pdf":
        print("File is already a PDF")
    elif file_ext in additional_supported_file_extensions:
        try:
            command = [
                libreoffice_path,
                "--headless",
                "--convert-to",
                "pdf",
                input_file,
                "--outdir",
                os.path.dirname(output_file) or ".",
            ]
            subprocess.run(command, check=True)
            print("Conversion successful!")
        except subprocess.CalledProcessError:
            print("Error during conversion")
    else:
        print("Invalid file format. Only PDF, DOCX, DOC and PPTX files are supported")


def _flatten_list(l: List[List[str]]) -> List[str]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in l for item in sublist]


def _get_first_n_characters(doc_name: str, n_characters: int = 25) -> str:
    """Get the first N characters of a string and create a filename."""
    doc_type = doc_name.split(".")[-1]
    doc_name = "_".join(doc_name.split(".")[:-1])
    doc_name = doc_name[:n_characters].replace("/", "-").replace(":", "_").replace(" ", "-")
    return doc_name + "." + doc_type


def _download_pdf(url: str, filepath: str):
    """Download a PDF file from a URL and save it."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

def encode_image(image_path: str) -> str:
    """Encode an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")