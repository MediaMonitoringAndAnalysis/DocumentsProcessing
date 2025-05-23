import sys
import contextlib
import os
import requests
import fitz
import subprocess
import base64
import pandas as pd
from typing import List, Dict, Literal, Optional, Union
from documents_processing.figures_extraction import extract_figures
from nltk.tokenize import sent_tokenize
from punctuators.models import PunctCapSegModelONNX
from documents_processing.utils import (
    SuppressPrint,
    convert_to_pdf,
    _flatten_list,
    _get_first_n_characters,
    _download_pdf,
    encode_image,
)
from documents_processing.prompts import (
    system_prompts,
    metadata_extraction_prompt,
    interview_metadata_extraction_prompt,
)
from llm_multiprocessing_inference import get_answers
from PIL import Image
import pytesseract
import io
from tqdm import tqdm

inference_pipelines = {
    "OpenAI": {
        "model_name": "gpt-4o-mini",
        "inference_pipeline_name": "OpenAI",
        "api_key": os.getenv("openai_api_key"),
    },
    "Ollama": {
        "model_name": "gemma3:12b-it-q4_K_M",
        "inference_pipeline_name": "Ollama",
        "api_key": None,
    },
}


def _ocr_handwritten_pdf(file_path: str) -> Dict[str, str]:
    """
    Opens a PDF file and initializes an empty list to hold the OCR-extracted text.
    Iterates through each page of the PDF, rendering it to an image.
    Uses Tesseract OCR, via pytesseract.image_to_string, to extract text from the image of each page.
    Appends the extracted text to the list and closes the PDF document after processing all pages.
    Returns the list of extracted text blocks.
    """
    # Open the PDF
    doc = fitz.open(file_path)
    text = {}

    for page_num in range(len(doc)):
        # Get the page
        page = doc.load_page(page_num)

        # Render page to an image
        pix = page.get_pixmap()
        image_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(image_bytes))

        # Use Tesseract to do OCR on the image
        text[f"page {page_num + 1}"] = pytesseract.image_to_string(image)

    doc.close()
    return text


class DocumentsDataExtractor:
    def __init__(
        self,
        inference_pipeline_name: Optional[Literal["Ollama", "OpenAI"]] = None,
        punct_model_name: str = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if inference_pipeline_name is None:
            self.inference_pipeline_name = None
            self.model_name = None
            self.api_key = None
        else:
            self.inference_pipeline_name = inference_pipelines[inference_pipeline_name][
                "inference_pipeline_name"
            ]
            self.model_name = (
                inference_pipelines[inference_pipeline_name]["model_name"] or model_name
            )
            self.api_key = (
                inference_pipelines[inference_pipeline_name]["api_key"] or api_key
            )

        self.punct_extractor = PunctCapSegModelONNX.from_pretrained(punct_model_name)

    def _clean_entries(self, entries: List[str]) -> List[str]:
        """Clean the extracted entries from a PDF document."""
        final_entries = []
        punc_entries: List[List[str]] = self.punct_extractor.infer(
            entries, apply_sbd=True
        )
        punc_entries = _flatten_list(punc_entries)

        for entry in punc_entries:
            if len(entry) > 5 and entry.count(" ") > 6 and "©" not in entry:
                final_entries.append(entry)

        return final_entries

    def _extract_pdf_text(self, filepath: str) -> Dict[str, str]:
        """Extract text from a PDF document."""
        doc = fitz.open(filepath)
        doc_text = {}
        for page in doc:
            text_blocks = page.get_textpage().extractBLOCKS()
            clean_text = [block[4] for block in text_blocks]
            clean_text = self._clean_entries(clean_text)
            doc_text[f"page {page.number + 1}"] = " ".join(clean_text)
        return doc_text

    def _get_images_description(
        self, figures_paths: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Generate image descriptions and update extracted text."""
        figs_df = pd.DataFrame()
        prompts = []
        pages = []
        for fig_type, fig_paths in figures_paths.items():
            for fig_path in fig_paths:
                one_fig_metadata = {
                    "Entry Type": f"PDF {fig_type}",
                    "entry_fig_path": fig_path,
                }
                figs_df = pd.concat([figs_df, pd.DataFrame([one_fig_metadata])])

                base64_image = encode_image(fig_path)
                if self.inference_pipeline_name == "OpenAI":  # OpenAI
                    one_fig_prompt = [
                        {"role": "system", "content": system_prompts[fig_type]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "low",
                                    },
                                }
                            ],
                        },
                    ]
                else:  # Ollama
                    one_fig_prompt = [
                        {"role": "system", "content": system_prompts[fig_type]},
                        {
                            "role": "user",
                            "images": [base64_image],
                        },
                    ]
                prompts.append(one_fig_prompt)

                file_name = fig_path.split("/")[-1].split(".")[0]
                page_number = int(file_name.split("_")[1]) + 1
                pages.append(f"page {page_number}")

        if len(prompts) > 0:
            answers = get_answers(
                prompts=prompts,
                default_response="-",
                response_type="unstructured",
                api_pipeline=self.inference_pipeline_name,
                model=self.model_name,
                api_key=self.api_key,
                show_progress_bar=False,
            )

            figs_df["text"] = [{p: a} for p, a in zip(pages, answers)]
        return figs_df

    def extract_metadata(
        self,
        metadata_pages_paths: List[os.PathLike],
        extraction_prompt: str,
        default_answer: Dict[str, str],
    ) -> Dict[str, str]:
        """Extract metadata from a PDF file."""

        metadata_dict = default_answer

        for one_page_path in metadata_pages_paths:
            base64_image = encode_image(one_page_path)

            if self.inference_pipeline_name == "OpenAI":  # OpenAI

                prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": extraction_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ]
            else:  # Ollama
                prompt = [
                    {
                        "role": "system",
                        "content": extraction_prompt,
                    },
                    {
                        "role": "user",
                        "images": [base64_image],
                    },
                ]

            try:
                answer = get_answers(
                    prompts=[prompt],
                    default_response=default_answer,
                    response_type="structured",
                    api_pipeline=self.inference_pipeline_name,
                    model=self.model_name,
                    api_key=self.api_key,
                    show_progress_bar=False,
                )[0]
            except Exception as e:
                print(f"Error in inference in metadata extraction: {e}")
                answer = default_answer

            for field in list(default_answer.keys()):
                if answer.get(field, "-") != "-":
                    metadata_dict[field] = answer[field]

            if all(value != "-" for value in metadata_dict.values()):
                return metadata_dict

        return metadata_dict

    def __call__(
        self,
        file_name: str,
        doc_folder_path: os.PathLike,
        figures_saving_path: os.PathLike,
        doc_url: Optional[str] = None,
        metadata_extraction_type: Union[
            bool, Literal["interview", "document", "none"]
        ] = "none",
        extract_figures_bool: bool = False,
        relevant_pages_for_metadata_extraction: Optional[List[int]] = None,
        return_original_pages_numbers: bool = False,
    ) -> pd.DataFrame:
        """Extract information from a document.

        Args:
            file_name (str): Name of the document file to process.
            doc_folder_path (os.PathLike): Path to the folder containing the document.
            figures_saving_path (os.PathLike): Path where extracted figures will be saved.
            doc_url (Optional[str], optional): URL to download the document if not found locally. Defaults to None.
            metadata_extraction_type (Union[bool, Literal["interview", "document", "none"]], optional): Type of metadata to extract. Defaults to "none".
            extract_figures_bool (bool, optional): Whether to extract figures from the document. Defaults to False.
            relevant_pages_for_metadata_extraction (Optional[List[int]], optional): Specific pages to extract metadata from. Defaults to None.
            return_original_pages_numbers (bool, optional): Whether to return text with original page numbers. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing extracted information from the document.
        """
        # file_name = _get_first_n_characters(file_name)
        project_extracted_text = pd.DataFrame(
            columns=["text", "Entry Type", "entry_fig_path"]
        )
        # doc_folder_path = os.path.abspath(doc_folder_path)
        doc_file_path = os.path.join(doc_folder_path, file_name)

        if not os.path.exists(doc_file_path):
            _download_pdf(doc_url, doc_file_path)

        # Convert to PDF if the file is not a PDF
        if not doc_file_path.lower().endswith(".pdf"):
            doc_folder_path = os.path.dirname(doc_file_path) + "_converted"
            os.makedirs(doc_folder_path, exist_ok=True)
            base_name = ".".join(file_name.split(".")[:-1])
            converted_pdf_path = os.path.join(doc_folder_path, f"{base_name}.pdf")
            convert_to_pdf(doc_file_path, converted_pdf_path)
            doc_file_path = converted_pdf_path

        extracted_text: Dict[str, str] = self._extract_pdf_text(doc_file_path)
        if len(" ".join(list(extracted_text.values()))) < 10:
            extracted_text: Dict[str, str] = _ocr_handwritten_pdf(doc_file_path)
        df_raw_text = pd.DataFrame(
            [
                {
                    "text": extracted_text,
                    "Entry Type": "PDF Text",
                    "entry_fig_path": "-",
                }
            ]
        )

        # get the number of pages
        n_pages = len(extracted_text)

        project_extracted_text = pd.concat([project_extracted_text, df_raw_text])

        if self.inference_pipeline_name is not None and (
            extract_figures_bool
            or metadata_extraction_type != "none"
            or metadata_extraction_type
        ):
            with SuppressPrint():
                figures_paths, metadata_pages_paths = extract_figures(
                    saved_pages_images_path=figures_saving_path,
                    pdf_file_path=doc_file_path,
                    pdf_saved_name=file_name,
                    metadata_extraction_type=metadata_extraction_type,
                    relevant_pages_for_metadata_extraction=relevant_pages_for_metadata_extraction,
                )
                if extract_figures_bool:
                    images_extracted_text = self._get_images_description(figures_paths)
                    project_extracted_text = pd.concat(
                        [project_extracted_text, images_extracted_text]
                    )

        if not return_original_pages_numbers:
            project_extracted_text["text"] = project_extracted_text["text"].apply(
                lambda x: " ".join(list(x.values()))
            )

        field_to_final_name = {
            "date": "Document Publishing Date",
            "author": "Document Source",
            "title": "Document Title",
            "interviewee": "Interviewee",
            "type": "Document Type",
            "primary_country": "Primary Country",
        }

        if self.inference_pipeline_name is not None and (
            metadata_extraction_type != "none" or metadata_extraction_type
        ):
            if metadata_extraction_type is True:
                metadata_extraction_type = "document"
            if metadata_extraction_type == "interview":
                extraction_prompt = interview_metadata_extraction_prompt
                default_answer = {
                    "date": "-",
                    "author": "-",
                    "title": "-",
                    "interviewee": "-",
                }
            else:
                extraction_prompt = metadata_extraction_prompt
                default_answer = {
                    "date": "-",
                    "author": "-",
                    "title": "-",
                    "type": "-",
                    "primary_country": "-",
                }
            metadata = self.extract_metadata(
                metadata_pages_paths, extraction_prompt, default_answer=default_answer
            )
            for field, data in metadata.items():
                project_extracted_text[field_to_final_name[field]] = str(data)

            project_extracted_text["Number of Pages"] = n_pages

        project_extracted_text["File Name"] = file_name

        return project_extracted_text
