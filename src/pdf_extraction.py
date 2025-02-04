import sys
import contextlib
import os
import requests
import fitz
import base64
import pandas as pd
from typing import List, Dict, Literal
from ast import literal_eval
from tqdm import tqdm
from src.figures_extraction import extract_figures, _extract_first_page_to_base64
from nltk.tokenize import sent_tokenize
from punctuators.models import PunctCapSegModelONNX
from src.model_inference import VLMInferencePipeline, OpenAIInferencePipeline


image_description_prompt = """
I'm going to provide you an image from a humanitarian report. Your objective is to create a text that covers all the analytical information present in the image.
More specificaly, all numbers, specific populations, locations, and any other relevant information should be included in the text.
Make the text as detailed as possible. Use a descriptive language and only mention the information that is present in the image as informative text and information.
The text should be in the form of self-contained paragraphs, and should not be a list of bullet points. Do not provide any general description of the image or introductory text. Instead, directly present the information.
When providing information, present it without reffering the image. For example, instead of saying "The age and gender breakdown of the population indicates that 1% are aged 60 years or above", say "1% are aged 60 years or above".
If the image does not contain any relevant information for the humanitarian report (faces, landscapes, logos, etc.), return an empty string ('-').
Return all text in English.
"""

table_description_prompt = """
I'm going to provide you with atable. Your objective is to create a text that covers all the information present in the table.
More specificaly, all numbers, specific populations, locations, and any other relevant information should be included in the text.
Make the text as detailed as possible. Use a descriptive language and only mention the information that is present in the table as informative text and information.
The text should be in the form of self-contained paragraphs, and should not be a list of bullet points. Do not provide any general description of the table or introductory text. Instead, directly present the information.
When providing information, present it without reffering the table. For example, instead of saying "The age and gender breakdown of the population indicates that 1% are aged 60 years or above", say "1% are aged 60 years or above".
Return all text in English.
"""

metadata_extraction_prompt = """This is a page of a document. I want to extract the document metadata from this page.
Extract the document publishing date, author organisations and the document title. 
Return only the results in a dictionnary JSON response without unnecessary spaces in the following format:
{
    "date": dd/mm/yyyy,
    "author": List[str]: The author organisations,
    "title": str: The title of the document
} 
If you cannot find any of the information, leave the field empty ('-').
Extract the information yourself and do not rely on any external library."""


system_prompts = {
    "Picture": image_description_prompt,
    "Table": table_description_prompt,
}


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


class PDFExtractor:
    def __init__(
        self,
        model_name: str,
        punct_model_name: str = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
        api_key: str = None,
        inference_pipeline_name: Literal["VLM", "OpenAI"] = "VLM",
    ):
        self.inference_pipeline = (
            VLMInferencePipeline(model_name)
            if inference_pipeline_name == "VLM"
            else OpenAIInferencePipeline(model_name, api_key)
        )

        self.punct_extractor = PunctCapSegModelONNX.from_pretrained(punct_model_name)

    @staticmethod
    def _flatten_list(l: List[List[str]]) -> List[str]:
        """Flatten a list of lists into a single list."""
        return [item for sublist in l for item in sublist]

    def _clean_entries(self, entries: List[str]) -> List[str]:
        """Clean the extracted entries from a PDF document."""
        final_entries = []
        punc_entries: List[List[str]] = self.punct_extractor.infer(
            entries, apply_sbd=True
        )
        punc_entries = self._flatten_list(punc_entries)

        for entry in punc_entries:
            if len(entry) > 5 and entry.count(" ") > 6 and "©" not in entry:
                final_entries.append(entry)

        return final_entries

    @staticmethod
    def _get_first_n_words(s: str, n: int = 7) -> str:
        """Get the first N words of a string and create a filename."""
        return (
            " ".join(s.split()[:n])
            .replace("/", "-")
            .replace(":", "_")
            .replace(" ", "-")
            + ".pdf"
        )

    @staticmethod
    def _download_pdf(url: str, filepath: str):
        """Download a PDF file from a URL and save it."""
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")

    def _extract_pdf_text(self, filepath: str) -> List[str]:
        """Extract text from a PDF document."""
        doc = fitz.open(filepath)
        doc_text = []
        for page in doc:
            text_blocks = page.get_textpage().extractBLOCKS()
            clean_text = [block[4] for block in text_blocks]
            clean_text = self._clean_entries(clean_text)
            doc_text.extend(clean_text)
        return " ".join(doc_text)

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode an image to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _get_images_description(
        self, figures_paths: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Generate image descriptions and update extracted text."""
        figs_df = pd.DataFrame()
        prompts = []
        for fig_type, fig_paths in figures_paths.items():
            for fig_path in fig_paths:
                one_fig_metadata = {
                    "Entry Type": f"PDF {fig_type}",
                    "entry_fig_path": fig_path,
                }
                figs_df = pd.concat([figs_df, pd.DataFrame([one_fig_metadata])])

                base64_image = self.encode_image(fig_path)
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
                prompts.append(one_fig_prompt)

        answers: List[str] = []
        for prompt in prompts:
            try:
                one_entry_answer = self.inference_pipeline.inference(
                    prompt, structured_output=False
                )

            except Exception as e:
                print(f"Error in VLM inference: {e}")
                one_entry_answer = "-"

            answers.append(one_entry_answer)

        figs_df["text"] = answers
        return figs_df

    def extract_metadata(
        self,
        metadata_pages_paths: List[os.PathLike],
        default_answer: Dict[str, str] = {"date": "-", "author": "-", "title": "-"},
    ) -> dict:
        """Extract metadata from a PDF file."""

        metadata_dict = default_answer

        for one_page_path in metadata_pages_paths:
            base64_image = self.encode_image(one_page_path)

            prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": metadata_extraction_prompt},
                    ],
                },
            ]

            try:
                answer = self.inference_pipeline.inference(
                    prompt, structured_output=True
                )
            except Exception as e:
                print(f"Error in VLM inference: {e}")
                answer = default_answer

            for field in ["date", "author", "title"]:
                if answer.get(field, "-") != "-":
                    metadata_dict[field] = answer[field]

            if all(metadata_dict.values() != "-"):
                return metadata_dict

        return metadata_dict

    def __call__(
        self,
        pdf_file_name: str,
        pdf_doc_folder_path: os.PathLike,
        pdf_url: str = None,
        extract_metadata_bool: bool = False,
        extract_figures_bool: bool = False,
    ) -> pd.DataFrame:
        """Extract information from a PDF document."""
        project_extracted_text = pd.DataFrame(
            columns=["text", "Entry Type", "entry_fig_path"]
        )
        pdf_file_path = os.path.join(pdf_doc_folder_path, pdf_file_name)

        figures_saving_path = os.path.join(pdf_doc_folder_path, "..", "figures")

        if not os.path.exists(pdf_file_path):
            self._download_pdf(pdf_url, pdf_file_path)

        extracted_text = self._extract_pdf_text(pdf_file_path)
        df_raw_text = pd.DataFrame(
            [{"text": extracted_text, "Entry Type": "PDF Text", "entry_fig_path": "-"}]
        )

        project_extracted_text = pd.concat([project_extracted_text, df_raw_text])

        with SuppressPrint():
            figures_paths, metadata_pages_paths = extract_figures(
                saved_pages_images_path=figures_saving_path,
                pdf_file_path=pdf_file_path,
                pdf_saved_name=pdf_file_name,
            )
            if extract_figures_bool:
                images_extracted_text = self._get_images_description(figures_paths)
                project_extracted_text = pd.concat(
                    [project_extracted_text, images_extracted_text]
                )

        field_to_final_name = {
            "date": "Document Publishing Date",
            "author": "Document Source",
            "title": "Document Title",
        }

        if extract_metadata_bool:
            default_answer = {
                "date": "-",
                "author": "-",
                "title": "-",
            }
            metadata = self.extract_metadata(
                metadata_pages_paths, default_answer=default_answer
            )
            for field, data in metadata.items():
                project_extracted_text[field_to_final_name[field]] = str(data)

        return project_extracted_text
