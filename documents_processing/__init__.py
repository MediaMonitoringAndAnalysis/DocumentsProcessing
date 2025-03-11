"""
DocumentsProcessing: A Python library for extracting and processing information from PDF documents.

This library provides tools for extracting text, figures, tables, and metadata from PDF documents
using advanced vision-language models.
"""

from documents_processing.documents_data_extraction import DocumentsDataExtractor
from documents_processing.utils import supported_file_extensions

__version__ = "0.1.0"
__all__ = ["DocumentsDataExtractor", "supported_file_extensions"]