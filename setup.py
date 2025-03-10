from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="documents-processing",
    version="0.1.0",
    author="MMP Team",
    author_email="your.email@example.com",
    description="A Python library for extracting and processing information from PDF documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/documents-processing",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pdf2image",
        "numpy",
        "opencv-python",
        "pillow",
        "torch",
        "transformers",
        "requests",
        "PyMuPDF",
        "pandas",
        "tqdm",
        "nltk",
        "flask",
        "supervision",
        "ultralytics",
        "punctuators",
        "llm-multiprocessing-inference",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
        ],
    },
    include_package_data=True,
)