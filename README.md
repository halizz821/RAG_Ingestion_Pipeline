# RAG Ingestion Pipeline 

This project implements a **PDF ingestion pipeline for
Retrieval-Augmented Generation (RAG)**. It automatically parses PDF
documents, intelligently chunks them, enriches them using an LLM
(including tables and images), and stores the final representations in a
**Chroma vector database** for downstream RAG applications.

The pipeline is designed for **high-quality document understanding**,
including: - Text extraction\
- Table structure preservation\
- Image extraction and AI-based interpretation\
- Search-optimized chunk generation\
- Persistent vector storage

------------------------------------------------------------------------

## Overview

The ingestion process follows these steps:

1.  **PDF Parsing** using `unstructured`
2.  **Smart Chunking** using title-based segmentation
3.  **Content-Type Separation** (text, tables, images)
4.  **AI-Enhanced Summarization** for multimodal chunks
5.  **Embedding Generation** using OpenAI embeddings
6.  **Vector Storage** in a persistent Chroma database

This prepares your documents for accurate and efficient **RAG-based
question answering**.

------------------------------------------------------------------------

## Dependencies

All Python dependencies are defined in:

``` bash
pyproject.toml
```

You **do not need to manually install individual Python packages**. Just
follow the installation steps below.

------------------------------------------------------------------------

## System Requirements (IMPORTANT)

This pipeline depends on **external system tools** for PDF parsing and
OCR:

-   **Poppler** (for PDF rendering)
-   **Tesseract OCR** (for optical character recognition)

You must install these **before running the pipeline**.

------------------------------------------------------------------------

## Environment Variables

Create a `.env` file in the project root:

``` bash
OPENAI_API_KEY="your_api_key_here"
```

------------------------------------------------------------------------

## Installation

This project uses **uv** for environment and dependency management.

1.  Install `uv`:

``` bash
pip install uv
```

2.  Sync the environment:

``` bash
uv sync
```

This will automatically install all dependencies from `pyproject.toml`.

------------------------------------------------------------------------

## Folder Structure

Place all PDF files inside a folder named:

``` bash
./docs
```

Example structure:

``` text
project-root/
│
├── RAG_Ingestion.py
├── pyproject.toml
├── .env
├── docs/
│   ├── file1.pdf
│   ├── file2.pdf
│
└── db/
```

------------------------------------------------------------------------

## How to Run

Run the ingestion pipeline using:

``` bash
python RAG_Ingestion.py
```

By default: - PDFs are loaded from:

``` bash
./docs
```

-   The Chroma database is saved to:

``` bash
db/chroma_db_2
```

You can change these paths in:

``` python
build_vectorstore(path_folder="./docs", persist_dir="db/chroma_db_2")
```

------------------------------------------------------------------------

## Output

After execution, the pipeline produces:

-   A **persistent Chroma vector database**
-   AI-enriched document chunks
-   Embedded vectors ready for **RAG querying**


------------------------------------------------------------------------

## Key Features

-   High-resolution PDF parsing\
-   Structure-aware chunking\
-   AI-enriched multimodal summaries\
-   Metadata-preserving document storage\
-   Optimized for semantic search\
-   Persistent vector storage with Chroma

------------------------------------------------------------------------

## Notes

-   Large PDF files may take several minutes to process.

------------------------------------------------------------------------
## Acknowledgements
Thanks to  [harishneel1](https://github.com/harishneel1) that inspired this project.
