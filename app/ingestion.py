from pathlib import Path
from typing import Union
from uuid import uuid4

import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_document(file: Union[str, Path, object]) -> dict:
    """
    Process a PDF document from file path or uploaded file-like object.

    Args:
        file (str | Path | UploadedFile): Path to PDF or file-like object.

    Returns:
        dict: Contains metadata and list of chunk dicts.
    """
    doc_id = str(uuid4())

    # Determine input type
    if isinstance(file, (str, Path)):
        pdf = pymupdf.open(str(file))  # Load from path
    else:
        pdf = pymupdf.open(
            stream=file.file.read(), filetype="pdf"
        )  # Streamed file upload

    # Extract text from all pages
    all_text = ""
    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        text = page.get_text("text")
        all_text += f"\n\nPage {page_num + 1}\n{text}"

    # Chunk the text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(all_text)

    # Formatting and metadata
    chunked_docs = []
    for i, chunk in enumerate(chunks):
        chunked_docs.append(
            {"chunk_id": f"{doc_id}_{i}", "text": chunk, "doc_id": doc_id}
        )

    return {"num_chunks": len(chunked_docs), "doc_id": doc_id, "chunks": chunked_docs}
