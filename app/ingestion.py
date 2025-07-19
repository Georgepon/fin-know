import hashlib
from pathlib import Path
from typing import Union
from uuid import uuid4

import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_document(file: Union[str, Path, object]) -> dict:
    """Process a PDF document and return chunk metadata.

    Args:
        file: Path to the PDF or an uploaded file-like object.

    Returns:
        dict: Metadata including the chunks and a hash of the file contents.
    """
    doc_id = str(uuid4())
    file_content_bytes = None
    file_hash = None
    original_filename = "Unknown Document"  # Default

    # Determine input type and read bytes for hashing
    if isinstance(file, (str, Path)):
        file_path = Path(file)
        original_filename = file_path.name  # Extract filename
        file_content_bytes = file_path.read_bytes()
        pdf = pymupdf.open(str(file_path))  # Load from path
    else:  # Handle UploadedFile (Streamlit/FastAPI)
        if hasattr(file, "name"):  # Get filename from UploadedFile
            original_filename = file.name

        # Check for Streamlit UploadedFile
        if hasattr(file, "getvalue") and callable(file.getvalue):
            file_content_bytes = file.getvalue()
            # Reset pointer for pymupdf if needed
            file.seek(0)
            pdf = pymupdf.open(stream=file_content_bytes, filetype="pdf")
        # Check for FastAPI UploadFile
        elif hasattr(file, "file") and hasattr(file.file, "read"):
            file_content_bytes = file.file.read()
            # Reset pointer for pymupdf
            file.file.seek(0)
            pdf = pymupdf.open(stream=file_content_bytes, filetype="pdf")
        else:
            raise TypeError("Unsupported file input type")

    # Calculate hash
    if file_content_bytes:
        file_hash = hashlib.sha256(file_content_bytes).hexdigest()

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
            {
                "chunk_id": f"{doc_id}_{i}",
                "text": chunk,
                "doc_id": doc_id,
                "filename": original_filename,  # Add filename to metadata
            }
        )

    return {
        "num_chunks": len(chunked_docs),
        "doc_id": doc_id,
        "chunks": chunked_docs,
        "file_hash": file_hash,
    }
