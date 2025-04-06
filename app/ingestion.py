from uuid import uuid4

import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_document(file):
    """_summary_

    Args:
        file (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Load and read the PDF
    doc_id = str(uuid4())
    pdf = pymupdf.open(stream=file.file.read(), filetype="pdf")

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
