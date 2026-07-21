import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def process_pdf(uploaded_file, chunk_size=500, chunk_overlap=50, enable_ocr=True):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    documents = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        if not text.strip() and enable_ocr:
            text = f"[OCR Extracted text from Page {page_num + 1}]"

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_num + 1}
                )
            )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks
