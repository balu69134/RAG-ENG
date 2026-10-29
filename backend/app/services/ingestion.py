from dataclasses import dataclass
from io import BytesIO
import re

from pypdf import PdfReader

from backend.app.config import Settings


class DocumentProcessingError(ValueError):
    """Raised when an uploaded document cannot be processed."""


@dataclass(frozen=True)
class TextChunk:
    text: str
    page_number: int
    chunk_index: int


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract non-empty text from a PDF, retaining one-based page numbers."""
    try:
        document = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise DocumentProcessingError("The uploaded file is not a readable PDF.") from exc

    if document.is_encrypted:
        raise DocumentProcessingError("Password-protected PDFs are not supported.")

    pages: list[tuple[int, str]] = []
    try:
        for page_number, page in enumerate(document.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            pages.append((page_number, text))
    except Exception as exc:
        raise DocumentProcessingError("The PDF could not be read completely.") from exc

    if not any(text for _, text in pages):
        raise DocumentProcessingError(
            "No extractable text was found. Scanned PDFs need OCR before upload."
        )
    return pages


def _split_page(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text near whitespace while keeping a bounded overlap between chunks."""
    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break

        next_start = max(end - overlap, start + 1)
        # Avoid starting the next chunk in the middle of a word.
        whitespace = text.find(" ", next_start, min(next_start + 100, text_length))
        start = whitespace + 1 if whitespace >= 0 else next_start

    return chunks


def chunk_pages(pages: list[tuple[int, str]], settings: Settings) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    chunk_index = 0
    overlap = min(settings.chunk_overlap, settings.chunk_size // 2)
    for page_number, page_text in pages:
        for text in _split_page(page_text, settings.chunk_size, overlap):
            chunks.append(
                TextChunk(text=text, page_number=page_number, chunk_index=chunk_index)
            )
            chunk_index += 1
    return chunks
