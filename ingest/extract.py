"""Extract plain text from uploaded documents.

Supports .txt/.md (read), .docx (python-docx), and .pdf (pypdf). Returns a
trimmed string; unknown types raise ValueError so the route can report it.
"""
from __future__ import annotations

import io
import os

_MAX_CHARS = 60_000  # cap how much reference text we feed into the brief


def extract_text(filename: str, data: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".txt", ".md", ".markdown"):
        text = data.decode("utf-8", errors="replace")
    elif ext == ".docx":
        text = _from_docx(data)
    elif ext == ".pdf":
        text = _from_pdf(data)
    else:
        raise ValueError(f"Unsupported file type: {ext or '(none)'}")
    return text.strip()[:_MAX_CHARS]


def _from_docx(data: bytes) -> str:
    from docx import Document  # python-docx

    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # one bad page shouldn't sink the whole upload
            continue
    return "\n\n".join(pages)
