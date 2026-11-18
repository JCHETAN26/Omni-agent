"""Structure-aware chunking.

Strategy (cheap and deterministic, no model calls):
1. Split the document into sections on Markdown headings; each chunk records
   its heading path (e.g. "Storage > Disk Errors") as `section_heading`.
   Headings inside fenced code blocks are ignored.
2. Inside a section, greedily pack paragraphs up to `chunk_size` characters.
   Oversized paragraphs fall back to sentence splitting, then hard splits.
3. Adjacent chunks within the same section share `chunk_overlap` characters
   (cut on a word boundary) so answers spanning a boundary stay retrievable.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from backend.core.models import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CHUNK_NS = uuid.UUID("6f1c2a8e-3b7d-4c55-9a0e-0f6d2b9e7a11")


@dataclass
class _Section:
    heading: str
    text: str


def _split_sections(doc: Document) -> list[_Section]:
    sections: list[_Section] = []
    path: list[tuple[int, str]] = []  # (level, title) stack
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            heading = " > ".join(t for _, t in path) or doc.title
            sections.append(_Section(heading, body))
        buf.clear()

    for line in doc.text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _hard_split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _units(section_text: str, size: int) -> list[str]:
    """Break a section into pieces that each fit within `size`."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", section_text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        for sent in _SENTENCE_RE.split(para):
            units.extend([sent] if len(sent) <= size else _hard_split(sent, size))
    return units


def _tail(text: str, n: int) -> str:
    """Last ~n characters of text, starting on a word boundary."""
    if n <= 0 or len(text) <= n:
        return text if n > 0 else ""
    cut = text[-n:]
    space = cut.find(" ")
    return cut[space + 1 :] if 0 <= space < len(cut) - 1 else cut


def _pack(units: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            carry = _tail(current, overlap)
            current = f"{carry}\n\n{unit}" if carry and len(carry) + len(unit) + 2 <= size else unit
        else:
            current = unit
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: Document, chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    out: list[Chunk] = []
    for section in _split_sections(doc):
        for text in _pack(_units(section.text, chunk_size), chunk_size, chunk_overlap):
            position = len(out)
            out.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(_CHUNK_NS, f"{doc.doc_id}:{position}")),
                    doc_id=doc.doc_id,
                    document_title=doc.title,
                    section_heading=section.heading,
                    position=position,
                    text=text,
                )
            )
    return out
