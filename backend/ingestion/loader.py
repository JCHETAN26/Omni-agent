"""Load a directory of Markdown documents (title = first heading)."""

from __future__ import annotations

from pathlib import Path

from backend.core.models import Document


def load_markdown_dir(path: str | Path) -> list[Document]:
    docs = []
    for p in sorted(Path(path).glob("*.md")):
        text = p.read_text()
        first = next((ln for ln in text.splitlines() if ln.strip()), p.stem)
        docs.append(Document(doc_id=p.stem, title=first.lstrip("# ").strip(), text=text))
    return docs
