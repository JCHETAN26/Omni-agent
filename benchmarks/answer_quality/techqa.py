"""TechQA-RAG-Eval loader (nvidia/TechQA-RAG-Eval, Apache-2.0).

910 real IBM technical-support forum questions over a corpus of 28,482 IBM
technotes. 610 are answerable, each with a reference answer taken from exactly
one gold technote; 300 are not answerable from the corpus (``is_impossible``),
which is what tests abstention.

Expected layout after ``fetch``::

    <dir>/train.json          questions, answers, is_impossible, gold contexts
    <dir>/corpus/*.txt        one technote per file ("Title: ...\\n\\nText:\\n ...")
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from backend.core.models import Document

DATASET = "nvidia/TechQA-RAG-Eval"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
# sha256 of the files this evaluation was run against.
SHA256 = {
    "train.json": "69d97231509482ed6bd5ec1c4bc0607acb82a88d11169eb8383592d0ca8b93c7",
    "corpus.zip": "c06aa287dcc1abf8db6b49b8495df095db73342d729f6451ac330785245d10be",
}
_TITLE_SUFFIX = re.compile(r"\s+-\s+United States\s*$")


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    reference: str
    answerable: bool
    gold_docs: tuple[str, ...]  # technote ids (file stems) holding the answer


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(directory: Path) -> None:
    """Download and verify the dataset, then unpack the corpus."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in SHA256.items():
        path = directory / name
        if not path.exists() or _sha256(path) != digest:
            with urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
                path.write_bytes(response.read())
        if (got := _sha256(path)) != digest:
            raise ValueError(f"{name}: sha256 {got} != expected {digest}")
    corpus = directory / "corpus"
    if not corpus.is_dir():
        with zipfile.ZipFile(directory / "corpus.zip") as archive:
            for member in archive.namelist():
                # Only plain corpus/<name>.txt members; never write outside `directory`.
                if not re.fullmatch(r"corpus/[A-Za-z0-9_.-]+\.txt", member):
                    continue
                (corpus / Path(member).name).parent.mkdir(exist_ok=True)
                (corpus / Path(member).name).write_bytes(archive.read(member))


def load_questions(directory: Path) -> list[Question]:
    rows = json.loads((directory / "train.json").read_text())
    out = []
    for row in rows:
        answerable = not bool(row["is_impossible"])
        gold = tuple(Path(c["filename"]).stem for c in row["contexts"]) if answerable else ()
        out.append(
            Question(
                qid=row["id"],
                question=row["question"].strip(),
                reference=row["answer"].strip() if answerable else "",
                answerable=answerable,
                gold_docs=gold,
            )
        )
    return out


def parse_technote(doc_id: str, raw: str) -> Document:
    title, _, body = raw.partition("\n")
    title = _TITLE_SUFFIX.sub("", title.removeprefix("Title:").strip()) or doc_id
    body = body.strip()
    body = body.removeprefix("Text:").strip()
    return Document(doc_id=doc_id, title=title, text=body or title)


def load_corpus(directory: Path) -> list[Document]:
    return [
        parse_technote(path.stem, path.read_text(errors="replace"))
        for path in sorted((directory / "corpus").glob("*.txt"))
    ]
