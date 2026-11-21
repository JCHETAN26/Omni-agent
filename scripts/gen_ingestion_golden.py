"""Golden fixtures that pin the Go ingestion pipeline to the Python one.

    python -m scripts.gen_ingestion_golden   # chunker; tokenizer too if transformers is installed

Writes:
  ingestion/internal/chunker/testdata/golden.json    chunk_document() output
  ingestion/internal/tokenizer/testdata/golden.json  all-MiniLM-L6-v2 input_ids (max_length=256)

The Go tests must reproduce both exactly: chunk IDs are the Elasticsearch and
Qdrant record IDs, and token IDs decide the embedding. tests/test_ingestion_golden.py
fails if the chunker fixture is stale.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from backend.core.models import Document
from backend.ingestion.chunker import chunk_document
from backend.ingestion.loader import load_markdown_dir

ROOT = Path(__file__).resolve().parent.parent
CHUNKER_GOLDEN = ROOT / "ingestion/internal/chunker/testdata/golden.json"
TOKENIZER_GOLDEN = ROOT / "ingestion/internal/tokenizer/testdata/golden.json"
TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"

TRICKY = [
    "# Title\n\nPlain paragraph. Second sentence! Third? Yes.",
    "No heading at all, just text that runs on and on without any punctuation " * 30,
    "## Deep\n### Deeper\nbody\n# Top again\ntext\n## Child ##\nmore",
    "```\n# not a heading\n```\n# Real heading\nafter fence\n~~~\n## also not\n~~~\nend",
    "Windows\r\nline\r\nendings\r\n\r\nand a second paragraph.\rold mac.",
    "Unicode line separators\x0bvertical\x0cform\x1cfile\x1dgroup\x1erecord\x85nel",
    "Ideographic　space. After　　full stop! And nbsp? Done.",
    "Émojis 🔥🔥 count as one character each. " * 40,
    "Accented café naïve résumé façade. 日本語のテキスト。中文文本！"
    "Ελληνικά κείμενα; русский текст.",
    "#NoSpaceIsNotAHeading\n####### seven hashes is not a heading\n#\tTab heading\nbody",
    "   \n\n  \t  \n",
    "Sentence one.   \n  Sentence two after newline.\n\n\n\nNew paragraph.  Trailing spaces.   ",
    "x" * 2000,
    "word " * 500,
    ("A fairly long sentence that keeps going to test the packing logic. " * 12 + "\n\n") * 4,
]


def _random_docs(n: int, seed: int = 11) -> list[Document]:
    rng = random.Random(seed)
    words = "disk error firmware rack cooling kafka broker index shard replica latency".split()
    seps = [" ", " ", " ", ". ", "! ", "? ", "\n", "\n\n", "\n\n\n", ".　", " \t "]
    docs = []
    for i in range(n):
        parts = []
        for _ in range(rng.randint(5, 400)):
            if rng.random() < 0.03:
                parts.append(f"\n{'#' * rng.randint(1, 4)} {rng.choice(words).title()}\n")
            parts.append(rng.choice(words) + rng.choice(seps))
        docs.append(Document(doc_id=f"rand-{i}", title=f"Random {i}", text="".join(parts)))
    return docs


def chunker_cases() -> list[dict]:
    docs = load_markdown_dir(ROOT / "samples")
    docs += [Document(doc_id=f"tricky-{i}", title=f"T{i}", text=t) for i, t in enumerate(TRICKY)]
    docs += _random_docs(40)
    cases = []
    for size, overlap in ((800, 120), (200, 50), (64, 0)):
        for d in docs:
            chunks = chunk_document(d, size, overlap)
            cases.append(
                {
                    "doc": d.model_dump(include={"doc_id", "title", "text"}),
                    "size": size,
                    "overlap": overlap,
                    "chunks": [c.payload() for c in chunks],
                }
            )
    return cases


def tokenizer_cases() -> list[dict]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    texts = list(TRICKY) + [
        "Hello, World! It's 3.14 o'clock -- (really) [sic] {ok} <tag> a/b\\c @user #tag $5 100%",
        "İstanbul ǅ ß ﬁ Æ œ Ⅻ ①",
        "zero​width‍joiner﻿bom and\u0000null�replacement",
        "supercalifragilisticexpialidocious " + "a" * 120 + " pneumonoultramicroscopic",
        "naïve coöperate é (combining) Å Å",
        "مرحبا بالعالم שלום עולם नमस्ते दुनिया",
        "",
        "UPPER lower MiXeD 12345 3,000.50 v2.0.1 e-mail_address foo@bar.com",
    ]
    for d in _random_docs(10, seed=5):
        texts.append(d.text)
    out = []
    for t in texts:
        enc = tok(t, truncation=True, max_length=256)
        out.append({"text": t, "input_ids": enc["input_ids"]})
    return out


def main() -> None:
    CHUNKER_GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    CHUNKER_GOLDEN.write_text(json.dumps(chunker_cases(), ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {CHUNKER_GOLDEN}")
    try:
        cases = tokenizer_cases()
    except ImportError:
        print("transformers not installed: tokenizer golden not regenerated")
        return
    TOKENIZER_GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    TOKENIZER_GOLDEN.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {TOKENIZER_GOLDEN} ({len(cases)} texts)")


if __name__ == "__main__":
    main()
