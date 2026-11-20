"""Load BEIR-format datasets (corpus.jsonl, queries.jsonl, qrels/<split>.tsv).

CQADupStack ships one directory per StackExchange forum. Ids are only unique
within a forum, so every document and query id is prefixed with ``<forum>/``.
That lets several forums share one OmniAgent index without collisions.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen

from backend.core.models import Document

CQADUPSTACK_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip"
)
CQADUPSTACK_FORUMS = (
    "android", "english", "gaming", "gis", "mathematica", "physics",
    "programmers", "stats", "tex", "unix", "webmasters", "wordpress",
)  # fmt: skip


@dataclass
class Dataset:
    queries: dict[str, str] = field(default_factory=dict)  # query_id -> text
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)  # query_id -> {doc_id: rel}
    forum_of: dict[str, str] = field(default_factory=dict)  # query_id -> forum
    corpus_files: list[tuple[str, Path]] = field(default_factory=list)  # (forum, corpus.jsonl)

    def documents(self) -> Iterator[Document]:
        """Stream the corpus so a 450k-document collection never sits in memory twice."""
        for forum, path in self.corpus_files:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    yield Document(
                        doc_id=f"{forum}/{row['_id']}",
                        title=row.get("title") or "",
                        text=row.get("text") or "",
                        metadata={"forum": forum},
                    )

    def count_documents(self) -> int:
        return sum(sum(1 for _ in path.open(encoding="utf-8")) for _, path in self.corpus_files)


def load(root: Path, forums: Sequence[str], split: str = "test") -> Dataset:
    data = Dataset()
    for forum in forums:
        base = root / forum
        if not (base / "corpus.jsonl").exists():
            raise FileNotFoundError(f"missing BEIR corpus for {forum!r} under {root}")
        data.corpus_files.append((forum, base / "corpus.jsonl"))
        texts = {}
        with (base / "queries.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                texts[row["_id"]] = row["text"]
        with (base / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            next(reader)  # header: query-id, corpus-id, score
            for query_id, doc_id, score in reader:
                qid = f"{forum}/{query_id}"
                data.qrels.setdefault(qid, {})[f"{forum}/{doc_id}"] = int(score)
                data.queries[qid] = texts[query_id]
                data.forum_of[qid] = forum
    return data


def _fetch(url: str, target: Path) -> str:
    """Stream ``url`` to ``target`` atomically; return the file's SHA-256."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with urlopen(url, timeout=120) as response, partial.open("wb") as out:
        while block := response.read(1 << 20):
            digest.update(block)
            out.write(block)
    partial.rename(target)
    return digest.hexdigest()


def _hf_revision(repo: str) -> str:
    with urlopen(f"https://huggingface.co/api/datasets/{repo}", timeout=60) as response:
        return json.load(response)["sha"]


def download_cqadupstack(dest: Path, forums: Sequence[str]) -> Path:
    """Fetch each forum's BEIR files from the MTEB mirror on the Hugging Face Hub.

    The original BEIR archive host (``CQADUPSTACK_URL``) has been unreachable, and
    MTEB's ``mteb/cqadupstack-<forum>`` repositories carry the same BEIR files
    (corpus.jsonl, queries.jsonl, qrels/test.tsv). The dataset commit and every
    file's SHA-256 are written to ``manifest.json`` so a run can be reproduced.
    """
    root = dest / "cqadupstack"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for forum in forums:
        base = root / forum
        if (base / "corpus.jsonl").exists():
            continue  # already downloaded (or supplied locally)
        repo = f"mteb/cqadupstack-{forum}"
        revision = _hf_revision(repo)
        files = {}
        for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
            files[name] = _fetch(url, base / name)
        manifest[forum] = {"repo": repo, "revision": revision, "sha256": files}
        root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return root
