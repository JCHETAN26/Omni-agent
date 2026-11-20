"""Learning-to-rank stage: a small PyTorch scorer over hybrid-retrieval features.

Candidates come from hybrid retrieval (BM25 + dense, RRF-fused). Each one gets
query-relative features: lexical and dense scores and ranks, the RRF score, the
cross-encoder score when the reranker ran, and cheap query/title overlap
signals. An MLP trained with a listwise (ListNet) loss reorders the list.

The saved model records its feature list, normalisation statistics, and
whether it needs cross-encoder scores. Serving checks that contract, so a model
is never run on features it was not trained with.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.core.models import SearchHit

FEATURES = (
    "sparse_score_norm",  # BM25 / best BM25 in the list
    "has_sparse",
    "sparse_rr",  # 1 / BM25 rank
    "sparse_z",  # BM25 z-score within the list
    "dense_score",  # cosine similarity
    "has_dense",
    "dense_rr",  # 1 / dense rank
    "dense_gap",  # cosine - best cosine in the list
    "rrf_norm",  # RRF / best RRF in the list
    "ce_score",  # cross-encoder logit
    "ce_gap",  # logit - best logit in the list
    "title_overlap",  # share of query terms in the document title
    "text_overlap",  # share of query terms in the chunk text
    "log_length",  # log(1 + chunk tokens)
    "position",  # chunk order inside its document, capped
)
CE_FEATURES = {"ce_score", "ce_gap"}

_TOKEN = re.compile(r"[a-z0-9_]+")


def _terms(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def features(
    query: str, hits: Sequence[SearchHit], ce_scores: Sequence[float | None] | None = None
) -> np.ndarray:
    """One row per hit, columns in FEATURES order."""
    n = len(hits)
    rows = np.zeros((n, len(FEATURES)), dtype="float32")
    if not n:
        return rows
    ce = list(ce_scores) if ce_scores is not None else [None] * n
    sparse = [h.sparse_score for h in hits if h.sparse_score is not None]
    dense = [h.dense_score for h in hits if h.dense_score is not None]
    ce_known = [s for s in ce if s is not None]
    best_sparse = max(sparse) if sparse else 0.0
    mean_sparse = float(np.mean(sparse)) if sparse else 0.0
    std_sparse = float(np.std(sparse)) if len(sparse) > 1 else 0.0
    best_dense = max(dense) if dense else 0.0
    best_rrf = max(h.score for h in hits) or 1.0
    best_ce = max(ce_known) if ce_known else 0.0
    qterms = _terms(query) or {""}
    for i, (hit, ce_score) in enumerate(zip(hits, ce, strict=True)):
        chunk = hit.chunk
        if hit.sparse_score is not None:
            rows[i, 0] = hit.sparse_score / best_sparse if best_sparse > 0 else 0.0
            rows[i, 1] = 1.0
            rows[i, 2] = 1.0 / hit.sparse_rank if hit.sparse_rank else 0.0
            rows[i, 3] = (hit.sparse_score - mean_sparse) / std_sparse if std_sparse else 0.0
        if hit.dense_score is not None:
            rows[i, 4] = hit.dense_score
            rows[i, 5] = 1.0
            rows[i, 6] = 1.0 / hit.dense_rank if hit.dense_rank else 0.0
            rows[i, 7] = hit.dense_score - best_dense
        rows[i, 8] = hit.score / best_rrf
        if ce_score is not None:
            rows[i, 9] = ce_score
            rows[i, 10] = ce_score - best_ce
        rows[i, 11] = len(qterms & _terms(chunk.document_title)) / len(qterms)
        text_terms = _terms(chunk.text)
        rows[i, 12] = len(qterms & text_terms) / len(qterms)
        rows[i, 13] = math.log1p(len(chunk.text.split()))
        rows[i, 14] = min(chunk.position, 10) / 10
    return rows


def _mlp(n_features: int, hidden: Sequence[int]):
    import torch

    layers: list = []
    width = n_features
    for size in hidden:
        layers += [torch.nn.Linear(width, size), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
        width = size
    layers.append(torch.nn.Linear(width, 1))
    return torch.nn.Sequential(*layers)


@dataclass
class QueryGroup:
    """Training rows for one query: features and graded labels for its candidates."""

    x: np.ndarray
    y: np.ndarray


class LtrModel:
    def __init__(
        self, net, mean: np.ndarray, std: np.ndarray, hidden: Sequence[int], uses_ce: bool
    ):
        self.net = net
        self.mean = mean.astype("float32")
        self.std = std.astype("float32")
        self.hidden = list(hidden)
        self.uses_ce = uses_ce
        self.columns = [i for i, f in enumerate(FEATURES) if uses_ce or f not in CE_FEATURES]

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        return ((x[:, self.columns] - self.mean) / self.std).astype("float32")

    def score(self, x: np.ndarray) -> np.ndarray:
        import torch

        if not len(x):
            return np.zeros(0, dtype="float32")
        self.net.eval()
        with torch.inference_mode():
            return self.net(torch.from_numpy(self._prepare(x))).squeeze(-1).numpy()

    # -- persistence -------------------------------------------------------------

    def save(self, directory: Path) -> None:
        import torch

        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), directory / "model.pt")
        meta = {
            "features": [FEATURES[i] for i in self.columns],
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "hidden": self.hidden,
            "uses_ce": self.uses_ce,
        }
        (directory / "model.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> LtrModel:
        import torch

        meta = json.loads((directory / "model.json").read_text())
        model = cls(
            None, np.array(meta["mean"]), np.array(meta["std"]), meta["hidden"], meta["uses_ce"]
        )
        if meta["features"] != [FEATURES[i] for i in model.columns]:
            raise ValueError("saved LTR model was trained on a different feature set")
        net = _mlp(len(model.columns), model.hidden)
        net.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
        model.net = net
        return model


def _ndcg10(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")[:10]
    discounts = 1 / np.log2(np.arange(2, 12))
    dcg = float((labels[order] * discounts[: len(order)]).sum())
    ideal = np.sort(labels)[::-1][:10]
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    return dcg / idcg if idcg else 0.0


def train(
    train_groups: Sequence[QueryGroup],
    dev_groups: Sequence[QueryGroup],
    *,
    uses_ce: bool,
    hidden: Sequence[int] = (64, 32),
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 13,
    log=print,
) -> tuple[LtrModel, dict]:
    """ListNet training with early stopping on dev nDCG@10 (candidate-level)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    useful = [g for g in train_groups if g.y.max() > 0]
    if not useful:
        raise ValueError("no training query has a relevant candidate")
    columns = [i for i, f in enumerate(FEATURES) if uses_ce or f not in CE_FEATURES]
    stacked = np.concatenate([g.x for g in useful])[:, columns]
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    model = LtrModel(_mlp(len(columns), hidden), mean, std, hidden, uses_ce)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=lr, weight_decay=1e-4)
    prepared = [
        (torch.from_numpy(model._prepare(g.x)), torch.from_numpy(g.y.astype("float32")))
        for g in useful
    ]

    def dev_ndcg() -> float:
        judged = [g for g in dev_groups if g.y.max() > 0]
        return float(np.mean([_ndcg10(model.score(g.x), g.y) for g in judged])) if judged else 0.0

    best, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.net.train()
        random.shuffle(prepared)
        total = 0.0
        for x, y in prepared:
            logits = model.net(x).squeeze(-1)
            target = torch.softmax(torch.where(y > 0, y, torch.full_like(y, -1e4)), dim=0)
            loss = -(target * torch.log_softmax(logits, dim=0)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        score = dev_ndcg()
        history.append({"epoch": epoch, "loss": total / len(prepared), "dev_ndcg10": score})
        log(f"epoch {epoch:2d} loss {total / len(prepared):.4f} dev nDCG@10 {score:.4f}")
        if score > best:
            best, stale = score, 0
            best_state = {k: v.clone() for k, v in model.net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.net.load_state_dict(best_state)
    return model, {"best_dev_ndcg10": best, "history": history, "train_queries": len(useful)}
