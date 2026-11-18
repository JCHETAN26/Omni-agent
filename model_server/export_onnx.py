"""Export the Triton models to ONNX and verify them against PyTorch.

    python -m model_server.export_onnx            # writes model_repository/*/1/model.onnx

Each exported graph's input/output names and shapes must match its config.pbtxt.
The script checks that, and checks numerical parity with onnxruntime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from google.protobuf import text_format
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from tritonclient.grpc import model_config_pb2

REPO = Path(__file__).resolve().parent / "model_repository"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class MeanPooledEmbedder(nn.Module):
    """Transformer + mean pooling + L2 norm == SentenceTransformer.encode(normalize=True)."""

    def __init__(self, name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return nn.functional.normalize(pooled, p=2, dim=1)


class CrossEncoder(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(name)

    def forward(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        ).logits


def load_config(model: str) -> model_config_pb2.ModelConfig:
    cfg = model_config_pb2.ModelConfig()
    text_format.Parse((REPO / model / "config.pbtxt").read_text(), cfg)
    return cfg


def export(module: nn.Module, inputs: dict[str, torch.Tensor], output: str, path: Path) -> None:
    module.eval()
    batch = torch.export.Dim("batch", max=64)
    seq = torch.export.Dim("seq", max=512)
    torch.onnx.export(
        module,
        tuple(inputs.values()),
        str(path),
        input_names=list(inputs),
        output_names=[output],
        dynamic_shapes={name: {0: batch, 1: seq} for name in inputs},
        dynamo=True,
        external_data=False,
    )


def verify(path: Path, cfg, inputs: dict[str, torch.Tensor], expected: np.ndarray) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == [i.name for i in cfg.input], "input names"
    assert [o.name for o in sess.get_outputs()] == [o.name for o in cfg.output], "output names"
    (got,) = sess.run(None, {k: v.numpy() for k, v in inputs.items()})
    assert got.shape[1:] == tuple(cfg.output[0].dims), f"output dims {got.shape}"
    return float(np.abs(got - expected).max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(EMBEDDER)
    enc = tok(
        ["disk controller timeout", "a longer sentence about thermal throttling"],
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    model = MeanPooledEmbedder(EMBEDDER)
    path = REPO / "minilm_embedder" / "1" / "model.onnx"
    export(model, inputs, "sentence_embedding", path)
    with torch.no_grad():
        expected = model(**inputs).numpy()
    err = verify(path, load_config("minilm_embedder"), inputs, expected)
    print(f"minilm_embedder        -> {path.relative_to(REPO.parent)}  max|Δ|={err:.2e}")

    if args.skip_reranker:
        return
    tok = AutoTokenizer.from_pretrained(RERANKER)
    enc = tok(
        ["disk timeout", "disk timeout"],
        ["EIO-5 means the controller timed out", "PSU"],
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids")}
    model = CrossEncoder(RERANKER)
    path = REPO / "cross_encoder_reranker" / "1" / "model.onnx"
    export(model, inputs, "logits", path)
    with torch.no_grad():
        expected = model(**inputs).numpy()
    err = verify(path, load_config("cross_encoder_reranker"), inputs, expected)
    print(f"cross_encoder_reranker -> {path.relative_to(REPO.parent)}  max|Δ|={err:.2e}")


if __name__ == "__main__":
    main()
