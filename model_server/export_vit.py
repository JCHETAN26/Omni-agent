"""Export torchvision ViT-B/16 (IMAGENET1K_V1) to ONNX for Triton.

    python -m model_server.export_vit       # writes vit_b16{,_nobatch}/1/model.onnx + labels.txt

The graph takes `pixel_values` (N, 3, 224, 224) with a dynamic batch axis and
returns `logits` (N, 1000). Preprocessing happens before the model (the C++
omni_vision extension), matching the weights' own evaluation transform:
resize 256, center-crop 224, ImageNet mean/std. The script checks the ONNX
graph against PyTorch with onnxruntime at batch sizes 1 and 8.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent / "model_repository"
MODELS = ("vit_b16", "vit_b16_nobatch")  # same graph: dynamic batching on / off


def export(out: Path, opset: int = 17) -> dict:
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (example,),
        str(out),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    worst = 0.0
    for n in (1, 8):
        x = torch.randn(n, 3, 224, 224)
        with torch.no_grad():
            want = model(x).numpy()
        got = session.run(["logits"], {"pixel_values": x.numpy()})[0]
        assert got.shape == (n, 1000)
        worst = max(worst, float(np.abs(got - want).max()))
        assert (got.argmax(1) == want.argmax(1)).all(), "top-1 differs from PyTorch"
    if worst > 1e-3:
        raise AssertionError(f"ONNX logits differ from PyTorch by {worst:.2e}")
    return {"labels": list(weights.meta["categories"]), "max_abs_diff": worst}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", type=Path, default=REPO)
    args = p.parse_args()
    first = args.repo / MODELS[0] / "1" / "model.onnx"
    info = export(first)
    for name in MODELS:
        model_dir = args.repo / name
        (model_dir / "labels.txt").write_text("\n".join(info["labels"]) + "\n")
        target = model_dir / "1" / "model.onnx"
        if target != first:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(first, target)
        print(f"{name} -> {target} ({target.stat().st_size / 2**20:.0f} MiB)")
    print(f"ONNX vs PyTorch max |diff| = {info['max_abs_diff']:.2e} (batch 1 and 8), top-1 equal")


if __name__ == "__main__":
    main()
