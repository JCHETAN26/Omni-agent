"""Image preprocessing for the ViT model: the C++ `omni_vision` extension when
it is built, otherwise the same pipeline in Python OpenCV.

Pipeline (torchvision ViT-B/16 IMAGENET1K_V1 evaluation transform):
decode -> resize short side to 256 (bilinear) -> center-crop 224 ->
BGR to RGB -> [0, 1] -> normalize with ImageNet mean/std -> float32 NCHW.

`reference_*` functions are the Python implementation: the fallback path and
the reference the C++ extension is tested against.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
RESIZE_SHORT = 256
CROP = 224


def _cv2():
    import cv2

    return cv2


def reference_frame(
    bgr: np.ndarray, resize_short: int = RESIZE_SHORT, crop: int = CROP
) -> np.ndarray:
    """HxWx3 uint8 BGR frame -> (1, 3, crop, crop) float32."""
    cv2 = _cv2()
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("frame must be an HxWx3 uint8 BGR array")
    h, w = bgr.shape[:2]
    scale = resize_short / min(h, w)
    nh, nw = max(crop, round(h * scale)), max(crop, round(w * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (nh - crop) // 2, (nw - crop) // 2
    rgb = resized[y0 : y0 + crop, x0 : x0 + crop, ::-1].astype(np.float32) / 255.0
    out = (rgb - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return np.ascontiguousarray(out.transpose(2, 0, 1)[None])


def decode(data: bytes) -> np.ndarray:
    cv2 = _cv2()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image bytes")
    return img


def reference_encoded(data: bytes) -> np.ndarray:
    return reference_frame(decode(data))


class Preprocessor:
    """Picks the C++ extension when importable; `name` says which one runs."""

    def __init__(self, prefer_native: bool = True):
        self._native = None
        if prefer_native:
            try:
                import omni_vision

                self._native = omni_vision
            except ImportError:
                self._native = None
        self.name = f"cpp-opencv-{self._native.opencv_version}" if self._native else "python-opencv"

    @property
    def native(self) -> bool:
        return self._native is not None

    def encoded(self, data: bytes) -> np.ndarray:
        if self._native:
            try:
                return self._native.preprocess_encoded(data)
            except ValueError:
                raise
            except RuntimeError as exc:  # pybind11 maps std::invalid_argument to ValueError
                raise ValueError(str(exc)) from exc
        return reference_encoded(data)

    def batch(self, items: Sequence[bytes]) -> np.ndarray:
        if self._native:
            return self._native.preprocess_batch(list(items))
        return np.concatenate([reference_encoded(d) for d in items], axis=0)

    def frame(self, bgr: np.ndarray) -> np.ndarray:
        if self._native:
            return self._native.preprocess_frame(np.ascontiguousarray(bgr))
        return reference_frame(bgr)
