"""Live frames: preprocessing (Python reference and C++ extension), Triton
classification parsing, visual context, and /ws/frames end to end."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core.cache import InMemoryCache
from backend.main import create_app
from backend.vision import context
from backend.vision.preprocess import MEAN, STD, Preprocessor, reference_encoded, reference_frame
from model_server.vision_client import (
    Label,
    MockVisionClient,
    VisionUnavailable,
    parse_classification,
)
from tests.conftest import make_settings

cv2 = pytest.importorskip("cv2")

# One uint8 step after normalization, in the channel with the smallest std.
LSB = 1 / (255 * min(STD))


def _frame(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (5, 5), 0)  # camera-like, not pure noise


def _jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def test_reference_normalizes_constant_frame_exactly():
    bgr = np.zeros((300, 400, 3), np.uint8)
    bgr[..., 0], bgr[..., 1], bgr[..., 2] = 10, 128, 250  # B, G, R
    out = reference_frame(bgr)
    assert out.shape == (1, 3, 224, 224) and out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    for c, value in enumerate((250, 128, 10)):  # output is RGB
        np.testing.assert_allclose(out[0, c], (value / 255 - MEAN[c]) / STD[c], atol=1e-6)


def test_reference_crops_the_center_after_resizing_the_short_side():
    # 256x256 needs no resize: the output is exactly the central 224x224 window.
    bgr = _frame(256, 256)
    out = reference_frame(bgr)
    window = bgr[16:240, 16:240, ::-1].astype(np.float32) / 255
    expected = ((window - np.float32(MEAN)) / np.float32(STD)).transpose(2, 0, 1)
    np.testing.assert_allclose(out[0], expected, atol=1e-6)
    # Landscape 480x640 -> 256x341 -> crop columns 58..282.
    assert reference_frame(_frame(480, 640)).shape == (1, 3, 224, 224)
    assert reference_frame(_frame(50, 90)).shape == (1, 3, 224, 224)  # upscaled


def test_reference_rejects_bad_input():
    with pytest.raises(ValueError):
        reference_frame(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        reference_encoded(b"not an image")


def test_python_fallback_preprocessor_matches_reference():
    pre = Preprocessor(prefer_native=False)
    assert pre.name == "python-opencv" and not pre.native
    data = _jpeg(_frame(480, 640))
    np.testing.assert_array_equal(pre.encoded(data), reference_encoded(data))
    batch = pre.batch([data, data])
    assert batch.shape == (2, 3, 224, 224)


# --- C++ extension (skipped unless native/vision is built) -------------------

SHAPES = [(256, 256), (480, 640), (720, 1280), (1080, 1920), (640, 480), (225, 300), (50, 90)]


@pytest.fixture(scope="module")
def native():
    return pytest.importorskip("omni_vision", reason="build native/vision to test the extension")


def test_native_identity_resize_matches_reference_exactly(native):
    # No resize: decode-free path isolates crop + normalize + layout.
    bgr = _frame(256, 256, seed=3)
    np.testing.assert_allclose(native.preprocess_frame(bgr), reference_frame(bgr), atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
def test_native_frame_matches_reference(native, shape):
    bgr = _frame(*shape, seed=sum(shape))
    got, want = native.preprocess_frame(bgr), reference_frame(bgr)
    assert got.shape == want.shape == (1, 3, 224, 224) and got.dtype == np.float32
    diff = np.abs(got - want)
    # The extension links the system OpenCV, the reference uses the pip wheel;
    # bilinear resize may round a pixel differently by one uint8 step.
    assert diff.max() <= LSB + 1e-5, diff.max()
    assert (diff > 1e-5).mean() < 0.02


def test_native_encoded_and_batch_match_reference(native):
    frames = [_frame(480, 640, seed=i) for i in range(3)]
    blobs = [_jpeg(f) for f in frames]
    single = native.preprocess_encoded(blobs[0])
    batch = native.preprocess_batch(blobs)
    assert batch.shape == (3, 3, 224, 224) and batch.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(batch[0], single[0])
    for i, blob in enumerate(blobs):
        # JPEG decoders may also differ by a step between libjpeg builds.
        assert np.abs(batch[i] - reference_encoded(blob)[0]).max() <= 2 * LSB + 1e-5


def test_native_output_owns_its_buffer(native):
    out = native.preprocess_frame(_frame(256, 256))
    assert out.base is not None  # held by the capsule, no copy made by NumPy
    del out  # freeing must not crash


def test_native_rejects_bad_input(native):
    with pytest.raises(ValueError):
        native.preprocess_encoded(b"garbage")
    with pytest.raises((ValueError, RuntimeError)):
        native.preprocess_frame(np.zeros((10, 10, 4), np.uint8))
    with pytest.raises(ValueError):
        native.preprocess_frame(_frame(256, 256), crop=300)


# --- classification and visual context --------------------------------------


def test_parse_triton_classification():
    raw = np.array([b"9.5:281:tabby cat", b"7.25:285:Egyptian cat", b"1.0:7"], dtype=object)
    assert parse_classification(raw) == [
        Label("tabby cat", 9.5),
        Label("Egyptian cat", 7.25),
        Label("class_7", 1.0),
    ]
    with pytest.raises(ValueError):
        parse_classification(np.array([b"nonsense"], dtype=object))


def test_mock_classifier_is_deterministic():
    pixels = reference_frame(_frame(256, 256))
    a = asyncio.run(MockVisionClient(top_k=3).classify(pixels))
    b = asyncio.run(MockVisionClient(top_k=3).classify(pixels))
    assert a == b and len(a) == 3


def test_visual_context_expires():
    cache = InMemoryCache()

    async def run():
        await context.record(cache, "t-1", 4, [Label("monitor", 0.9), Label("screen", 0.05)], 120)
        fresh = await context.describe(cache, "t-1", 120)
        stale = await context.describe(cache, "t-1", 120, now=10**12)
        missing = await context.describe(cache, "nope", 120)
        return fresh, stale, missing

    fresh, stale, missing = asyncio.run(run())
    assert "frame #4" in fresh and "monitor (0.90)" in fresh
    assert stale == "" and missing == ""


# --- /ws/frames end to end ----------------------------------------------------


def test_ws_frames_classifies_and_feeds_the_agent():
    with TestClient(create_app(make_settings(vision_backend="mock"))) as c:
        with c.websocket_connect("/ws/frames?thread_id=vision-thread-1") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "session.created" and hello["classifier"] == "mock"
            ws.send_bytes(_jpeg(_frame(480, 640)))
            result = ws.receive_json()
            assert result["type"] == "frame.result" and result["seq"] == 1
            assert len(result["labels"]) == 5
            assert set(result["timings"]) == {"preprocess_ms", "infer_ms", "server_ms"}
            ws.send_bytes(b"not a jpeg")
            assert ws.receive_json() == {
                "type": "frame.error",
                "seq": 2,
                "error": "could not decode image bytes",
            }
        steps = []
        with c.websocket_connect("/ws/agent?thread_id=vision-thread-1") as ws:
            ws.receive_json()
            ws.send_json({"type": "query", "id": "v1", "text": "what's 6*7?"})
            while (ev := ws.receive_json())["type"] != "response.end":
                if ev["type"] == "agent.step":
                    steps.append(ev)
        visual = [s for s in steps if s["node"] == "visual_context"]
        assert visual and "frame #1" in visual[0]["observation"]
        assert 'omniagent_frames_total{outcome="classified"}' in c.get("/metrics").text


def test_ws_frames_degrades_when_triton_is_down(monkeypatch):
    with TestClient(create_app(make_settings(vision_backend="mock"))) as c:

        async def down(_pixels):
            raise VisionUnavailable("vit_b16: connection refused")

        monkeypatch.setattr(c.app.state.services.vision, "classify", down)
        with c.websocket_connect("/ws/frames") as ws:
            ws.receive_json()
            ws.send_bytes(_jpeg(_frame(256, 256)))
            reply = ws.receive_json()
            assert reply["type"] == "frame.unavailable" and "refused" in reply["reason"]


def test_ws_frames_closes_when_vision_is_off():
    from starlette.websockets import WebSocketDisconnect

    with TestClient(create_app(make_settings())) as c:
        with pytest.raises(WebSocketDisconnect) as exc:
            with c.websocket_connect("/ws/frames") as ws:
                ws.receive_json()
        assert exc.value.code == 1008
