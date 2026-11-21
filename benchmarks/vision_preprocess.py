"""C++ (omni_vision) vs Python OpenCV preprocessing for the ViT model.

    python -m benchmarks.vision_preprocess --output results.json

For each frame size, times decode + resize + crop + normalize to NCHW float32
on the same JPEG bytes, single-threaded and with N worker threads (the C++
path releases the GIL; the Python path holds it for the NumPy steps). Also
reports the numerical difference between the two paths.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from backend.vision.preprocess import reference_encoded, reference_frame


def make_frames(h: int, w: int, n: int, seed: int = 7) -> list[bytes]:
    import cv2

    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        img = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (9, 9), 0)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert ok
        frames.append(buf.tobytes())
    return frames


def per_frame_ms(fn, frames: list[bytes], repeat: int) -> list[float]:
    times = []
    for _ in range(repeat):
        for f in frames:
            t = time.perf_counter()
            fn(f)
            times.append((time.perf_counter() - t) * 1000)
    return times


def threaded_fps(fn, frames: list[bytes], threads: int, total: int) -> float:
    work = [frames[i % len(frames)] for i in range(total)]
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(fn, work[: threads * 2]))  # warm the pool
        t = time.perf_counter()
        list(pool.map(fn, work))
        return total / (time.perf_counter() - t)


def main() -> None:
    import cv2
    import omni_vision

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", nargs="+", default=["720x1280", "1080x1920"])
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output")
    args = p.parse_args()

    report = {
        "host": platform.platform(),
        "python_opencv": cv2.__version__,
        "cpp_opencv": omni_vision.opencv_version,
        "threads": args.threads,
        "sizes": [],
    }
    for size in args.sizes:
        h, w = map(int, size.split("x"))
        frames = make_frames(h, w, args.frames)
        for f in frames[:2]:  # warm both paths
            omni_vision.preprocess_encoded(f)
            reference_encoded(f)
        row = {"size": size, "jpeg_kb": round(statistics.mean(len(f) for f in frames) / 1024, 1)}
        for name, fn in (("python", reference_encoded), ("cpp", omni_vision.preprocess_encoded)):
            t = per_frame_ms(fn, frames, args.repeat)
            row[name] = {
                "ms_p50": round(statistics.median(t), 3),
                "ms_mean": round(statistics.mean(t), 3),
                "fps_1_thread": round(1000 / statistics.mean(t), 1),
                f"fps_{args.threads}_threads": round(
                    threaded_fps(fn, frames, args.threads, args.frames * args.repeat), 1
                ),
            }
        row["speedup_1_thread"] = round(row["python"]["ms_mean"] / row["cpp"]["ms_mean"], 2)
        key = f"fps_{args.threads}_threads"
        row[f"speedup_{args.threads}_threads"] = round(row["cpp"][key] / row["python"][key], 2)
        # Transform-only comparison on the decoded frame (no JPEG decoder difference).
        bgr = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
        diff = np.abs(omni_vision.preprocess_frame(bgr) - reference_frame(bgr))
        enc = np.abs(omni_vision.preprocess_encoded(frames[0]) - reference_encoded(frames[0]))
        row["max_abs_diff_frame"] = float(diff.max())
        row["frac_diff_gt_1e-5_frame"] = round(float((diff > 1e-5).mean()), 5)
        row["max_abs_diff_encoded"] = float(enc.max())
        report["sizes"].append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
