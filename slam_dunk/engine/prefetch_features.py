"""Extract the visual keyframes while the independent VIO initializer runs."""

import os, sys, json, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import pycolmap as pc
import signal
from feature_cache import save_features, load_features
from image_features import extract_image

cv2.setNumThreads(1)
# This is a disposable cache worker. Completed files are already atomic.
signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
src, work = map(Path, sys.argv[1:3])
started = time.monotonic()
rows = [line.split() for line in (work / "frames.txt").read_text().splitlines()]
stride = int(os.environ.get("STRIDE", max(1, int(np.ceil(len(rows) / 12000)))))
rows = rows[::stride]
step = float(os.environ["BA_STEP"])
sel = [0]
for i in range(1, len(rows)):
    if int(rows[i][0]) - int(rows[sel[-1]][0]) >= step * 1e9 - 0.01:
        sel.append(i)
if sel[-1] != len(rows) - 1:
    sel.append(len(rows) - 1)
rows = [rows[i] for i in sel]
cal = json.loads((src / "calibration.json").read_text())
cams = []
for c in range(2):
    j = cal[f"cam{c}"]
    cams.append(
        pc.Camera(
            model=j["model"],
            width=j["resolution"]["width"],
            height=j["resolution"]["height"],
            params=j["params"],
        )
    )
cache = work / "features"
cache.mkdir(exist_ok=True)
nfeat = int(os.environ.get("SIFT_FEATURES", "600"))
contrast = float(os.environ.get("SIFT_CONTRAST", ".006"))
precise = os.environ.get("PRECISE_UPSCALE", "0") == "1"
local = threading.local()


def extract(k):
    i, c = divmod(k, 2)
    cached = load_features(cache / f"{rows[i][0]}_{c}.bin")
    if cached is not None:
        return cached
    xy, desc, rays = extract_image(
        rows[i][1 + c],
        cams[c],
        local,
        nfeat,
        contrast,
        precise,
        os.environ.get("SIFT_DARK_ADAPT", "1") == "1",
    )
    save_features(cache / f"{rows[i][0]}_{c}.bin", xy, desc, rays, local.feature_scales)
    return xy, desc, rays


with ThreadPoolExecutor(int(os.environ.get("SIFT_THREADS", "4"))) as pool:
    features = list(pool.map(extract, range(2 * len(rows))))
print("Prefetched", len(rows) * 2, "images in", time.monotonic() - started, flush=True)

if os.environ.get("TEMPORAL_PREFETCH", "0") == "1":
    from temporal_cache import build_temporal_cache

    build_temporal_cache(
        features,
        np.array([int(x[0]) for x in rows], np.int64),
        work,
        int(os.environ.get("SIFT_THREADS", "4")),
    )
