"""Recording-local lossless cache for OpenCV SIFT features and calibrated rays."""

from pathlib import Path
import os
import numpy as np


def save_features(path, xy, desc, rays, scales=None):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    # SIFT normally returns quantized float descriptors. Preserve a float
    # descriptor unchanged if a library version returns a wider representation.
    quantized = np.asarray(desc, dtype=np.uint8)
    compact = np.array_equal(desc, quantized)
    header = len(xy) | (0 if compact else 0x80000000) | (0x40000000 if scales is not None else 0)
    with tmp.open("wb") as f:
        np.array([header], dtype=np.uint32).tofile(f)
        np.ascontiguousarray(xy, dtype=np.float32).tofile(f)
        np.ascontiguousarray(
            quantized if compact else desc, dtype=np.uint8 if compact else np.float32
        ).tofile(f)
        np.ascontiguousarray(rays, dtype=np.float64).tofile(f)
        if scales is not None:
            np.ascontiguousarray(scales, dtype=np.float32).tofile(f)
    os.replace(tmp, path)


def load_features(path, with_scales=False):
    try:
        with Path(path).open("rb") as f:
            header = np.fromfile(f, np.uint32, 1)
            if not len(header):
                return None
            raw = int(header[0])
            n = raw & 0x3FFFFFFF
            if n > 10000:
                return None
            dtype = np.float32 if raw & 0x80000000 else np.uint8
            xy = np.fromfile(f, np.float32, 2 * n).reshape(n, 2).astype(np.float64)
            desc = np.fromfile(f, dtype, 128 * n).reshape(n, 128).astype(np.float32)
            rays = np.fromfile(f, np.float64, 3 * n).reshape(n, 3)
            scales = (
                np.fromfile(f, np.float32, n) if raw & 0x40000000 else np.full(n, 3.2, np.float32)
            )
            if len(scales) != n:
                return None
        return (xy, desc, rays, scales) if with_scales else (xy, desc, rays)
    except (OSError, ValueError):
        return None
