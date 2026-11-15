"""Auxiliary short optical-flow tracks in the descriptor detector's coordinates."""

import os
import numpy as np


def auxiliary_tracks(path, timestamps, offset, pixel_bias=0.25):
    try:
        raw = np.fromfile(path, np.float64)
    except OSError:
        return np.empty((0, 6)), np.empty(0, dtype=np.int64)
    tracks = raw[: raw.size // 6 * 6].reshape(-1, 6)
    tracks = tracks[np.isfinite(tracks).all(axis=1)]
    indices = np.searchsorted(timestamps, tracks[:, 0])
    keep = indices < len(timestamps)
    keep[keep] &= timestamps[indices[keep]] == tracks[keep, 0].astype(np.int64)
    tracks = tracks[keep]
    if not len(tracks):
        return tracks, np.empty(0, dtype=np.int64)
    # Limit accumulated frame-to-frame localization drift. Every segment stays
    # independent of the descriptor landmarks and has one observation per image.
    duration = float(os.environ.get("FLOW_SEGMENT", "10"))
    labels = np.c_[tracks[:, 2], np.floor((tracks[:, 0] - timestamps[0]) / (duration * 1e9))]
    _, identifiers, counts = np.unique(labels, axis=0, return_inverse=True, return_counts=True)
    tracks[:, 2] = identifiers + offset
    keep = counts[identifiers] >= int(os.environ.get("FLOW_MIN_OBS", "3"))
    tracks = tracks[keep]
    # OpenCV's standard SIFT pyramid carries a quarter-pixel localization
    # offset. Express both independent feature families in that convention.
    tracks[:, 3:5] += pixel_bias
    return tracks, np.searchsorted(timestamps, tracks[:, 0])
