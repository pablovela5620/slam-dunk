"""Read native checkpoints, including a partially written last line at a deadline."""

from pathlib import Path
import os
import numpy as np


def load_poses(path):
    try:
        a = np.loadtxt(path, ndmin=2)
    except ValueError:
        rows = []
        with Path(path).open() as f:
            for line in f:
                fields = line.split()
                if len(fields) != 8:
                    continue
                try:
                    rows.append([float(v) for v in fields])
                except ValueError:
                    continue
        a = np.asarray(rows, dtype=float).reshape(-1, 8)
    if a.ndim != 2 or a.shape[1] != 8:
        raise ValueError("no complete pose checkpoint")
    good = np.isfinite(a).all(axis=1) & (np.linalg.norm(a[:, 4:], axis=1) > 1e-6)
    a = a[good]
    if not len(a):
        raise ValueError("no finite pose checkpoint")
    if np.any(np.diff(a[:, 0]) < 0):
        a = a[np.argsort(a[:, 0], kind="stable")]
    return a[np.r_[True, np.diff(a[:, 0]) > 0]]


def nearest_indices(stamps, query):
    i = np.clip(np.searchsorted(stamps, query), 0, len(stamps) - 1)
    prev = np.maximum(0, i - 1)
    return np.where(np.abs(stamps[prev] - query) <= np.abs(stamps[i] - query), prev, i)


def save_poses(path, timestamps, positions, quaternions):
    """Publish complete poses without converting integer timestamps to floats."""
    path = Path(path)
    poses = np.column_stack((positions, quaternions))
    if len(timestamps) != len(poses) or poses.shape[1] != 7 or not np.isfinite(poses).all():
        raise ValueError("invalid dense trajectory")
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w") as output:
        for stamp, pose in zip(timestamps, poses):
            output.write(str(int(stamp)) + " " + " ".join(f"{value:.10f}" for value in pose) + "\n")
    os.replace(temporary, path)
