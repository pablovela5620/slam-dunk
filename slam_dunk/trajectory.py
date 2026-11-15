"""Dense trajectory contract; timestamps are parsed as integers before comparison."""

from pathlib import Path
import numpy as np
from .runtime import sha256


def validate_trajectory(frame_csv, trajectory):
    frames = np.loadtxt(frame_csv, delimiter=",", usecols=0, dtype=np.int64, ndmin=1)
    rows = [
        line.split()
        for line in Path(trajectory).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows or any(len(row) != 8 for row in rows):
        raise ValueError("Trajectory must contain timestamp-ns, position XYZ, quaternion XYZW")
    timestamps = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    values = np.asarray([row[1:] for row in rows], dtype=np.float64)
    norm_error = float(np.max(abs(np.linalg.norm(values[:, 3:], axis=1) - 1)))
    result = dict(
        expected_frames=len(frames),
        output_frames=len(rows),
        exact_grid=bool(np.array_equal(frames, timestamps)),
        finite=bool(np.isfinite(values).all()),
        quaternion_norm_error=norm_error,
        sha256=sha256(trajectory),
    )
    if not result["exact_grid"] or not result["finite"] or norm_error >= 1e-5:
        raise ValueError(f"Invalid dense trajectory: {result}")
    return result
