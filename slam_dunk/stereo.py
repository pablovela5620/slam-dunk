"""Import benchmark frontend measurements into the estimator's stereo frame.

Camera 0 is the fixed reference and camera 1 carries the relative stereo path.
The dataset frontend runs in the benchmark's native stream order; this single
import boundary permutes every camera-indexed measurement without recomputing
features, triangulation, timestamps, or the VIO seed.
"""

import json
from pathlib import Path
import shutil

import numpy as np

SCHEMA = "stereo-reference-v1"


def require_stereo_packet(packet):
    marker = Path(packet) / "input.bin.stereo"
    if not marker.is_file() or marker.read_text().strip() != SCHEMA:
        raise ValueError("Expected a stereo-reference-v1 packet; rerun prepare with this release")


def _swap_packet(path):
    """Permute camera blocks and IDs in place, keeping observation order intact."""
    header = np.fromfile(path, dtype=np.int64, count=4)
    if len(header) != 4 or np.any(header < 0):
        raise ValueError("Invalid frontend packet header")
    n, p, o, i = map(int, header)
    expected = 32 + 8 * (32 + 14 + 11 * n + 3 * p + 5 * o + 7 * i)
    if path.stat().st_size != expected:
        raise ValueError("Invalid frontend packet size")
    data = np.memmap(path, dtype=np.float64, mode="r+", offset=32)
    data[:32] = data[:32].reshape(2, 16)[::-1].copy().ravel()
    data[32:46] = data[32:46].reshape(2, 7)[::-1].copy().ravel()
    start = 46 + 11 * n + 3 * p
    observations = data[start:start + 5 * o].reshape(o, 5)
    if not np.isin(observations[:, 1], [0, 1]).all():
        raise ValueError("Invalid observation camera ID")
    observations[:, 1] = 1 - observations[:, 1]
    data.flush()
    del observations, data

    meta_path = Path(str(path) + ".meta")
    meta = np.fromfile(meta_path, dtype=np.float64)
    if len(meta) != 2 + 7 * n:
        raise ValueError("Invalid camera metadata size")
    meta[:2] = meta[:2][::-1].copy()
    rows = meta[2:].reshape(n, 7)
    rows[:] = rows[:, [1, 0, 2, 3, 4, 6, 5]]
    meta.tofile(meta_path)

    candidate_path = Path(str(path) + ".candidates")
    if candidate_path.exists():
        m, count = np.fromfile(candidate_path, dtype=np.int64, count=2)
        if m < 0 or count < 0 or candidate_path.stat().st_size != 16 + 8 * m + 64 * count:
            raise ValueError("Invalid candidate packet size")
        if count:
            candidates = np.memmap(candidate_path, dtype=np.float64, mode="r+",
                                   offset=int(16 + 8 * m), shape=(int(count), 8))
            if not np.isin(candidates[:, 2], [0, 1]).all():
                raise ValueError("Invalid candidate camera ID")
            candidates[:, 2] = 1 - candidates[:, 2]
            candidates.flush()
            del candidates

    clock_path = Path(str(path) + ".camtime")
    if clock_path.exists():
        offsets = np.fromfile(clock_path, dtype=np.float64)
        if len(offsets) != 2 * n:
            raise ValueError("Invalid camera clock metadata size")
        offsets.reshape(n, 2)[:, ::-1].copy().tofile(clock_path)


def export_packet(recording, frontend, output):
    """Export a new canonical packet from the dataset frontend's unfitted graph."""
    recording, frontend, output = map(Path, (recording, frontend, output))
    output.mkdir(exist_ok=False)
    for source in sorted(frontend.glob("input.bin*")):
        shutil.copyfile(source, output / source.name)
    for name in ("imu.bin", "vio.txt", "export.npz"):
        shutil.copyfile(frontend / name, output / name)
    _swap_packet(output / "input.bin")
    cameras = json.loads((frontend / "calib.json").read_text())
    (output / "calib.json").write_text(json.dumps(cameras[::-1]) + "\n")
    calibration = json.loads((recording / "calibration.json").read_text())
    calibration["cam0"], calibration["cam1"] = calibration["cam1"], calibration["cam0"]
    (output / "calibration.json").write_text(json.dumps(calibration) + "\n")
    # Output times follow the benchmark grid, independently of camera numbering.
    shutil.copyfile(recording / "cam0/data.csv", output / "frame_times.csv")
    # Publish the schema last: incomplete exports cannot be refined accidentally.
    (output / "input.bin.stereo").write_text(SCHEMA + "\n")
    return output
