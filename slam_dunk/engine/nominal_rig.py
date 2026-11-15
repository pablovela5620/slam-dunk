"""Reconcile the two nominal-camera orientation estimates of the IMU.

Camera 0 defines the fixed stereo reference; camera 1 has a relative stereo path.
Use their symmetric rotational midpoint and preserve the original inertial
position estimate exactly. Only this recording's calibration and states enter.
"""

import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def fit(src, work, frames, p, q):
    src, work = Path(src), Path(work)
    calibration = json.loads((src / "calibration.json").read_text())
    nominal = calibration["cam1"]["T_b_s"]
    reference = Rotation.from_quat(calibration["cam0"]["T_b_s"]["qvec"])
    diag = np.atleast_2d(np.loadtxt(work / "ba.txt.diag"))
    with (work / "ba.txt").open() as poses:
        first = int(poses.readline().split()[0])
    chunk_dt = 5.0
    if (work / "native-settings.json").exists():
        config = json.loads((work / "native-settings.json").read_text())
        chunk_dt = float(config.get("CHUNK_DT", chunk_dt))
    if not (np.isfinite(diag).all() and diag.shape[1] >= 14 and chunk_dt > 0):
        raise ValueError("invalid optimized rig")
    ci = np.clip(np.floor((frames - first) * 1e-9 / chunk_dt).astype(int), 0, len(diag) - 1)
    opt = reference * Rotation.from_quat(diag[ci][:, [8, 9, 10, 7]])
    nom = Rotation.from_quat(nominal["qvec"])
    delta = opt * nom.inv()
    half = Rotation.from_rotvec(0.5 * delta.as_rotvec())
    body = Rotation.from_quat(q)
    newp = p
    newq = (body * half).as_quat()
    if not (np.isfinite(newp).all() and np.isfinite(newq).all()):
        raise ValueError("non-finite nominal rig fit")
    report = dict(
        method="symmetric_camera_rotation_midpoint_original_IMU_position",
        chunks=len(diag),
        chunk_seconds=chunk_dt,
        max_rotation_change_deg=float(np.rad2deg(half.magnitude()).max()),
        max_position_change_m=float(np.linalg.norm(newp - p, axis=1).max()),
    )
    return newp, newq, report
