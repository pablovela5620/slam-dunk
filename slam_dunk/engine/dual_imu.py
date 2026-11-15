"""Factory calibration and recording-local calibration of a second rigid IMU.

The virtual sensor stays at imu-right's origin and in its coordinate frame.
No reference trajectory or outside information is used.
"""

import json, os
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation


def calibrated_imu(src, label, filename):
    src = Path(src)
    factory = json.loads((src / "factory_calibration.json").read_text())
    calibration = next(x for x in factory["ImuCalibrations"] if x["Label"] == label)
    raw = np.loadtxt(src / filename, delimiter=",", ndmin=2)[:, :7]
    raw = raw[np.isfinite(raw).all(axis=1)]
    if np.any(np.diff(raw[:, 0]) < 0):
        raw = raw[np.argsort(raw[:, 0], kind="stable")]
    raw = raw[np.r_[True, np.diff(raw[:, 0]) > 0]]
    if len(raw) < 2:
        raise ValueError("second IMU has fewer than two samples")
    result = []
    for key, values in [("Gyroscope", raw[:, 1:4]), ("Accelerometer", raw[:, 4:7])]:
        c = calibration[key]
        values = (values - np.array(c["Bias"]["Offset"])) @ np.linalg.inv(
            c["Model"]["RectificationMatrix"]
        ).T
        offset = c.get("TimeOffsetSec_Device_" + ("Gyro" if key == "Gyroscope" else "Accel"), 0)
        result.append((raw[:, 0] * 1e-9 - offset, values))
    return result, calibration


def interpolate(t, series):
    return np.column_stack([np.interp(t, series[0], series[1][:, j]) for j in range(3)])


def fit_affine(x, y, noise, ridge=0.0):
    """Robust regression, with identity as a weak intrinsic calibration prior."""
    design = np.c_[x, np.ones(len(x))]
    prior = np.eye(4)
    prior[3, 3] = 0
    target = np.r_[np.eye(3), np.zeros((1, 3))]
    weights = np.ones(len(x))
    for _ in range(4):
        fit = np.linalg.solve(
            design.T @ (weights[:, None] * design) + ridge * prior,
            design.T @ (weights[:, None] * y) + ridge * target,
        )
        residual = np.linalg.norm(design @ fit - y, axis=1)
        weights = np.minimum(1.0, noise / np.maximum(residual, 1e-12))
    return fit


def virtual_imu(src, primary, gyro_weight=0.0, accel_weight=0.5, robust=False):
    """Fuse calibrated independent measurements after rigid-body correction.

    `primary` uses nanosecond timestamps and has factory corrected gyro/accel.
    A robust affine and time calibration aligns imu-left to imu-right using
    motion within this recording. The fitted bias is constant, so independent
    time-varying bias information is retained by averaging the measurements.
    """
    secondary, c1 = calibrated_imu(src, "imu-left", "imu1/data.csv")
    fac = json.loads((Path(src) / "factory_calibration.json").read_text())
    c0 = next(x for x in fac["ImuCalibrations"] if x["Label"] == "imu-right")

    def pose(c):
        p = c["T_Device_Imu"]
        q = p["UnitQuaternion"]
        return Rotation.from_quat([*q[1], q[0]]), np.array(p["Translation"])

    r0, p0 = pose(c0)
    r1, p1 = pose(c1)
    relative = r0.inv() * r1
    lever = r0.inv().apply(p1 - p0)
    secondary = [(t, relative.apply(v)) for t, v in secondary]
    t = primary[:, 0] * 1e-9
    # Fit at 100 Hz with modest smoothing; retain full-rate data for fusion.
    stride = max(1, int(round(0.01 / np.median(np.diff(t)))))
    indices = np.arange(10, len(t) - 10, stride)
    if len(indices) > 80000:
        indices = indices[:: int(np.ceil(len(indices) / 80000))]
    smoothed0 = gaussian_filter1d(primary[:, 1:4], 2.0, axis=0)
    smoothed1 = (secondary[0][0], gaussian_filter1d(secondary[0][1], 1.6, axis=0))
    if len(indices) < 50:
        raise ValueError("insufficient motion samples for relative calibration")
    target = smoothed0[indices]

    def gyro_fit(delay, full=False):
        x = interpolate(t[indices] + delay, smoothed1)
        fit = fit_affine(x, target, 0.02, ridge=0.01)
        residual = np.linalg.norm(np.c_[x, np.ones(len(x))] @ fit - target, axis=1)
        return fit if full else np.mean(np.minimum(residual, 0.05) ** 2)

    optimum = minimize_scalar(
        gyro_fit, bounds=(-0.008, 0.008), method="bounded", options={"xatol": 1e-6}
    )
    delay = float(optimum.x)
    gf = gyro_fit(delay, True)
    g1 = interpolate(t + delay, secondary[0]) @ gf[:3] + gf[3]
    # Translation of the rigid sensors introduces tangential and centripetal
    # accelerations. Remove them before combining specific-force measurements.
    omega = gaussian_filter1d(0.5 * (primary[:, 1:4] + g1), 3.0, axis=0)
    alpha = np.gradient(omega, t, axis=0)
    correction = np.cross(alpha, lever) + np.cross(omega, np.cross(omega, lever))
    atarget = primary[indices, 4:7]
    accel1 = interpolate(t, secondary[1]) - correction
    af = fit_affine(accel1[indices], atarget, 0.2, ridge=0.1)
    if np.linalg.cond(gf[:3]) > 1.3 or np.linalg.cond(af[:3]) > 1.3:
        raise ValueError("ill-conditioned relative IMU calibration")
    a1 = accel1 @ af[:3] + af[3]
    # Measure disagreements and retain the primary outside the sensor overlap.
    eg = np.linalg.norm(g1 - primary[:, 1:4], axis=1)
    ea = np.linalg.norm(a1 - primary[:, 4:7], axis=1)
    wg = gyro_weight * (np.minimum(1.0, 0.15 / np.maximum(eg, 1e-9)) if robust else np.ones(len(t)))
    wa = accel_weight * (np.minimum(1.0, 1.0 / np.maximum(ea, 1e-9)) if robust else np.ones(len(t)))
    inside = (t > secondary[0][0][0] + 0.02) & (t < secondary[0][0][-1] - 0.02)
    wg *= inside
    wa *= inside
    fused = primary.copy()
    fused[:, 1:4] += wg[:, None] * (g1 - primary[:, 1:4])
    fused[:, 4:7] += wa[:, None] * (a1 - primary[:, 4:7])
    diagnostics = dict(
        delay=delay,
        gyro_fit=gf.tolist(),
        accel_fit=af.tolist(),
        lever=lever.tolist(),
        gyro_disagreement=float(np.median(eg)),
        accel_disagreement=float(np.median(ea)),
    )
    return fused, diagnostics, np.c_[primary[:, 0], g1, a1]


if __name__ == "__main__":
    import sys

    src, work = map(Path, sys.argv[1:3])
    primary = np.fromfile(work / "imu.bin", dtype=np.float64).reshape(-1, 7)
    fused, diag, secondary = virtual_imu(
        src,
        primary,
        float(os.environ.get("ARIA_DUAL_G", "0")),
        float(os.environ.get("ARIA_DUAL_A", ".5")),
    )
    # Only the fused measurements enter estimation. The aligned secondary
    # series was an unused diagnostic sidecar (tens of MB per recording).
    fused.tofile(work / "imu_fused.bin")
    (work / "dual_imu.json").write_text(json.dumps(diag, indent=2))
    print(json.dumps(diag), flush=True)
