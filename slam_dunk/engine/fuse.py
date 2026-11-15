from trajectory_io import load_poses, nearest_indices, save_poses
import sys, os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import CubicSpline

src, work, out = map(Path, sys.argv[1:4])
base = load_poses(work / "vio.txt")
ba = load_poses(work / "ba.txt")
frames = np.loadtxt(src / "frame_times.csv", delimiter=",", usecols=0, dtype=np.int64)
t0 = frames[0]
t = (frames - t0) * 1e-9
vt = (base[:, 0] - t0) * 1e-9
bt = (ba[:, 0] - t0) * 1e-9
from gyro_interp import integrate

imu = np.fromfile(work / "imu.bin", dtype=np.float64).reshape(-1, 7)
rawgyro = imu[:, 1:4].copy()
correction = work / "ba.txt.gyromat"
if correction.exists():
    matrix = np.loadtxt(correction).reshape(3, 3)
    imu[:, 1:4] = imu[:, 1:4] @ (np.eye(3) + matrix).T
if (work / "ba.txt.gyrot").exists():
    gt = np.loadtxt(work / "ba.txt.gyrot")
    if np.any(gt[:3]):
        meta = np.fromfile(work / "input.bin.meta", dtype=np.float64)[2:].reshape(-1, 7)
        temperature = np.interp(imu[:, 0], ba[:, 0], 0.5 * (meta[:, 5] + meta[:, 6]))
        imu[:, 1:4] += rawgyro * gt[:3] * (temperature - gt[3])[:, None]
shift = 0.0
if (work / "ba.txt.timing").exists():
    shift = float(np.loadtxt(work / "ba.txt.timing")[0])
gq = integrate((imu[:, 0] - t0) * 1e-9, imu[:, 1:4], t + shift)
rg = R.from_quat(gq)
idx = nearest_indices(frames, ba[:, 0].astype(np.int64))
corr = R.from_quat(ba[:, 4:]) * rg[idx].inv()
q = (Slerp(bt, corr)(np.clip(t, bt[0], bt[-1])) * rg).as_quat()

# Integrate calibrated acceleration at IMU rate and bridge optimized keyframes.
# Retain the initializer-based reconstruction if optional refinement files fail.
try:
    from imu_bridge import positions

    p = positions(work, frames, ba, imu, corr, shift)
    if not np.isfinite(p).all():
        raise ValueError("non-finite inertial bridge")
except Exception as error:
    print("Dense IMU reconstruction fallback:", str(error), flush=True)
    vpos = np.column_stack([np.interp(t, vt, base[:, j]) for j in range(1, 4)])
    atpos = np.column_stack([np.interp(bt, vt, base[:, j]) for j in range(1, 4)])
    offset = ba[:, 1:4] - atpos
    p = vpos + np.column_stack([np.interp(t, bt, offset[:, j]) for j in range(3)])
    # Preserve head-motion curvature between the VIO poses using the accelerometer.
    if os.environ.get("ACC_INTERP", "1") == "1" and len(base) > 1:
        time_path = work / "ba.txt.timing"
        ashift = shift
        if time_path.exists():
            ashift += float(np.loadtxt(time_path)[1])
        ar = np.zeros(3)
        sc = np.zeros(3)
        if (work / "ba.txt.accalign").exists():
            ar = np.loadtxt(work / "ba.txt.accalign")
        if (work / "ba.txt.accscale").exists():
            sc = np.loadtxt(work / "ba.txt.accscale")
        ax, ay, az = ar
        A = np.eye(3) + np.diag(sc[:3]) + np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
        if len(sc) > 3:
            A += np.array([[0.0, sc[3], sc[4]], [sc[3], 0.0, sc[5]], [sc[4], sc[5], 0.0]])
        it = (imu[:, 0] - t0) * 1e-9
        acc = np.column_stack([np.interp(t + ashift, it, imu[:, 4 + j]) for j in range(3)]) @ A.T
        if (work / "ba.txt.diag").exists():
            diag = np.atleast_2d(np.loadtxt(work / "ba.txt.diag"))
            chunk = float(os.environ.get("CHUNK_DT", "5"))
            k = np.clip(np.floor((t - bt[0]) / chunk).astype(int), 0, len(diag) - 1)
            acc -= diag[k, 4:7]
        grav = np.array([0.0, 0.0, -9.80665])
        if (work / "ba.txt.gravity").exists():
            grav = np.loadtxt(work / "ba.txt.gravity")
        acc = R.from_quat(q).apply(acc) + grav
        dt = np.diff(t)
        vel = np.zeros_like(acc)
        vel[1:] = np.cumsum(0.5 * (acc[:-1] + acc[1:]) * dt[:, None], axis=0)
        cp = np.zeros_like(acc)
        cp[1:] = np.cumsum(
            vel[:-1] * dt[:, None] + (2 * acc[:-1] + acc[1:]) * dt[:, None] ** 2 / 6, axis=0
        )
        vi = nearest_indices(frames, base[:, 0].astype(np.int64))
        linear = np.column_stack([np.interp(t, vt, cp[vi, j]) for j in range(3)])
        curve = cp - linear
        curve[(t < vt[0]) | (t > vt[-1])] = 0.0
        p += curve
    # Cover any short camera tail beyond the final IMU-supported state.
    if len(base) > 1 and np.any(t > vt[-1]):
        endpoint = np.column_stack([np.interp(vt[-2:], t, p[:, j]) for j in range(3)])
        velocity = (endpoint[1] - endpoint[0]) / max(vt[-1] - vt[-2], 1e-6)
        speed = np.linalg.norm(velocity)
        if speed > 4:
            velocity *= 4 / speed
        after = t > vt[-1]
        p[after] = endpoint[1] + (t[after] - vt[-1])[:, None] * velocity
if not (np.isfinite(p).all() and np.isfinite(q).all()):
    raise RuntimeError("non-finite reconstructed pose")
if os.environ.get("ARIA_NOMINAL_RIG_ROTATION", "0") == "1":
    original_p, original_q = p, q
    rig_receipt = work / "rig-export.json"
    try:
        import json
        from nominal_rig import fit

        if os.environ.get("ARIA_AUDIT_RIG", "0") == "1":
            save_poses(work / "imu_before_rig.txt", frames, p, q)
        newp, newq, rig_report = fit(src, work, frames, p, q)
        receipt_tmp = Path(str(rig_receipt) + ".tmp")
        receipt_tmp.write_text(json.dumps(rig_report, indent=2) + "\n")
        receipt_tmp.replace(rig_receipt)
        p, q = newp, newq
        print("NOMINAL_RIG_EXPORT", json.dumps(rig_report, sort_keys=True), flush=True)
    except Exception as error:
        p, q = original_p, original_q
        try:
            rig_receipt.unlink(missing_ok=True)
        except OSError:
            pass
        print("Nominal rig export retained original IMU estimate:", str(error), flush=True)
save_poses(out, frames, p, q)
