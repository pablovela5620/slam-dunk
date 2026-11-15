"""Reconstruct translation from calibrated dense IMU acceleration and optimized states."""

import os
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import CubicHermiteSpline


def positions(work, frames, ba, imu, corr, shift, mode="linear"):
    from gyro_interp import integrate

    t0 = frames[0]
    t = (frames - t0) * 1e-9
    bt = (ba[:, 0] - t0) * 1e-9
    it = (imu[:, 0] - t0) * 1e-9
    dense_rot = R.from_quat(integrate(it, imu[:, 1:4], it + shift))
    dense_rot = Slerp(bt, corr)(np.clip(it, bt[0], bt[-1])) * dense_rot
    ashift = shift
    if (work / "ba.txt.timing").exists():
        ashift += np.loadtxt(work / "ba.txt.timing")[1]
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
    acc = np.column_stack([np.interp(it + ashift, it, imu[:, 4 + j]) for j in range(3)]) @ A.T
    if (work / "ba.txt.diag").exists():
        diag = np.atleast_2d(np.loadtxt(work / "ba.txt.diag"))
        chunk = float(os.environ.get("CHUNK_DT", "5"))
        k = np.clip(np.floor((it - bt[0]) / chunk).astype(int), 0, len(diag) - 1)
        acc -= diag[k, 4:7]
    grav = (
        np.loadtxt(work / "ba.txt.gravity")
        if (work / "ba.txt.gravity").exists()
        else np.array([0.0, 0.0, -9.80665])
    )
    acc = dense_rot.apply(acc) + grav
    dt = np.diff(it)
    vel = np.zeros_like(acc)
    vel[1:] = np.cumsum(0.5 * (acc[:-1] + acc[1:]) * dt[:, None], axis=0)
    cp = np.zeros_like(acc)
    cp[1:] = np.cumsum(
        vel[:-1] * dt[:, None] + (2 * acc[:-1] + acc[1:]) * dt[:, None] ** 2 / 6, axis=0
    )
    cp_b = np.column_stack([np.interp(bt, it, cp[:, j]) for j in range(3)])
    cp_f = np.column_stack([np.interp(t, it, cp[:, j]) for j in range(3)])
    offset = ba[:, 1:4] - cp_b
    if mode == "hermite" and (work / "ba.txt.vel").exists():
        v = np.atleast_2d(np.loadtxt(work / "ba.txt.vel"))
        vi = np.column_stack([np.interp(bt, it, vel[:, j]) for j in range(3)])
        p = cp_f + CubicHermiteSpline(bt, offset, v[:, 1:4] - vi)(np.clip(t, bt[0], bt[-1]))
    else:
        p = cp_f + np.column_stack([np.interp(t, bt, offset[:, j]) for j in range(3)])
    # Extend a short camera tail using the endpoint's corrected velocity.
    for endpoint, other, mask in [(-1, -2, t > bt[-1]), (0, 1, t < bt[0])]:
        if np.any(mask):
            velocity = np.array([np.interp(bt[endpoint], it, vel[:, j]) for j in range(3)]) + (
                offset[endpoint] - offset[other]
            ) / (bt[endpoint] - bt[other])
            speed = np.linalg.norm(velocity)
            if speed > 4:
                velocity *= 4 / speed
            p[mask] = ba[endpoint, 1:4] + (t[mask] - bt[endpoint])[:, None] * velocity
    return p
