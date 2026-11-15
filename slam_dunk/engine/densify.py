from trajectory_io import load_poses, nearest_indices, save_poses
import sys, os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

src, sparse, out = map(Path, sys.argv[1:4])
ts = np.loadtxt(src / "cam0/data.csv", delimiter=",", usecols=0, dtype=np.int64)
a = load_poses(sparse)
a = np.atleast_2d(a)
a = a[np.isfinite(a).all(axis=1)]
a = a[np.r_[True, np.diff(a[:, 0]) > 0]]
if not len(a):
    raise RuntimeError("no poses")
t0 = ts[0]
t = (ts - t0) * 1e-9
st = (a[:, 0] - t0) * 1e-9
p = np.column_stack([np.interp(t, st, a[:, j]) for j in range(1, 4)])
if len(a) > 1:
    imu_path = sparse.parent / "imu.bin"
    if imu_path.exists():
        from gyro_interp import integrate

        imu = np.fromfile(imu_path, dtype=np.float64).reshape(-1, 7)
        rg = Rotation.from_quat(integrate((imu[:, 0] - t0) * 1e-9, imu[:, 1:4], t))
        idx = nearest_indices(ts, a[:, 0].astype(np.int64))
        correction = Rotation.from_quat(a[:, 4:]) * rg[idx].inv()
        q = (Slerp(st, correction)(np.clip(t, st[0], st[-1])) * rg).as_quat()
    else:
        q = Slerp(st, Rotation.from_quat(a[:, 4:]))(np.clip(t, st[0], st[-1])).as_quat()
else:
    q = np.tile(a[0, 4:], (len(ts), 1))

if not (np.isfinite(p).all() and np.isfinite(q).all()):
    raise RuntimeError("non-finite reconstructed pose")
save_poses(out, ts, p, q)
