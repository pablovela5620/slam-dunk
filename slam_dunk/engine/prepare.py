import sys, json, os
from pathlib import Path
import numpy as np, pycolmap as pc

src, work = map(Path, sys.argv[1:3])
work.mkdir(parents=True, exist_ok=True)
cal = json.loads((src / "calibration.json").read_text())
fac = json.loads((src / "factory_calibration.json").read_text())
ic = next(x for x in fac["ImuCalibrations"] if x["Label"] == "imu-right")
raw = np.atleast_2d(np.loadtxt(src / "imu0/data.csv", delimiter=","))
if raw.shape[1] < 7:
    raise ValueError("IMU rows need timestamp, gyro, and acceleration")
raw = raw[:, :7]
raw = raw[np.isfinite(raw).all(axis=1)]
if len(raw) < 2:
    raise ValueError("fewer than two finite IMU samples")
if np.any(np.diff(raw[:, 0]) < 0):
    raw = raw[np.argsort(raw[:, 0], kind="stable")]
raw = raw[np.r_[True, np.diff(raw[:, 0]) > 0]]
if len(raw) < 2:
    raise ValueError("fewer than two distinct IMU timestamps")
t = raw[:, 0] * 1e-9
rect = []
tt = []
for k, x in [("Gyroscope", raw[:, 1:4]), ("Accelerometer", raw[:, 4:7])]:
    c = ic[k]
    rect.append(
        (x - np.array(c["Bias"]["Offset"]))
        @ np.linalg.inv(np.array(c["Model"]["RectificationMatrix"])).T
    )
    tt.append(
        t - float(c.get("TimeOffsetSec_Device_" + ("Gyro" if k == "Gyroscope" else "Accel"), 0))
    )
# Align the accelerometer and gyro on the gyro's corrected clock.
acc = np.column_stack([np.interp(tt[0], tt[1], rect[1][:, j]) for j in range(3)])
np.c_[tt[0] * 1e9, rect[0], acc].astype(np.float64).tofile(work / "imu.bin")
cc = []
for k in ["cam0", "cam1"]:
    c = cal[k]
    w = c["resolution"]["width"]
    h = c["resolution"]["height"]
    cam = pc.Camera(model=c["model"], width=w, height=h, params=c["params"])
    uv = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).reshape(-1, 2).astype(float)
    xy = cam.cam_from_img(uv)
    rr = np.linalg.norm(xy, axis=1)
    theta = np.arctan(rr)
    eq = xy * (theta / np.maximum(rr, 1e-15))[:, None] * 240 + np.array([w / 2, h / 2])
    valid = np.isfinite(eq).all(axis=1) & (rr < 20)
    eq[~valid] = -1000
    eq.astype(np.float32).tofile(work / (k + "_lut.bin"))
    c["virtual"] = [240.0, 240.0, w / 2, h / 2]
    cc.append(c)
(work / "calib.json").write_text(json.dumps(cc))
rows = [
    x.split(",")
    for x in (src / "cam0/data.csv").read_text().splitlines()
    if x and not x.startswith("#")
]
rows1 = [
    x.split(",")
    for x in (src / "cam1/data.csv").read_text().splitlines()
    if x and not x.startswith("#")
]
t1 = np.array([int(x[0]) for x in rows1], dtype=np.int64)
with (work / "frames.txt").open("w") as f:
    for row in rows:
        t0 = int(row[0])
        j = int(np.searchsorted(t1, t0))
        j = min(j, len(t1) - 1)
        if j and abs(t1[j - 1] - t0) < abs(t1[j] - t0):
            j -= 1
        f.write(f'{t0} {src/"cam0/data"/row[1]} {src/"cam1/data"/rows1[j][1]}\n')
print("Prepared", len(rows), "frames", len(raw), "IMU samples", flush=True)
