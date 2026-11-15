from trajectory_io import load_poses
import os, sys, json, time
from pathlib import Path
import numpy as np, pycolmap as pc
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d

src, work = map(Path, sys.argv[1:3])
start = time.monotonic()
a = load_poses(work / "vio.txt")
cal = json.loads((src / "calibration.json").read_text())
# Sample every half second, keeping temporal detail in the VIO for final interpolation.
step = float(os.environ.get("BA_STEP", ".5"))
sel = [0]
for i in range(1, len(a)):
    if a[i, 0] - a[sel[-1], 0] >= step * 1e9 - 0.01:
        sel.append(i)
if sel[-1] != len(a) - 1:
    sel.append(len(a) - 1)
sel = np.array(sel)
frames = a[sel]
ts = frames[:, 0].astype(np.int64)
t = frames[:, 0] * 1e-9
rot = R.from_quat(frames[:, 4:])
rw = rot.as_matrix()
pos = frames[:, 1:4]
NC = len(frames)
track_scales = (
    np.fromfile(work / "track_scales.bin", np.float32)
    if (work / "track_scales.bin").exists()
    else np.empty(0)
)
rawtrack = np.fromfile(work / os.environ.get("TRACKS_FILE", "tracks.bin"), dtype=np.float64)
track = rawtrack[: rawtrack.size // 6 * 6].reshape(-1, 6)
oi = np.searchsorted(ts, track[:, 0])
ok = oi < len(ts)
ok[ok] &= ts[oi[ok]] == track[ok, 0].astype(np.int64)
scales = track_scales.astype(float) if len(track_scales) == len(track) else np.full(len(track), 3.2)
track = track[ok]
oi = oi[ok]
scales = scales[ok]
flowoffset = int(track[:, 2].max()) + 1 if len(track) else 0
if os.environ.get("MIX_FLOW", "0") == "1":
    from flow_tracks import auxiliary_tracks

    flow, flow_indices = auxiliary_tracks(
        work / "tracks.bin", ts, flowoffset, float(os.environ.get("FLOW_PIXEL_BIAS", ".25"))
    )
    print("Adding", len(flow), "independent flow observations", flush=True)
    track = np.concatenate([track, flow])
    oi = np.r_[oi, flow_indices]
    scales = np.r_[scales, np.full(len(flow), np.nan)]
pixel_bias = float(os.environ.get("SIFT_PIXEL_BIAS", "0"))
track[:, 3:5] -= pixel_bias
if os.environ.get("ARIA_FACTORY_DOMAIN", "0") == "1":
    from factory_domain import supported_pixels

    supported, domain_report = supported_pixels(src, cal, track)
    track = track[supported]
    oi = oi[supported]
    scales = scales[supported]
    (work / "factory-domain.json").write_text(json.dumps(domain_report, indent=2) + "\n")
    print("FACTORY_DOMAIN", json.dumps(domain_report, sort_keys=True), flush=True)
ids = track[:, 2].astype(np.int64)
order = np.argsort(ids, kind="stable")
track = track[order]
oi = oi[order]
scales = scales[order]
ids = ids[order]
unique, first, count = np.unique(ids, return_index=True, return_counts=True)
good = count >= int(os.environ.get("MIN_OBS", "4"))
keep = np.repeat(good, count)
track = track[keep]
oi = oi[keep]
scales = scales[keep]
ids = ids[keep]
unique, first, count = np.unique(ids, return_index=True, return_counts=True)
# Account for the documented residual light sensitivity during SLAM readout.
# Undo the small rotational displacement to put all rays at frame time.
pls = float(os.environ.get("PLS_STRENGTH", "0"))
if pls:
    im = np.fromfile(work / "imu.bin", dtype=np.float64).reshape(-1, 7)
    for c in range(2):
        selc = track[:, 1] == c
        k = cal[f"cam{c}"]
        meta = np.loadtxt(src / f"cam{c}/meta.csv", delimiter=",")
        exposure = np.interp(track[selc, 0], meta[:, 0], meta[:, 1])
        readout = 0.00912 * np.clip(track[selc, 4] / k["resolution"]["height"], 0, 1)
        delay = (
            0.5
            * readout
            * 0.01
            * (1 + readout / np.maximum(exposure, 1e-6))
            / (1 + 0.01 * readout / np.maximum(exposure, 1e-6))
            * pls
        )
        omega = np.column_stack(
            [np.interp(track[selc, 0], im[:, 0], im[:, j]) for j in range(1, 4)]
        )
        cam = pc.Camera(
            model=k["model"],
            width=k["resolution"]["width"],
            height=k["resolution"]["height"],
            params=k["params"],
        )
        xy = cam.cam_from_img(track[selc, 3:5])
        rays = np.c_[xy, np.ones(len(xy))]
        rc = R.from_quat(k["T_b_s"]["qvec"])
        corrected = rc.inv().apply(R.from_rotvec(omega * delay[:, None]).apply(rc.apply(rays)))
        track[selc, 3:5] = cam.img_from_cam(corrected)
M = len(unique)
# Triangulate complete tracks in one batch with least-squares ray intersections.
origin = np.zeros((len(track), 3))
direction = origin.copy()
Ks = []
ext = []
for c in range(2):
    k = cal[f"cam{c}"]
    K = k["params"]
    Ks.extend(K)
    r = R.from_quat(k["T_b_s"]["qvec"])
    p = np.array(k["T_b_s"]["tvec"])
    ext.extend([*r.as_quat()[[3, 0, 1, 2]], *p])
    ci = track[:, 1] == c
    indices = oi[ci]
    cam = pc.Camera(
        model=k["model"], width=k["resolution"]["width"], height=k["resolution"]["height"], params=K
    )
    xy = cam.cam_from_img(track[ci, 3:5])
    ray = np.c_[xy, np.ones(len(xy))]
    ray /= np.linalg.norm(ray, axis=1)[:, None]
    direction[ci] = np.einsum("nij,nj->ni", rw[indices], r.apply(ray))
    origin[ci] = pos[indices] + np.einsum("nij,j->ni", rw[indices], p)
A = np.eye(3)[None, :, :] - direction[:, :, None] * direction[:, None, :]
b = np.einsum("nij,nj->ni", A, origin)
AA = np.add.reduceat(A, first, axis=0)
bb = np.add.reduceat(b, first, axis=0)
try:
    pts = np.linalg.solve(AA + np.eye(3)[None, :, :] * 1e-9, bb)
except:
    pts = np.einsum("nij,nj->ni", np.linalg.pinv(AA), bb)
pi = np.repeat(np.arange(M), count)
dv = pts[pi] - origin
depth = np.einsum("ni,ni->n", dv, direction)
err = 240 * np.linalg.norm(np.cross(dv, direction), axis=1) / np.maximum(depth, 1e-8)
validobs = (
    np.isfinite(err)
    & (depth > 0.3)
    & (depth < 250)
    & (err < float(os.environ.get("TRI_ERR", "3.0")))
)
counts = np.add.reduceat(validobs.astype(int), first)
validpt = (
    (counts >= int(os.environ.get("MIN_OBS", "4")))
    & np.isfinite(pts).all(axis=1)
    & (np.linalg.norm(origin[first] - origin[first + count - 1], axis=1) > 0.1)
)
if os.environ.get("ARIA_HIGH_GAIN_REFIT", "0") == "1":
    from high_gain_refit import refit_high_gain

    pts, depth, err, validobs, validpt = refit_high_gain(
        src,
        ts,
        track,
        oi,
        pts,
        origin,
        direction,
        A,
        b,
        pi,
        first,
        count,
        depth,
        err,
        validobs,
        validpt,
        report_path=work / "high-gain-refit.json",
    )
validobs &= validpt[pi]
mapping = np.cumsum(validpt) - 1
if os.environ.get("SAVE_CANDIDATES"):
    localray = np.zeros((len(track), 3))
    for c in range(2):
        mask = track[:, 1] == c
        k = cal[f"cam{c}"]
        cam = pc.Camera(
            model=k["model"],
            width=k["resolution"]["width"],
            height=k["resolution"]["height"],
            params=k["params"],
        )
        localray[mask, :2] = cam.cam_from_img(track[mask, 3:5])
        localray[mask, 2] = 1
    validcandidate = ids < flowoffset
    with (work / "input.bin.candidates").open("wb") as f:
        np.array([M, int(validcandidate.sum())], np.int64).tofile(f)
        np.where(validpt, mapping, -1).astype(np.int64).tofile(f)
        np.c_[pi, oi, track[:, 1], track[:, 3:5], localray][validcandidate].astype(
            np.float64
        ).tofile(f)

obs = np.c_[oi[validobs], track[validobs, 1], mapping[pi[validobs]], track[validobs, 3:5]]
pts = pts[validpt]
vel = np.gradient(gaussian_filter1d(pos, 1, axis=0), t, axis=0)
imu = np.fromfile(work / "imu.bin", dtype=np.float64).reshape(-1, 7)
imu[:, 0] *= 1e-9
# Correct small residual gyro scale/misalignment from the visual-inertial rotations.
# Leave the factory calibration as the starting sensor model.
pose = np.c_[rot.as_quat()[:, [3, 0, 1, 2]], pos]
with (work / "input.bin").open("wb") as f:
    np.array([len(t), len(pts), len(obs), len(imu)], np.int64).tofile(f)
    for x in [np.array(Ks), np.array(ext), t, pose, vel, pts, obs, imu]:
        np.ascontiguousarray(x, dtype=np.float64).tofile(f)
np.where(unique[validpt] >= flowoffset, float(os.environ.get("FLOW_WEIGHT", ".25")), 1.0).astype(
    np.float64
).tofile(work / "input.bin.weights")
measurement_scale = scales[validobs]
good_scale = np.isfinite(measurement_scale) & (measurement_scale > 0)
weights = np.ones(len(obs))
power = float(os.environ.get("SIFT_SCALE_POWER", "1"))
if good_scale.any():
    weights[good_scale] = np.clip(
        (np.median(measurement_scale[good_scale]) / measurement_scale[good_scale]) ** power,
        0.2,
        2.0,
    )
    weights[good_scale] /= weights[good_scale].mean()
weights.astype(np.float64).tofile(work / "input.bin.obsweights")
# Keep the same localization precision for observations recovered later.
# Normalize with the original accepted SIFT observations, then apply that
# unchanged scale to candidate observations from the same extraction.
if os.environ.get("SAVE_CANDIDATES"):
    candidate_precision = np.ones(len(track))
    candidate_good = np.isfinite(scales) & (scales > 0)
    if good_scale.any():
        median_scale = np.median(measurement_scale[good_scale])
        normalization = np.mean(
            np.clip((median_scale / measurement_scale[good_scale]) ** power, 0.2, 2.0)
        )
        candidate_precision[candidate_good] = (
            np.clip((median_scale / scales[candidate_good]) ** power, 0.2, 2.0) / normalization
        )
    candidate_precision[validcandidate].astype(np.float64).tofile(
        work / "input.bin.candidateweights"
    )
if os.environ.get("SAVE_META", "0") == "1":
    from camera_meta import metadata_at

    ex = []
    temperatures = []
    for c in range(2):
        exposure, temperature = metadata_at(src / f"cam{c}/meta.csv", ts)
        ex.append(exposure)
        temperatures.append(temperature)
    omega = np.column_stack([np.interp(t, imu[:, 0], imu[:, j]) for j in range(1, 4)])
    with (work / "input.bin.meta").open("wb") as f:
        np.array([cal[f"cam{c}"]["resolution"]["height"] for c in range(2)], np.float64).tofile(f)
        np.c_[ex[0], ex[1], omega, temperatures[0], temperatures[1]].astype(np.float64).tofile(f)
np.savez(work / "export.npz", sel=sel, t=ts)
print(
    "Exported",
    len(t),
    "poses",
    len(pts),
    "landmarks",
    len(obs),
    "observations in",
    time.monotonic() - start,
    flush=True,
)
