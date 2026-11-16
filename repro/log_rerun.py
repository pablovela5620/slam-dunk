"""Log one SLAM Dunk run's real intermediates to a Rerun .rrd for the deep page.

Reads only files the pipeline wrote (frontend packet, backend work dirs) plus the
official-evaluator geometry dumped by cp_dump.py. Entities:

  /frontend/cam{0,1}/image, /flow, /flow_trails, /sift   Basalt KLT tracks + RootSIFT keyframe tracks
  /world/control_points                                  surveyed CPs (LV95 topo frame, local origin)
  /world/<cfg>/path, /now, /cp_triangulated, /cp_error   trajectories mapped with that config's
                                                          official CP Sim(3) (topo = s R x + t)
  /world/pgt                                             pseudo-GT, Sim(3)-fitted to full (display only)
  /world/landmarks                                       full BA landmarks (SAVE_STATE)
  /calib/<cfg>/...                                       per-5 s chunk focal and stereo drift
  /imu/<cfg>/...                                         gyro bias, gyro scale/misalignment
  /sensors/temperature_c                                 camera-record temperature (the thermal input)

Run in the `viz` pixi environment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

CONFIG_COLORS = {
    "vio": (255, 159, 67),
    "baseline": (84, 160, 255),
    "camera_only": (200, 120, 255),
    "gyro_only": (255, 107, 129),
    "full": (46, 213, 115),
}
PGT_COLOR = (200, 200, 200)
WORLD_META: dict = {}
CP_COLOR = (255, 221, 89)


@dataclass(frozen=True)
class Sim3:
    """topo = scale * R @ x + t (the official evaluator's alignment)."""

    scale: float
    rotation: Rotation
    translation: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.scale * self.rotation.apply(x) + self.translation


def load_traj(path: Path) -> np.ndarray:
    """Load an 8-column trajectory (t_ns x y z qx qy qz qw), sorted, finite rows only."""
    a = np.loadtxt(path, ndmin=2, comments="#")
    a = a[np.isfinite(a).all(axis=1)]
    return a[np.argsort(a[:, 0], kind="stable")]


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Sim3:
    """Least-squares similarity dst ≈ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rot = u @ sign @ vt
    scale = np.trace(np.diag(d) @ sign) / xs.var(0).sum()
    return Sim3(float(scale), Rotation.from_matrix(rot), mu_d - scale * rot @ mu_s)


def match_times(a: np.ndarray, b: np.ndarray, tol_ns: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices (ia, ib) of nearest-time matches within tol_ns."""
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    left = b[idx - 1]
    right = b[idx]
    idx = np.where(np.abs(a - left) <= np.abs(a - right), idx - 1, idx)
    ok = np.abs(b[idx] - a) <= tol_ns
    return np.nonzero(ok)[0], idx[ok]


def sim3_from_dump(dump: dict) -> Sim3:
    al = dump["alignment"]
    return Sim3(al["scale"], Rotation.from_quat(al["quat_xyzw"]), np.asarray(al["translation"]))


def log_frontend(packet: Path, prep: Path, t0: int, stride: int, trail: int) -> None:
    """Images with KLT flow tracks (+ short trails) and RootSIFT keyframe observations."""
    frontend = packet / "frontend"
    flow = np.fromfile(frontend / "tracks.bin", np.float64)
    flow = flow[: flow.size // 6 * 6].reshape(-1, 6)
    sift = np.fromfile(frontend / "tracks_ba.bin", np.float64)
    sift = sift[: sift.size // 6 * 6].reshape(-1, 6)
    for cam in (0, 1):
        stamps = np.loadtxt(prep / f"cam{cam}/data.csv", delimiter=",", usecols=0, dtype=np.int64)
        rr.log(f"frontend/cam{cam}", rr.Pinhole(resolution=[640, 480], focal_length=240.0), static=True)
        fc = flow[flow[:, 1] == cam]
        sc = sift[sift[:, 1] == cam]
        # Frontend stores the cam0 timestamp for both cameras' observations.
        cam0_stamps = np.loadtxt(prep / "cam0/data.csv", delimiter=",", usecols=0, dtype=np.int64)
        f_t = fc[:, 0].astype(np.int64)
        order = np.lexsort((f_t, fc[:, 2]))
        fc_sorted, ft_sorted = fc[order], f_t[order]
        ids_sorted = fc_sorted[:, 2].astype(np.int64)
        by_time = {}
        for i, t in enumerate(f_t):
            by_time.setdefault(int(t), []).append(i)
        s_t = sc[:, 0].astype(np.int64)
        sift_by_time = {}
        for i, t in enumerate(s_t):
            sift_by_time.setdefault(int(t), []).append(i)
        frame_dt = int(np.median(np.diff(cam0_stamps)))
        for k in range(0, len(cam0_stamps), stride):
            t = int(cam0_stamps[k])
            img_t = int(stamps[min(k, len(stamps) - 1)])
            rr.set_time("time", duration=(t - t0) * 1e-9)
            jpg = prep / f"cam{cam}/data/{img_t}.jpg"
            if jpg.is_file():
                rr.log(f"frontend/cam{cam}/image", rr.EncodedImage(path=jpg))
            rows = by_time.get(t, [])
            if rows:
                pts = fc[rows]
                ids = pts[:, 2].astype(np.int64)
                rr.log(
                    f"frontend/cam{cam}/flow",
                    rr.Points2D(pts[:, 3:5], radii=1.5, class_ids=(ids % 255).astype(np.uint16)),
                )
                strips = []
                for tid in ids:
                    lo = np.searchsorted(ids_sorted, tid, "left")
                    hi = np.searchsorted(ids_sorted, tid, "right")
                    tt = ft_sorted[lo:hi]
                    sel = (tt <= t) & (tt > t - trail * frame_dt)
                    if sel.sum() >= 2:
                        strips.append(fc_sorted[lo:hi][sel][:, 3:5])
                rr.log(
                    f"frontend/cam{cam}/flow_trails",
                    rr.LineStrips2D(strips, radii=0.6, colors=(0, 210, 255)),
                )
            else:
                rr.log(f"frontend/cam{cam}/flow", rr.Clear(recursive=False))
        # RootSIFT keyframe observations at their own (keyframe) times; they persist
        # in the view until the next keyframe.
        for t_kf, rows in sift_by_time.items():
            rr.set_time("time", duration=(t_kf - t0) * 1e-9)
            rr.log(f"frontend/cam{cam}/sift", rr.Points2D(sc[rows][:, 3:5], radii=3.0, colors=(255, 80, 80)))


def log_world(
    trajs: dict[str, np.ndarray],
    dumps: dict[str, dict],
    pgt: np.ndarray | None,
    landmarks: np.ndarray | None,
    t0: int,
    now_stride_s: float,
) -> None:
    """Trajectories, CPs, landmarks in the surveyed (topo) frame."""
    full_sim = sim3_from_dump(dumps["full"])
    origin = np.array([p["topo"] for p in dumps["full"]["points"]]).mean(0)
    origin[2] = 0.0  # keep heights, drop the LV95 east/north offset
    cps = dumps["full"]["points"]
    rr.log(
        "world/control_points",
        rr.Points3D(
            [np.asarray(p["topo"]) - origin for p in cps],
            radii=-7.0,
            colors=CP_COLOR,
            labels=[p["name"] for p in cps],
            show_labels=True,
        ),
        static=True,
    )
    for name, traj in trajs.items():
        if name not in dumps or "alignment" not in dumps[name]:
            continue
        sim = sim3_from_dump(dumps[name])
        xyz = sim.apply(traj[:, 1:4]) - origin
        rr.log(
            f"world/{name}/path",
            rr.LineStrips3D([xyz], radii=-1.6, colors=CONFIG_COLORS[name]),
            static=True,
        )
        tri, err_lines = [], []
        for p in dumps[name]["points"]:
            if p["triangulated"] is None:
                continue
            q = sim.apply(np.asarray(p["triangulated"])[None])[0] - origin
            surveyed = np.asarray(p["topo"]) - origin
            if surveyed[2] == -origin[2]:
                surveyed[2] = q[2]  # CP without surveyed height
            tri.append(q)
            err_lines.append([q, surveyed])
        if tri:
            rr.log(
                f"world/{name}/cp_triangulated",
                rr.Points3D(tri, radii=-4.5, colors=CONFIG_COLORS[name]),
                static=True,
            )
            rr.log(
                f"world/{name}/cp_error",
                rr.LineStrips3D(err_lines, radii=-1.2, colors=CONFIG_COLORS[name]),
                static=True,
            )
        errs = [p["error_2d_m"] if p["error_2d_m"] is not None else 10.0 for p in dumps[name]["points"]]
        rr.log(f"cp_error_cm/{name}", rr.BarChart(np.asarray(errs) * 100.0, color=CONFIG_COLORS[name]), static=True)
        # Moving marker, sampled.
        times = (traj[:, 0] - t0) * 1e-9
        keep = np.r_[0, np.nonzero(np.diff(np.floor(times / now_stride_s)))[0] + 1]
        rr.send_columns(
            f"world/{name}/now",
            indexes=[rr.TimeColumn("time", duration=times[keep])],
            columns=rr.Points3D.columns(positions=xyz[keep]).partition([1] * len(keep)),
        )
        rr.log(
            f"world/{name}/now",
            rr.Points3D.from_fields(radii=-6.0, colors=CONFIG_COLORS[name]),
            static=True,
        )
    full_local = full_sim.apply(trajs["full"][:, 1:4]) - origin
    WORLD_META.update(lo=full_local.min(0).tolist(), hi=full_local.max(0).tolist())
    if pgt is not None:
        full_xyz = full_sim.apply(trajs["full"][:, 1:4])
        ia, ib = match_times(pgt[:, 0], trajs["full"][:, 0], 2e6)
        if len(ia) > 10:
            fit = umeyama_sim3(pgt[ia, 1:4], full_xyz[ib])
            rr.log(
                "world/pgt",
                rr.LineStrips3D([fit.apply(pgt[:, 1:4]) - origin], radii=-1.0, colors=PGT_COLOR),
                static=True,
            )
    if landmarks is not None and len(landmarks):
        pts = full_sim.apply(landmarks) - origin
        lo_b, hi_b = full_local.min(0) - 40.0, full_local.max(0) + 40.0
        pts = pts[((pts >= lo_b) & (pts <= hi_b)).all(1)]
        z = pts[:, 2]
        lo, hi = np.percentile(z, [2, 98])
        u = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
        colors = np.stack([80 + 120 * u, 90 + 60 * u, 140 + 80 * (1 - u)], 1).astype(np.uint8)
        rr.log("world/landmarks", rr.Points3D(pts, radii=-1.0, colors=colors), static=True)


def rotation_angle_deg(q_xyzw: np.ndarray, q0_xyzw: np.ndarray) -> np.ndarray:
    return np.degrees((Rotation.from_quat(q0_xyzw).inv() * Rotation.from_quat(q_xyzw)).magnitude())


SHADES = {
    "full": [(46, 213, 115), (123, 237, 159), (0, 168, 120), (85, 239, 196)],
    "baseline": [(84, 160, 255), (162, 205, 255), (30, 110, 220), (120, 180, 240)],
    "camera_only": [(200, 120, 255), (225, 180, 255), (160, 80, 220), (190, 150, 240)],
    "gyro_only": [(255, 107, 129), (255, 170, 185), (220, 60, 90), (240, 140, 160)],
}


def series(path: str, cfg: str, j: int, name: str) -> None:
    """Static SeriesLines style: config colour family, readable legend name."""
    rr.log(path, rr.SeriesLines(colors=[SHADES.get(cfg, SHADES["full"])[j % 4]], names=[f"{cfg} {name}"], widths=[2.0]),
           static=True)


def log_calibration(cfg: str, work: Path, t_first_s: float, t0: int, chunk_dt: float) -> None:
    """Per-chunk camera and IMU states from batch_joint's SAVE_STATE / diag sidecars.

    Packet camera 0 is the recording's cam1 (slam-right, stereo reference); packet
    camera 1 is cam0 (slam-left), stored relative to the reference.
    """
    focals = np.loadtxt(work / "ba.txt.focals", ndmin=2)
    diag = np.loadtxt(work / "ba.txt.diag", ndmin=2)
    n = len(diag)
    times = (t_first_s - t0 * 1e-9) + chunk_dt * (np.arange(n) + 0.5)
    tc = [rr.TimeColumn("time", duration=times)]
    cams = json.loads((work / "calib.json").read_text())  # packet order: [right, left]
    f_ref = np.array([cams[0]["params"][0], cams[0]["params"][1], cams[1]["params"][0], cams[1]["params"][1]])
    if len(focals) == 1:
        focals = np.repeat(focals, n, axis=0)
    rel_ppt = (focals / f_ref - 1.0) * 1e3  # per mille
    for j, label in enumerate(("right_fx", "right_fy", "left_fx", "left_fy")):
        series(f"calib/{cfg}/focal_permille/{label}", cfg, j, label)
        rr.send_columns(f"calib/{cfg}/focal_permille/{label}", indexes=tc,
                        columns=rr.Scalars.columns(scalars=rel_ppt[:, j]))
    bg, ba = diag[:, 1:4], diag[:, 4:7]
    q = diag[:, [8, 9, 10, 7]]  # stored wxyz -> xyzw
    t = diag[:, 11:14]
    base = np.linalg.norm(t, axis=1)
    series(f"calib/{cfg}/stereo/baseline_change_mm", cfg, 0, "baseline length")
    series(f"calib/{cfg}/stereo/rotation_change_deg", cfg, 0, "rotation")
    rr.send_columns(f"calib/{cfg}/stereo/baseline_change_mm", indexes=tc,
                    columns=rr.Scalars.columns(scalars=(base - base[0]) * 1e3))
    rr.send_columns(f"calib/{cfg}/stereo/rotation_change_deg", indexes=tc,
                    columns=rr.Scalars.columns(scalars=rotation_angle_deg(q, q[:1].repeat(n, 0))))
    for j, ax in enumerate("xyz"):
        series(f"imu/{cfg}/gyro_bias_mdps/{ax}", cfg, j, f"gyro bias {ax}")
        series(f"imu/{cfg}/accel_bias_mg/{ax}", cfg, j, f"accel bias {ax}")
        rr.send_columns(f"imu/{cfg}/gyro_bias_mdps/{ax}", indexes=tc,
                        columns=rr.Scalars.columns(scalars=np.degrees(bg[:, j]) * 1e3))
        rr.send_columns(f"imu/{cfg}/accel_bias_mg/{ax}", indexes=tc,
                        columns=rr.Scalars.columns(scalars=ba[:, j] / 9.806 * 1e3))


def log_gyro(cfg: str, work: Path, temp_t: np.ndarray, temp_c: np.ndarray) -> dict:
    """Gyro response w = (I + G) w_raw + (T - Tref) g_t ⊙ w_raw - b, logged over time."""
    G = np.loadtxt(work / "ba.txt.gyromat").reshape(3, 3)
    gt = np.loadtxt(work / "ba.txt.gyrot").ravel()
    gthermal, tref = gt[:3], gt[3] if len(gt) > 3 else float(np.mean(temp_c))
    tc = [rr.TimeColumn("time", duration=temp_t)]
    dT = temp_c - tref
    for j, ax in enumerate("xyz"):
        scale_ppm = (G[j, j] + gthermal[j] * dT) * 1e6
        series(f"imu/{cfg}/gyro_scale_ppm/{ax}", cfg, j, f"gyro scale {ax}")
        rr.send_columns(f"imu/{cfg}/gyro_scale_ppm/{ax}", indexes=tc,
                        columns=rr.Scalars.columns(scalars=scale_ppm))
    sym = 0.5 * (G + G.T)
    skew = 0.5 * (G - G.T)
    labels = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    for name, (a, b) in labels.items():
        val = np.full(len(temp_t), sym[a, b] * 1e3)
        rr.send_columns(f"imu/{cfg}/gyro_nonorthogonality_mrad/{name}", indexes=tc,
                        columns=rr.Scalars.columns(scalars=val))
        val = np.full(len(temp_t), skew[a, b] * 1e3)
        rr.send_columns(f"imu/{cfg}/gyro_rotation_mrad/{name}", indexes=tc,
                        columns=rr.Scalars.columns(scalars=val))
    return dict(G=G.tolist(), g_thermal=gthermal.tolist(), t_ref=float(tref))


def blueprint() -> rrb.Blueprint:
    """Default layout: top-down world, both cameras, calibration/IMU plots (whole time range)."""
    eye = None
    if WORLD_META:
        lo, hi = np.asarray(WORLD_META["lo"]), np.asarray(WORLD_META["hi"])
        c = 0.5 * (lo + hi)
        ext = float(max(hi[0] - lo[0], hi[1] - lo[1]))
        eye = rrb.archetypes.EyeControls3D(kind="Orbital", position=[c[0], c[1] - 0.3 * ext, c[2] + 1.1 * ext],
                                           look_target=c.tolist(), eye_up=[0, 0, 1])
    whole = rrb.archetypes.TimeAxis(view_range=rrb.TimeRange(rrb.TimeRangeBoundary.infinite(), rrb.TimeRangeBoundary.infinite()))
    world = rrb.Spatial3DView(
        origin="world",
        name="Trajectories in the surveyed frame",
        contents=["+ /world/**"],
        line_grid=rrb.archetypes.LineGrid3D(visible=False),
        eye_controls=eye,
    )
    cams = rrb.Horizontal(
        rrb.Spatial2DView(origin="frontend/cam0", name="cam0 tracks"),
        rrb.Spatial2DView(origin="frontend/cam1", name="cam1 tracks"),
    )
    plots = rrb.Tabs(
        rrb.TimeSeriesView(origin="calib/full/focal_permille", name="Focal drift ‰ (full)", axis_x=whole),
        rrb.TimeSeriesView(origin="calib", name="Stereo drift", contents=["+ /calib/full/stereo/**", "+ /calib/baseline/stereo/**"], axis_x=whole),
        rrb.TimeSeriesView(origin="imu", name="Gyro scale ppm", contents=["+ /imu/full/gyro_scale_ppm/**", "+ /imu/baseline/gyro_scale_ppm/**"], axis_x=whole),
        rrb.TimeSeriesView(origin="sensors", name="Temperature °C", axis_x=whole),
    )
    return rrb.Blueprint(
        rrb.Horizontal(world, rrb.Vertical(cams, plots, row_shares=[1, 1]), column_shares=[3, 2]),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--run", type=Path, required=True, help="runs/<seq> (packet/, <cfg>/, cp-<cfg>.json)")
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", default=["baseline", "full"])
    ap.add_argument("--image-stride", type=int, default=10)
    ap.add_argument("--trail", type=int, default=12)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    rr.init("slam_dunk_" + args.seq, recording_id="slam-dunk-" + args.seq)
    rr.save(args.out)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    cam0 = np.loadtxt(args.prep / "cam0/data.csv", delimiter=",", usecols=0, dtype=np.int64)
    t0 = int(cam0[0])
    packet = args.run / "packet"

    meta = np.loadtxt(args.prep / "cam0/meta.csv", delimiter=",", ndmin=2)
    temp_t = (meta[:, 0] - t0) * 1e-9
    step = max(1, len(meta) // 2000)
    temp_t, temp_c = temp_t[::step], meta[::step, 3]
    rr.log("sensors/temperature_c", rr.SeriesLines(colors=[(255, 177, 66)], names=["camera temperature"], widths=[2.0]), static=True)
    rr.send_columns("sensors/temperature_c", indexes=[rr.TimeColumn("time", duration=temp_t)],
                    columns=rr.Scalars.columns(scalars=temp_c))

    trajs: dict[str, np.ndarray] = {"vio": load_traj(packet / "frontend/vio.txt")}
    dumps: dict[str, dict] = {}
    for name in ["vio", *args.configs]:
        dump_path = args.run / f"cp-{name}.json"
        if dump_path.is_file():
            dumps[name] = json.loads(dump_path.read_text())
    gyro = {}
    settings = json.loads((Path(__file__).resolve().parents[1] / "slam_dunk/native_defaults.json").read_text())
    chunk_dt = float(settings.get("CHUNK_DT", 5))
    for cfg in args.configs:
        work = args.run / cfg / "work"
        trajs[cfg] = load_traj(args.run / cfg / "trajectory.txt")
        kf = np.loadtxt(work / "ba.txt", ndmin=2)
        log_calibration(cfg, work, kf[0, 0] * 1e-9, t0, chunk_dt)
        gyro[cfg] = log_gyro(cfg, work, temp_t, temp_c)
        for p in ("ba.txt.gyromat", "ba.txt.gyrot", "ba.txt.timing", "ba.txt.accscale", "ba.txt.accalign"):
            if (work / p).is_file():
                rr.log(f"state/{cfg}/{p.removeprefix('ba.txt.')}",
                       rr.TextDocument((work / p).read_text()), static=True)

    pgt = load_traj(args.raw / "pgt.txt") if (args.raw / "pgt.txt").is_file() else None
    landmarks = None
    pts_path = args.run / "full/work/ba.txt.points"
    if pts_path.is_file():
        landmarks = np.fromfile(pts_path, np.float64).reshape(-1, 3)
        landmarks = landmarks[np.isfinite(landmarks).all(1)]
        if len(landmarks) > 60000:
            landmarks = landmarks[np.random.default_rng(0).choice(len(landmarks), 60000, replace=False)]
    log_world(trajs, dumps, pgt, landmarks, t0, now_stride_s=0.5)

    if not args.no_images:
        log_frontend(packet, args.prep, t0, args.image_stride, args.trail)
    bp = blueprint()
    rr.send_blueprint(bp, make_active=True, make_default=True)
    bp.save("slam_dunk_" + args.seq, str(args.out.with_suffix(".rbl")))
    (args.out.with_suffix(".gyro.json")).write_text(json.dumps(gyro, indent=2) + "\n")
    (args.out.with_suffix(".meta.json")).write_text(json.dumps(dict(WORLD_META, configs=args.configs), indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
