#!/usr/bin/env python3
"""Score trajectories exactly as the grader does.

  python /app/evaluate.py EST.txt --seq sequence_1_19  score EST on a dev sequence
  python /app/evaluate.py EST.txt --gt GT.txt          evo metrics against any trajectory file
  python /app/evaluate.py --check /app/submission      check the held-out submission (no ground truth)

Add --plot out.png for a top-down plot of the SE(3)-aligned trajectories, and
--no-cp to skip the control-point evaluation (about a minute per sequence).

Trajectory files: one pose per line, "timestamp tx ty tz qx qy qz qw", with the
timestamp in integer nanoseconds and the pose T_world_imu (meters, Hamilton
quaternion, scalar last). Lines starting with # are ignored.

The grade is cp_score, LaMAria's control-point score (0-100), computed by
LaMAria's own evaluator (see /opt/lamaria-eval/lamaria_score.py for the
steps) on the sequences that have control points: the 2 held-out ones and 9
dev ones. It uses the pose at each left-camera frame: the nearest estimated
pose within 1 ms. Reported with it: CP recall at 1 m, and pose recall at
1 m / 5 m against the ground truth after the control-point alignment.

The evo metrics compare each ground-truth pose with the nearest-in-time
estimated pose, so a gap is scored as if the last pose were held: the ATE
(position RMSE after a rigid SE(3) Umeyama alignment, evo_ape -a), rotation
error, RPE over 10 m, and the ATE after a Sim(3) alignment with its scale.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("LAMARIA_ROOT", "/data/lamaria"))
LAMARIA_EVAL = Path(os.environ.get("LAMARIA_EVAL", "/opt/lamaria-eval"))
HELDOUT = ("sequence_3_17",)
MATCH_NS = 1_000_000  # a pose counts for a frame if it is within 1 ms
CP_KEYS = ("cp_score", "cp_recall_1m", "pose_recall_1m", "pose_recall_5m")


class TrajectoryError(ValueError):
    pass


def read_trajectory(path):
    """Returns (timestamps [ns] int64, positions (N, 3), quaternions xyzw (N, 4)), sorted by time."""
    ts, pose = [], []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 8:
            raise TrajectoryError(f"{path}:{n}: expected 8 values, got {len(parts)}")
        try:
            t = int(parts[0]) if parts[0].lstrip("-").isdigit() else int(round(float(parts[0])))
            v = [float(x) for x in parts[1:]]
        except ValueError as e:
            raise TrajectoryError(f"{path}:{n}: {e}") from None
        if not np.all(np.isfinite(v)):
            raise TrajectoryError(f"{path}:{n}: non-finite value")
        ts.append(t)
        pose.append(v)
    if len(ts) < 3:
        raise TrajectoryError(f"{path}: fewer than 3 poses")
    ts, pose = np.array(ts, dtype=np.int64), np.array(pose)
    order = np.argsort(ts, kind="stable")
    ts, pose = ts[order], pose[order]
    keep = np.r_[True, np.diff(ts) > 0]
    q = pose[keep, 3:]
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm < 1e-6):
        raise TrajectoryError(f"{path}: zero quaternion")
    return ts[keep], pose[keep, :3], q / norm[:, None]


def cam0_timestamps(seq_dir):
    rows = (seq_dir / "cam0" / "data.csv").read_text().splitlines()
    return np.array([int(r.split(",")[0]) for r in rows if r and not r.startswith("#")], dtype=np.int64)


def nearest(stamps, t):
    """Index into sorted stamps of the nearest timestamp for each t."""
    i = np.clip(np.searchsorted(stamps, t), 1, len(stamps) - 1)
    return np.where(np.abs(stamps[i - 1] - t) <= np.abs(stamps[i] - t), i - 1, i)


def _evo(ts, p, q):
    from evo.core.trajectory import PoseTrajectory3D

    return PoseTrajectory3D(positions_xyz=p, orientations_quat_wxyz=q[:, [3, 0, 1, 2]], timestamps=ts * 1e-9)


def evaluate(gt, est):
    """gt, est: (timestamps, positions, quaternions). Returns (metrics, aligned estimate, reference)."""
    from evo.core import metrics

    gt_ts, gt_p, gt_q = gt
    est_ts, est_p, est_q = est
    idx = nearest(est_ts, gt_ts)
    covered = np.abs(est_ts[idx] - gt_ts) <= MATCH_NS
    ref = _evo(gt_ts, gt_p, gt_q)
    traj = _evo(gt_ts, est_p[idx], est_q[idx])

    se3 = copy.deepcopy(traj)
    se3.align(ref, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, se3))
    stats = ape.get_all_statistics()
    rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    rot.process_data((ref, se3))

    out = {
        "ate_rmse_m": float(stats["rmse"]),
        "ate_mean_m": float(stats["mean"]),
        "ate_median_m": float(stats["median"]),
        "ate_max_m": float(stats["max"]),
        "rot_rmse_deg": float(rot.get_statistic(metrics.StatisticsType.rmse)),
        "gt_poses": int(len(gt_ts)),
        "coverage": float(covered.mean()),
    }
    try:
        rpe = metrics.RPE(
            metrics.PoseRelation.translation_part, delta=10, delta_unit=metrics.Unit.meters, pairs_from_reference=True
        )
        rpe.process_data((ref, se3))
        out["rpe_10m_rmse_m"] = float(rpe.get_statistic(metrics.StatisticsType.rmse))
    except Exception as e:
        out.update(rpe_10m_rmse_m=None, rpe_error=str(e))
    try:
        sim3 = copy.deepcopy(traj)
        _, _, s = sim3.align(ref, correct_scale=True)
        ape_s = metrics.APE(metrics.PoseRelation.translation_part)
        ape_s.process_data((ref, sim3))
        out.update(ate_sim3_rmse_m=float(ape_s.get_statistic(metrics.StatisticsType.rmse)), sim3_scale=float(s))
    except Exception as e:  # e.g. a constant trajectory has no scale
        out.update(ate_sim3_rmse_m=None, sim3_scale=None, sim3_error=str(e))
    return out, se3, ref


def frame_poses(est, frames):
    """The estimated pose at each frame: the nearest one within 1 ms, stamped with the frame's time."""
    ts, p, q = est
    idx = nearest(ts, frames)
    ok = np.abs(ts[idx] - frames) <= MATCH_NS
    return frames[ok], p[idx[ok]], q[idx[ok]]


def write_trajectory(path, ts, p, q):
    with open(path, "w") as f:
        for t, pos, quat in zip(ts, p, q):
            f.write(f"{int(t)} " + " ".join(repr(float(v)) for v in (*pos, *quat)) + "\n")


def cp_evaluate(est, seq_dir, control_points, gt_path):
    """LaMAria's control-point metrics of est on the recording in seq_dir (0 if it can't be aligned)."""
    frames = cam0_timestamps(seq_dir)
    ts, p, q = frame_poses(est, frames)
    out = {"frames_with_pose": float(len(ts) / len(frames))}
    if len(ts) < 3:
        return {**out, **dict.fromkeys(CP_KEYS, 0.0), "cp_error": "fewer than 3 poses at left-camera frames"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "estimate.txt"
        write_trajectory(path, ts, p, q)
        cmd = [str(LAMARIA_EVAL / "venv/bin/python"), str(LAMARIA_EVAL / "lamaria_score.py"), str(path),
               str(control_points), str(seq_dir / "calibration.json"), "--gt", str(gt_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise RuntimeError(f"lamaria_score.py exited {r.returncode} without a result:\n{r.stderr[-3000:]}") from None
    if "error" in res:
        res["cp_error"] = res.pop("error")
        res.pop("traceback", None)
    return {**out, **dict.fromkeys(CP_KEYS, 0.0), **res}


def check_submission(folder):
    """Format and coverage of each held-out trajectory, without ground truth."""
    ok = True
    for seq in HELDOUT:
        path = Path(folder) / f"{seq}.txt"
        if not path.exists():
            print(f"{seq}: MISSING {path}")
            ok = False
            continue
        try:
            ts, _, _ = read_trajectory(path)
        except TrajectoryError as e:
            print(f"{seq}: INVALID {e}")
            ok = False
            continue
        frames = cam0_timestamps(DATA / "heldout" / seq)
        gap = np.abs(ts[nearest(ts, frames)] - frames)
        missing = int(np.sum(gap > MATCH_NS))
        print(f"{seq}: {len(ts)} poses, {missing} of {len(frames)} cam0 frames without a pose within 1 ms")
        ok &= missing == 0
    print("OK" if ok else "NOT OK")
    return ok


def plot(path, ref, est, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ref.positions_xyz[:, 0], ref.positions_xyz[:, 1], "k-", lw=1, label="ground truth")
    ax.plot(est.positions_xyz[:, 0], est.positions_xyz[:, 1], "r-", lw=1, label="estimate (SE(3)-aligned)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("estimate", nargs="?")
    ap.add_argument("--seq", help="dev sequence name, e.g. R_11_5cp")
    ap.add_argument("--gt", help="ground-truth file")
    ap.add_argument("--check", metavar="DIR", help="check a held-out submission folder")
    ap.add_argument("--plot", metavar="PNG")
    ap.add_argument("--no-cp", action="store_true", help="skip the control-point evaluation")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check_submission(args.check) else 1)
    if not args.estimate or not (args.seq or args.gt):
        ap.error("give an estimate and --seq or --gt (or --check DIR)")
    gt_path = Path(args.gt) if args.gt else DATA / "dev" / args.seq / "groundtruth.txt"
    estimate = read_trajectory(args.estimate)
    metrics_, est, ref = evaluate(read_trajectory(gt_path), estimate)
    seq_dir = DATA / "dev" / str(args.seq)
    if args.seq and not args.no_cp:
        if (seq_dir / "control_points.json").exists():
            metrics_ = {**cp_evaluate(estimate, seq_dir, seq_dir / "control_points.json", gt_path), **metrics_}
        else:
            metrics_ = {"cp_score": None, "cp_note": f"{args.seq} has no control points", **metrics_}
    print(json.dumps(metrics_, indent=1))
    if args.plot:
        plot(args.plot, ref, est, f"{args.seq or gt_path.name}: ATE {metrics_['ate_rmse_m']:.2f} m")


if __name__ == "__main__":
    main()
