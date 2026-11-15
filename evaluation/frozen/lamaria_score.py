#!/usr/bin/env python3
"""LaMAria's own control-point evaluation of one trajectory.

Runs in /opt/lamaria-eval/venv, which holds the evaluator's dependencies
(LaMAria's pycolmap 3.13 dev build among them); /app/evaluate.py calls it.

  lamaria_score.py EST CP_JSON CALIBRATION_JSON [--gt GT.txt]

EST holds T_world_imu poses, "timestamp tx ty tz qx qy qz qw", each within
1 ms of a left-camera frame. The steps are those of the evaluator's
evaluate_wrt_control_points.py and evaluate_wrt_pgt.py (cvg/lamaria at
238b6ca, in /opt/lamaria-eval/src):

1. Each control point is triangulated from its detections in the images,
   using the camera poses given by EST and the calibration.
2. A robust Sim(3) fit maps the triangulated points onto the surveyed ones,
   refined by a sparse alignment.
3. cp_score: each control point's horizontal error scores 100 at 5 cm or
   less, 90 at 20 cm, 75 at 50 cm, 60 at 1 m, 40 at 2 m, 20 at 5 m and 0 at
   10 m or more, linear in between; a point that could not be triangulated
   scores 0. The sequence's score is the mean over its control points.
   cp_recall_1m: % of control points within 1 m.
4. With --gt: pose_recall_1m / pose_recall_5m, % of the GT poses whose
   horizontal distance to the aligned estimate is within 1 m / 5 m.

Prints one JSON object. If the alignment fails, all scores are 0 and
"error" says why; an error inside the evaluator exits 1 with "error".
"""

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.environ.get("LAMARIA_SRC", "/opt/lamaria-eval/src"))
# An import failure is a broken install, not a bad trajectory: let it crash.
from lamaria.eval.pgt_evaluation import evaluate_wrt_pgt  # noqa: E402
from lamaria.eval.sparse_evaluation import evaluate_wrt_control_points  # noqa: E402
from lamaria.structs.control_point import load_cp_json, run_control_point_triangulation  # noqa: E402
from lamaria.structs.trajectory import Trajectory  # noqa: E402
from lamaria.utils.aria import initialize_reconstruction_from_calibration_file  # noqa: E402
from lamaria.utils.metrics import (  # noqa: E402
    calculate_control_point_recall,
    calculate_control_point_score,
    calculate_pose_recall,
)


def score(estimate, cp_json, calibration, gt=None):
    traj = Trajectory.load_from_file(estimate, invert_poses=False, corresponding_sensor="imu")
    if not traj.is_loaded():
        raise ValueError("the evaluator could not load the estimate")
    reconstruction = initialize_reconstruction_from_calibration_file(Path(calibration))
    control_points, timestamp_to_images = load_cp_json(Path(cp_json))
    reconstruction = traj.add_estimate_poses_to_reconstruction(reconstruction, timestamp_to_images)
    run_control_point_triangulation(reconstruction, control_points)
    result = evaluate_wrt_control_points(reconstruction, control_points)

    out = {"control_points": len(control_points)}
    if result is None:
        out.update(cp_score=0.0, cp_recall_1m=0.0, error="control-point alignment failed")
        if gt:
            out.update(pose_recall_1m=0.0, pose_recall_5m=0.0)
        return out
    out.update(
        cp_score=float(calculate_control_point_score(result)),
        cp_recall_1m=float(calculate_control_point_recall(result)),
        control_points_triangulated=sum(cp.is_triangulated() for cp in result.cp_summary.values()),
        cp_alignment_scale=float(result.alignment.scale),
    )
    if gt:
        est = Trajectory.load_from_file(estimate, invert_poses=False)
        gt_traj = Trajectory.load_from_file(gt, invert_poses=False)
        num_poses_gt = len(gt_traj)
        error = evaluate_wrt_pgt(est, gt_traj, result.alignment)
        for t in (1, 5):
            out[f"pose_recall_{t}m"] = 0.0 if error is None else float(calculate_pose_recall(error, num_poses_gt, t))
    return out


@contextlib.contextmanager
def quiet():
    """Silences the evaluator: its logging and progress bars, and Ceres' output from C++."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1)
        os.dup2(null.fileno(), 2)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("estimate")
    ap.add_argument("cp_json")
    ap.add_argument("calibration")
    ap.add_argument("--gt", help="ground-truth trajectory for pose recall")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        with quiet():
            out = score(args.estimate, args.cp_json, args.calibration, args.gt)
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        print(json.dumps(out))
        sys.exit(1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
