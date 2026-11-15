"""Evaluate LaMAria trajectories using the official CP scorer and Evo metrics."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("recording", type=Path, help="prepared cam0/data.csv, calibration.json and available labels")
parser.add_argument("estimate", type=Path, help="8 columns: integer ns, XYZ metres, Hamilton XYZW; .gz accepted")
parser.add_argument("output", type=Path)
parser.add_argument("--cp-python", type=Path, help="Python 3.12 environment with official-evaluator-requirements.lock")
parser.add_argument("--no-cp", action="store_true")
parser.add_argument("--public-pgt", type=Path, help="author natural-record pseudo-dense reference in camera0 frame")
parser.add_argument("--public-control-points", type=Path, help="author sparse annotation JSON")
args = parser.parse_args()
recording = args.recording.resolve()
if recording.name == "sequence_3_17" and args.public_pgt is None:
    parser.error("3_17 used a separate frame-converted annotation bind; this public wrapper does not reproduce that private bind")
if args.output.exists():
    parser.error("output already exists; choose a new result path")
spec = importlib.util.spec_from_file_location("archived_benchmark", HERE / "frozen/benchmark_evaluate.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
with tempfile.TemporaryDirectory(prefix="lamaria-evaluation-") as temporary:
    temporary = Path(temporary)
    estimate_path = args.estimate.resolve()
    if estimate_path.suffix == ".gz":
        expanded = temporary / "estimate.txt"
        expanded.write_bytes(gzip.decompress(estimate_path.read_bytes()))
        estimate_path = expanded
    estimate = evaluator.read_trajectory(estimate_path)
    frames = evaluator.cam0_timestamps(recording)
    nearest = evaluator.nearest(estimate[0], frames)
    coverage = float(np.mean(abs(estimate[0][nearest] - frames) <= evaluator.MATCH_NS))
    result = dict(frames_with_pose=coverage, camera_frame_coverage=coverage)
    ground_truth = recording / "groundtruth.txt"
    control_points = args.public_control_points.resolve() if args.public_control_points else recording / "control_points.json"
    if args.public_pgt:
        if not recording.name.startswith("sequence_"):
            parser.error("--public-pgt is only for the documented natural-record camera0 reference convention")
        reference_source = args.public_pgt.resolve()
        gt = evaluator.read_trajectory(reference_source)
        # Convert camera0 reference poses to the right-IMU body frame.
        calibration = json.loads((recording / "calibration.json").read_text())["cam0"]["T_b_s"]
        rotation = Rotation.from_quat(gt[2]) * Rotation.from_quat(calibration["qvec"]).inv()
        gt = (gt[0], gt[1] - rotation.apply(np.array(calibration["tvec"])), rotation.as_quat())
        ground_truth = temporary / "public-reference-imu.txt"
        evaluator.write_trajectory(ground_truth, *gt)
    if ground_truth.exists():
        metrics, _, _ = evaluator.evaluate(evaluator.read_trajectory(ground_truth), estimate)
        result.update(metrics)
    else:
        result.update(ate_rmse_m=None, rot_rmse_deg=None, rpe_10m_rmse_m=None, coverage=None,
                      pose_error_note="No dense ground truth supplied; camera-frame coverage is reported separately.")
    if control_points.exists() and not args.no_cp:
        if args.cp_python is None:
            parser.error("control points exist: provide --cp-python or explicitly choose --no-cp")
        os.environ["LAMARIA_SRC"] = str(HERE / "frozen/official")
        if ground_truth.exists():
            # Recreate the archived evaluator's interpreter layout without editing it.
            (temporary / "venv/bin").mkdir(parents=True)
            launcher = temporary / "venv/bin/python"
            launcher.write_text("#!/bin/sh\nexec " + shlex.quote(str(args.cp_python.absolute())) + ' "$@"\n')
            launcher.chmod(0o755)
            (temporary / "lamaria_score.py").symlink_to(HERE / "frozen/lamaria_score.py")
            evaluator.LAMARIA_EVAL = temporary
            result.update(evaluator.cp_evaluate(estimate, recording, control_points, ground_truth))
        else:
            # This is the unchanged closest-baseline wrapper's CP-only path.
            command = [str(args.cp_python), str(HERE / "frozen/lamaria_score.py"), str(estimate_path),
                       str(control_points), str(recording / "calibration.json")]
            scored = subprocess.run(command, capture_output=True, text=True, timeout=180, check=True)
            result.update(json.loads(scored.stdout))
    else:
        result.update(cp_score=None, cp_note="Explicitly skipped." if args.no_cp else "No control-point annotations supplied.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
