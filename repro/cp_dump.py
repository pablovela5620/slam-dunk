"""Re-run the official LaMAria CP evaluation and dump its geometry for visualization.

Same steps as evaluation/frozen/lamaria_score.py::score (official evaluator at the
frozen commit), plus the Sim(3) alignment, surveyed (topo) and triangulated control
points, and per-point horizontal errors. Run in the `cp-eval` pixi environment:

    pixi run -e cp-eval python repro/cp_dump.py EST CP_JSON CALIB OUT.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation/frozen/official"))
sys.path.insert(0, str(REPO / "evaluation/frozen"))

import numpy as np  # noqa: E402
from lamaria.eval.sparse_evaluation import evaluate_wrt_control_points  # noqa: E402
from lamaria.structs.control_point import load_cp_json, run_control_point_triangulation  # noqa: E402
from lamaria.structs.trajectory import Trajectory  # noqa: E402
from lamaria.utils.aria import initialize_reconstruction_from_calibration_file  # noqa: E402
from lamaria.utils.metrics import calculate_control_point_score, calculate_error  # noqa: E402
from lamaria_score import quiet  # noqa: E402


def dump(estimate: Path, cp_json: Path, calibration: Path) -> dict:
    """Return the CP score plus the alignment and per-point geometry."""
    traj = Trajectory.load_from_file(estimate, invert_poses=False, corresponding_sensor="imu")
    reconstruction = initialize_reconstruction_from_calibration_file(calibration)
    control_points, timestamp_to_images = load_cp_json(cp_json)
    reconstruction = traj.add_estimate_poses_to_reconstruction(reconstruction, timestamp_to_images)
    run_control_point_triangulation(reconstruction, control_points)
    result = evaluate_wrt_control_points(reconstruction, control_points)
    if result is None:
        return {"cp_score": 0.0, "error": "alignment failed"}
    errors = np.asarray(calculate_error(result), dtype=float)
    a = result.alignment
    points = []
    for (tag, cp), err in zip(result.cp_summary.items(), errors):
        points.append(
            dict(
                tag=int(tag),
                name=cp.name,
                topo=list(map(float, cp.topo)),
                triangulated=None if cp.triangulated is None else list(map(float, cp.triangulated)),
                error_2d_m=None if not np.isfinite(err) else float(err),
            )
        )
    return dict(
        cp_score=float(calculate_control_point_score(result)),
        # topo = scale * R * est + t
        alignment=dict(
            scale=float(a.scale),
            quat_xyzw=list(map(float, a.rotation.quat)),
            translation=list(map(float, a.translation)),
        ),
        points=points,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("estimate", type=Path)
    ap.add_argument("cp_json", type=Path)
    ap.add_argument("calibration", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    with quiet():
        out = dump(args.estimate, args.cp_json, args.calibration)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("cp_score",)}))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
