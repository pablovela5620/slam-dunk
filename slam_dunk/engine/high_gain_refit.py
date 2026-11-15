"""Conservative ray refit for uniformly high-gain recording-local tracks."""

import numpy as np
import json
from pathlib import Path
from photometric_features import gain_at_image
from triangulation import refine_tracks


def refit_high_gain(
    src,
    times,
    tracks,
    state_index,
    points,
    origins,
    rays,
    matrices,
    rhs,
    point_index,
    first,
    count,
    depth,
    error,
    valid,
    good,
    report_path=None,
):
    gain = np.full((len(times), 2), np.nan)
    for i, stamp in enumerate(times):
        for camera in (0, 1):
            value = gain_at_image(src / f"cam{camera}" / "data" / f"{int(stamp)}.jpg")
            if value is not None:
                gain[i, camera] = value
    per_observation = gain[state_index, tracks[:, 1].astype(int)]
    eligible = np.minimum.reduceat(per_observation, first) > 4.0
    report = {
        "method": "five_accepted_ray_refits_only_when_every_view_gain_gt4",
        "eligible_tracks": int(eligible.sum()),
        "all_tracks": len(points),
        "original_valid_observations": int(valid.sum()),
        "original_good_points": int(good.sum()),
    }
    print(
        "HIGH_GAIN_REFIT eligible",
        int(eligible.sum()),
        "of",
        len(points),
        "all_views_gain_gt4",
        flush=True,
    )
    if not eligible.any():
        report.update(unchanged=True)
        if report_path is not None:
            Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
        return points, depth, error, valid, good
    result = refine_tracks(points, origins, rays, matrices, rhs, point_index, first, count, "refit")
    new_points, new_depth, new_error, new_valid, new_good = result
    shift = np.linalg.norm(new_points[eligible] - points[eligible], axis=1)
    report.update(
        unchanged=False,
        seed_shift_m_quantiles=np.quantile(shift, [0, 0.5, 0.9, 0.99, 1]).tolist(),
        refit_valid_observations=int(np.where(eligible[point_index], new_valid, valid).sum()),
        refit_good_points=int(np.where(eligible, new_good, good).sum()),
        formerly_rejected_now_good=int((eligible & ~good & new_good).sum()),
        formerly_good_now_rejected=int((eligible & good & ~new_good).sum()),
    )
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    observation_mask = eligible[point_index]
    # Mixed/ordinary-gain tracks preserve the original complete export state.
    points = points.copy()
    points[eligible] = new_points[eligible]
    depth = np.where(observation_mask, new_depth, depth)
    error = np.where(observation_mask, new_error, error)
    valid = np.where(observation_mask, new_valid, valid)
    good = np.where(eligible, new_good, good)
    return points, depth, error, valid, good
