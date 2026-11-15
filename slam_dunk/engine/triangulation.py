"""Robust recording-local ray intersection for controlled geometry ablations."""

import os
import numpy as np


def refine_tracks(points, origins, rays, matrices, rhs, point_index, first, count, mode):
    minimum = int(os.environ.get("MIN_OBS", "3"))
    threshold = float(os.environ.get("TRI_ERR", "5"))
    points = points.copy()
    active = None
    for iteration in range(5):
        delta = points[point_index] - origins
        depth = np.einsum("ni,ni->n", delta, rays)
        error = 240 * np.linalg.norm(np.cross(delta, rays), axis=1) / np.maximum(depth, 1e-8)
        if mode == "refit":
            weight = (
                (depth > 0.3) & (depth < 250) & np.isfinite(error) & (error < threshold)
            ).astype(float)
        elif mode == "prune":
            inlier = (depth > 0.3) & (depth < 250) & np.isfinite(error) & (error < threshold)
            active = inlier if active is None else active & inlier
            weight = active.astype(float)
        elif mode == "angular":
            weight = 1 / np.maximum(np.abs(depth), 0.5) ** 2
            weight *= 1 / (1 + (error / 2.0) ** 2)
            weight[~np.isfinite(weight)] = 0
        else:
            raise ValueError("unknown triangulation mode: " + mode)
        maximum = np.maximum.reduceat(weight, first)
        weight /= np.maximum(maximum[point_index], 1e-30)
        normal = np.add.reduceat(matrices * weight[:, None, None], first, axis=0)
        value = np.add.reduceat(rhs * weight[:, None], first, axis=0)
        try:
            candidate = np.linalg.solve(normal + np.eye(3)[None] * 1e-12, value[..., None])[..., 0]
        except np.linalg.LinAlgError:
            candidate = np.einsum("nij,nj->ni", np.linalg.pinv(normal), value)
        enough = np.add.reduceat((weight > 1e-10).astype(int), first) >= minimum
        enough &= np.isfinite(candidate).all(axis=1)
        points[enough] = candidate[enough]
    delta = points[point_index] - origins
    depth = np.einsum("ni,ni->n", delta, rays)
    error = 240 * np.linalg.norm(np.cross(delta, rays), axis=1) / np.maximum(depth, 1e-8)
    valid = (depth > 0.3) & (depth < 250) & np.isfinite(error) & (error < threshold)
    if mode == "prune":
        valid &= active
    counts = np.add.reduceat(valid.astype(int), first)
    good = (
        (counts >= minimum)
        & np.isfinite(points).all(axis=1)
        & (np.linalg.norm(origins[first] - origins[first + count - 1], axis=1) > 0.1)
    )
    return points, depth, error, valid, good
