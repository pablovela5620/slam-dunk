"""Fisheye features, with detector sensitivity adjusted for underexposure."""

import numpy as np
import cv2
import os
from pathlib import Path


def extract_image(path, camera, local, nfeatures, contrast, precise=False, dark_adapt=False):
    local.feature_scales = np.zeros(0, np.float32)
    empty = (np.zeros((0, 2)), np.zeros((0, 128), np.float32), np.zeros((0, 3)))
    im = cv2.imread(str(path), 0)
    if im is None:
        return empty
    # A lower contrast threshold recovers structure in underexposed images.
    threshold = (0.006 if im.mean() < 35 else 0.025) if dark_adapt else contrast
    if not hasattr(local, "detectors"):
        local.detectors = {}
    key = (nfeatures, threshold, precise)
    if key not in local.detectors:
        local.detectors[key] = cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=threshold,
            edgeThreshold=12,
            enable_precise_upscale=precise,
        )
    kp, desc = local.detectors[key].detectAndCompute(im, None)
    if desc is None:
        return empty
    xy = np.array([p.pt for p in kp])
    rr = camera.cam_from_img(xy)
    rays = np.c_[rr, np.ones(len(rr))]
    norm = np.linalg.norm(rays, axis=1)
    rays /= norm[:, None]
    valid = np.isfinite(rays).all(axis=1) & (norm < 15)
    local.feature_scales = np.array([p.size for p in kp], np.float32)[valid]
    # Optional measurement diagnostics; these do not change the extracted tracks.
    if os.environ.get("SAVE_STATE") and os.environ.get("ARIA_WORK_DIR"):
        folder = Path(os.environ["ARIA_WORK_DIR"]) / "qualities"
        folder.mkdir(exist_ok=True)
        name = Path(path).stem + "_" + Path(path).parent.parent.name + ".npy"
        quality = np.array([[p.size, p.response, p.angle, p.octave] for p in kp], np.float32)
        np.save(folder / name, np.c_[xy[valid], quality[valid]])
    return xy[valid], desc[valid], rays[valid]


def root_descriptors_inplace(features):
    """Normalize completed, recording-local descriptor arrays without a second copy."""
    for _, descriptors, _ in features:
        denominator = np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-10)
        np.divide(descriptors, denominator, out=descriptors)
        np.sqrt(descriptors, out=descriptors)
    return features
