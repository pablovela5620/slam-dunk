"""Admit pixels only in this recording camera model's supported domain."""

import json, hashlib
from pathlib import Path
import numpy as np


def supported_pixels(src, calibration, tracks):
    from projectaria_tools.core import calibration as aria

    src = Path(src)
    factory_text = (src / "factory_calibration.json").read_text()
    device = aria.device_calibration_from_json_string(factory_text)
    if device is None:
        raise ValueError("factory camera calibration unavailable")
    good = np.ones(len(tracks), bool)
    report = {
        "method": "current_factory_SDK_unproject_project_domain",
        "factory_sha256": hashlib.sha256(factory_text.encode()).hexdigest(),
        "cameras": [],
    }
    for c, label in enumerate(["camera-slam-left", "camera-slam-right"]):
        camera = device.get_camera_calib(label)
        k = calibration[f"cam{c}"]
        rows = np.flatnonzero(tracks[:, 1] == c)
        entry = {"camera": c, "observations": len(rows)}
        if camera is None:
            entry.update(applied=False, reason="camera_missing")
            report["cameras"].append(entry)
            continue
        expected = np.array([k["resolution"]["width"], k["resolution"]["height"]])
        params = np.asarray(k["params"])
        factory_params = np.asarray(camera.get_projection_params())
        matching = (
            params.size == 16
            and abs(params[0] - params[1]) < 1e-10
            and factory_params.shape == (15,)
            and np.max(abs(factory_params - np.r_[params[0], params[2:]])) < 1e-8
            and np.array_equal(camera.get_image_size(), expected)
        )
        if not matching:
            entry.update(applied=False, reason="provided_model_differs_from_factory")
            report["cameras"].append(entry)
            continue
        local = np.empty(len(rows), bool)
        for j, i in enumerate(rows):
            ray = camera.unproject(tracks[i, 3:5])
            local[j] = ray is not None and camera.project(ray) is not None
        good[rows] = local
        entry.update(
            applied=True,
            rejected=int((~local).sum()),
            valid_radius=camera.get_valid_radius(),
            max_solid_angle=camera.get_max_solid_angle(),
        )
        report["cameras"].append(entry)
    report["retained"] = int(good.sum())
    report["rejected"] = int((~good).sum())
    return good, report
