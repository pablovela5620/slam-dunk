"""Interpolate optional exposure and temperature without propagating missing values."""

from pathlib import Path
import numpy as np


def metadata_at(path, timestamps):
    exposure = np.full(len(timestamps), 0.005)
    temperature = np.zeros(len(timestamps))
    try:
        a = np.loadtxt(Path(path), delimiter=",", ndmin=2)
        a = a[np.isfinite(a[:, 0])]
        if not len(a):
            return exposure, temperature
        if np.any(np.diff(a[:, 0]) < 0):
            a = a[np.argsort(a[:, 0], kind="stable")]
        good = np.isfinite(a[:, 1]) & (a[:, 1] > 0)
        if good.any():
            exposure = np.interp(timestamps, a[good, 0], a[good, 1])
        if a.shape[1] > 3:
            good = np.isfinite(a[:, 3])
            if good.any():
                temperature = np.interp(timestamps, a[good, 0], a[good, 3])
    except (OSError, ValueError, IndexError):
        pass
    return exposure, temperature
