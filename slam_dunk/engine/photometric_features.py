"""Recording-local camera gain for a conservative optional feature budget."""

from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=4)
def _gain_metadata(camera_folder):
    try:
        rows = np.loadtxt(Path(camera_folder) / "meta.csv", delimiter=",", usecols=(0, 2), ndmin=2)
        if not len(rows) or not np.isfinite(rows).all():
            return None
        if np.any(np.diff(rows[:, 0]) <= 0) or np.any(rows[:, 1] <= 0):
            return None
        return rows[:, 0], rows[:, 1]
    except (OSError, ValueError):
        return None


def gain_at_image(image_path):
    path = Path(image_path)
    data = _gain_metadata(str(path.parent.parent))
    if data is None:
        return None
    try:
        stamp = int(path.stem)
    except ValueError:
        return None
    times, gains = data
    i = int(np.searchsorted(times, stamp))
    i = min(i, len(times) - 1)
    if i and abs(times[i - 1] - stamp) < abs(times[i] - stamp):
        i -= 1
    # Missing metadata must not silently borrow a distant exposure regime.
    if abs(times[i] - stamp) > 100_000_000:
        return None
    return float(gains[i])
