"""Recover the current recording's complete two-pass optimizer state on failure."""

import os
import shutil
from pathlib import Path

SUFFIXES = (
    ".vel",
    ".gravity",
    ".timing",
    ".gyromat",
    ".gyrot",
    ".accalign",
    ".accscale",
    ".diag",
    ".intr",
    ".pls",
    "",
)


def restore_base(work):
    work = Path(work)
    if not (work / "ba.txt.base").exists():
        return False
    for suffix in SUFFIXES:
        source = work / ("ba.txt.base" + suffix)
        target = work / ("ba.txt" + suffix)
        if source.exists():
            temporary = Path(str(target) + ".restore")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
    return True
