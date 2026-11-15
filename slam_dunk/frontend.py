"""Build a reusable, unfitted Full frontend packet from prepared stereo sensors."""

import json
import os
from pathlib import Path
import shutil
import sys
import time

from .runtime import ENGINE, Runner, sha256
from .stereo import export_packet


def prepare(recording, output, *, seed=None):
    """Run original preparation, Basalt flow, RootSIFT, dual IMU and triangulation.

    An optional unrefined VIO seed replaces the tracking VIO before triangulation,
    as in the paper's matched-packet protocol. Flow is still needed for tracks.
    Every model ablation can then consume the same packet independently.
    """
    recording = Path(recording).resolve()
    for name in (
        "calibration.json",
        "factory_calibration.json",
        "imu0/data.csv",
        "cam0/data.csv",
        "cam0/meta.csv",
        "cam1/data.csv",
        "cam1/meta.csv",
    ):
        if not (recording / name).is_file():
            raise ValueError(f"Missing sensor input: {recording / name}")
    runner = Runner(output)
    work = runner.output / "frontend"
    work.mkdir()
    started = time.monotonic()
    settings = json.loads((ENGINE / "settings.json").read_text())
    settings.update(THREADS="4", DETECT_SCALE=".5", DETECT_FRACTION=".8")
    try:
        _track_and_initialize(recording, work, settings, runner, started, seed)
        step = _match_keyframes(recording, work, settings, runner)
        _fuse_secondary_imu(recording, work, settings, runner)
        packet = _export_packet(recording, work, settings, runner, step)
        runner.receipt.update(status="complete", keyframe_step_seconds=float(step))
        return packet
    except BaseException as error:
        runner.receipt.update(status="failed", error=repr(error))
        raise
    finally:
        runner.receipt["seconds"] = time.monotonic() - started
        runner.flush()


def _track_and_initialize(recording, work, settings, runner, started, seed):
    runner.checked(
        [sys.executable, ENGINE / "prepare.py", recording, work],
        60,
        settings,
        runner.output / "prepare.log",
    )
    if seed is None:
        runner.checked(
            [sys.executable, "-m", "slam_dunk.vio", work, str(time.monotonic() - started)],
            1405,
            settings,
            runner.output / "flow.log",
        )
    else:
        runner.checked(
            [ENGINE / "native/build/aria_vio", work, work / "vio.txt"],
            600,
            settings,
            runner.output / "flow.log",
        )
    if not (work / "tracks.bin").is_file():
        raise RuntimeError("Flow/VIO did not produce tracks")
    if seed is not None:
        seed = Path(seed).resolve()
        shutil.copyfile(work / "vio.txt", runner.output / "tracking_vio.txt")
        shutil.copyfile(seed, work / "vio.txt")
        runner.receipt["seed_sha256"] = sha256(seed)


def _match_keyframes(recording, work, settings, runner):
    frames = (work / "frames.txt").read_text().splitlines()
    duration = (int(frames[-1].split()[0]) - int(frames[0].split()[0])) * 1e-9
    step = str(max(0.6, duration / 1500.0))
    features = dict(
        settings,
        SIFT_FEATURES="900",
        SIFT_CONTRAST=".006",
        SIFT_DARK_ADAPT="0",
        PRECISE_UPSCALE="0",
        ROOT_SIFT="1",
        TRACK_CONSISTENCY="1",
        MATCH_MAX_DISTANCE=".7",
        BA_STEP=step,
        SIFT_THREADS="8",
        SIFT_BUDGET="498",
    )
    runner.checked(
        [sys.executable, ENGINE / "sift_tracks.py", recording, work],
        500,
        features,
        runner.output / "sift.log",
    )
    return step


def _fuse_secondary_imu(recording, work, settings, runner):
    if (recording / "imu1/data.csv").is_file():
        runner.checked(
            [sys.executable, ENGINE / "dual_imu.py", recording, work],
            45,
            settings,
            runner.output / "dual.log",
        )
        if (work / "imu_fused.bin").stat().st_size != (work / "imu.bin").stat().st_size:
            raise RuntimeError("Dual IMU sample count changed")
        os.replace(work / "imu_fused.bin", work / "imu.bin")
        runner.receipt["imu_domain"] = "virtual_dual"
    else:
        runner.receipt["imu_domain"] = "primary_only"


def _export_packet(recording, work, settings, runner, step):
    flags = dict(
        settings,
        BA_STEP=step,
        TRACKS_FILE="tracks_ba.bin",
        MIN_OBS="3",
        TRI_ERR="5",
        SAVE_META="1",
        SAVE_CANDIDATES="1",
        SIFT_PIXEL_BIAS="0",
        MIX_FLOW="1",
        FLOW_PIXEL_BIAS=".25",
    )
    runner.checked(
        [sys.executable, ENGINE / "export_tracks.py", recording, work],
        35,
        flags,
        runner.output / "export.log",
    )
    return export_packet(recording, work, runner.output / "work")
