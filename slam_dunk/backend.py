"""Full native optimization, original recovery policy, and dense export."""

import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .config import Config
from .runtime import ENGINE, Runner, sha256
from .trajectory import validate_trajectory
from .stereo import require_stereo_packet

PACKET_FILES = ("imu.bin", "vio.txt", "calib.json", "calibration.json", "export.npz",
                "frame_times.csv")


def packet_files(packet):
    packet = Path(packet).resolve()
    require_stereo_packet(packet)
    required = ("input.bin", "input.bin.meta") + PACKET_FILES
    for name in required:
        if not (packet / name).is_file():
            raise ValueError(f"Missing cold frontend packet input: {packet / name}")
    return sorted(set(packet.glob("input.bin*")) | {packet / name for name in PACKET_FILES})


def _engine_module(name):
    spec = importlib.util.spec_from_file_location("_slam_dunk_" + name, ENGINE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_backend(packet, output, config=None, *, binary=None):
    """Refine one cold frontend packet with a fixed 1,200-second native budget.

    Each call copies unfitted inputs and starts from the packet's VIO seed.
    Fitted sidecars are never copied. A new output directory is required.
    """
    config = Config() if config is None else config
    files = packet_files(packet)
    binary = Path(binary or ENGINE / "native/build/batch_joint").resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"Native backend not built: {binary}; run build_native.sh")
    runner = Runner(output, config.to_dict())
    work = runner.output / "work"
    work.mkdir()
    started = time.monotonic()
    try:
        hashes = _copy_packet(files, packet, work)
        restored = _solve(work, binary, config, runner, hashes)
        _export_trajectory(work, runner)
        return _validate_result(files, hashes, work, runner, restored)
    except BaseException as error:
        runner.receipt.update(status="failed", error=repr(error))
        raise
    finally:
        runner.receipt["seconds"] = time.monotonic() - started
        runner.flush()


def _copy_packet(files, packet, work):
    hashes = {p.name: sha256(p) for p in files}
    for source in files:
        shutil.copyfile(source, work / source.name)
    return hashes


def _solve(work, binary, config, runner, hashes):
    env = config.native_environment()
    runner.receipt.update(input_hashes=hashes, binary_sha256=sha256(binary), native_environment=env)
    runner.flush()
    recovery = _engine_module("iteration_recovery")
    final = _engine_module("final_refinement")
    rc, timed_out, report = recovery.run_with_iteration_recovery(
        runner.call,
        [str(binary), str(work / "input.bin"), str(work / "ba.txt")],
        1200.0,
        env,
        work / "ba.log",
    )
    restored = False
    if timed_out or rc:
        restored = final.restore_base(work)
    runner.receipt.update(
        native_exit=rc, native_timeout=timed_out, iteration_recovery=report, restored_base=restored
    )
    if not (work / "ba.txt").is_file():
        raise RuntimeError("Native backend produced no endpoint")
    return restored


def _export_trajectory(work, runner):
    final = _engine_module("final_refinement")
    settings = json.loads((ENGINE / "settings.json").read_text())
    command = [sys.executable, ENGINE / "fuse.py", work, work, runner.output / "trajectory.txt"]
    try:
        export_rc = runner.call(command, 30, settings, runner.output / "export.log")
    except subprocess.TimeoutExpired:
        export_rc = 124
    if export_rc and final.restore_base(work):
        runner.receipt["dense_reconstruction_fallback"] = True
        export_rc = runner.call(command, 30, settings, runner.output / "export-fallback.log")
    if export_rc:
        raise RuntimeError(f"Dense export failed with exit {export_rc}")


def _validate_result(files, hashes, work, runner, restored):
    runner.receipt["coverage"] = validate_trajectory(
        work / "frame_times.csv", runner.output / "trajectory.txt"
    )
    runner.receipt["inputs_unchanged"] = all(sha256(p) == hashes[p.name] for p in files)
    if not runner.receipt["inputs_unchanged"]:
        raise RuntimeError("Source packet changed during inference")
    text = (work / "ba.log").read_text(errors="replace")
    runner.receipt["stop_reasons"] = re.findall(r"^STOP_REASON (.*)$", text, re.M)
    runner.receipt["final_selections"] = re.findall(r"^FINAL_SELECTION (.*)$", text, re.M)
    runner.receipt["status"] = "restored_base" if restored else "complete"
    return runner.output / "trajectory.txt"
