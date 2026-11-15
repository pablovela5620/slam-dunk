"""Intrinsic-only recovery for gross VIO startup failure; native code is unchanged."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

POLICY = {
    "version": 1,
    "p99_speed_limit_mps": 50.0,
    "minimum_span_fraction": 0.995,
    "minimum_state_fraction": 0.99,
    "retry_start_seconds": [1.0, 5.0],
    "downstream_reserve_seconds": 300.0,
    "maximum_attempt_seconds": 600.0,
}


def health(path, timestamps, stride, returncode=0, error=None):
    result = {"passed": False, "reasons": [], "returncode": returncode, "error": error}
    if returncode != 0 or error:
        result["reasons"].append("native_call_failed")
    try:
        poses = np.atleast_2d(np.loadtxt(path))
        if poses.shape[1] != 8 or len(poses) < 2 or not np.isfinite(poses).all():
            raise ValueError("missing_or_nonfinite_pose_rows")
        delta = np.diff(poses[:, 0]) * 1e-9
        if not np.all(delta > 0):
            raise ValueError("nonmonotonic_pose_timestamps")
        expected = np.asarray(timestamps[::stride], dtype=np.int64)
        if not np.isin(poses[:, 0], expected).all():
            result["reasons"].append("unexpected_pose_timestamps")
        span = (poses[-1, 0] - poses[0, 0]) / max(1, int(expected[-1] - expected[0]))
        count_fraction = len(poses) / max(1, len(expected))
        result.update(
            states=len(poses),
            expected_states=len(expected),
            span_fraction=float(span),
            state_fraction=count_fraction,
            first_timestamp_ns=int(poses[0, 0]),
            last_timestamp_ns=int(poses[-1, 0]),
        )
        if (
            span < POLICY["minimum_span_fraction"]
            or count_fraction < POLICY["minimum_state_fraction"]
        ):
            result["reasons"].append("incomplete_state_coverage")
        speeds = np.linalg.norm(np.diff(poses[:, 1:4], axis=0), axis=1) / delta
        if not np.isfinite(speeds).all():
            raise ValueError("nonfinite_velocity")
        p99 = float(np.quantile(speeds, 0.99))
        result.update(
            p99_speed_mps=p99,
            maximum_speed_mps=float(speeds.max()),
            position_extent_m=np.ptp(poses[:, 1:4], axis=0).tolist(),
        )
        if p99 > POLICY["p99_speed_limit_mps"]:
            result["reasons"].append("grossly_implausible_speed")
        if not np.allclose(np.linalg.norm(poses[:, 4:8], axis=1), 1, atol=1e-3):
            result["reasons"].append("invalid_quaternion_norm")
    except (OSError, ValueError, IndexError) as exception:
        result["reasons"].append(str(exception))
    result["passed"] = not result["reasons"]
    return result


def recover(
    work, root, run, env, started, deadline, primary_returncode, primary_seconds, primary_error=None
):
    work, root = Path(work), Path(root)
    frames = work / "frames.txt"
    original = frames.read_bytes()
    rows = original.decode().splitlines(keepends=True)
    timestamps = np.asarray([int(row.split()[0]) for row in rows], dtype=np.int64)
    stride = max(1, int(float(env.get("STRIDE", math.ceil(len(rows) / 12000)))))
    forced = os.environ.get("ARIA_DEV_FORCE_STARTUP_RECOVERY", "0") == "1"
    dense_enabled = os.environ.get("ARIA_DENSE_STARTUP", "0") == "1"
    primary = health(work / "vio.txt", timestamps, stride, primary_returncode, primary_error)
    report = {
        "policy": POLICY,
        "development_force": forced,
        "original_stride": stride,
        "original_frames": len(rows),
        "original_frames_sha256": hashlib.sha256(original).hexdigest(),
        "triggered": forced or not primary["passed"],
        "selected_attempt": "primary" if primary["passed"] else None,
        "attempts": [
            {"name": "primary", "skip_seconds": 0.0, "seconds": primary_seconds, "health": primary}
        ],
    }

    report["dense_retry_policy"] = {
        "version": 1,
        "enabled": dense_enabled,
        "frame_stride": 1,
        "start_seconds": 0.0,
        "trigger": "all original-cadence startup attempts intrinsically unhealthy",
        "maximum_attempt_seconds": POLICY["maximum_attempt_seconds"],
        "shared_recording_deadline": True,
    }

    def save():
        report["full_frames_restored"] = frames.read_bytes() == original
        report["pipeline_elapsed_seconds"] = time.monotonic() - started
        target = work / "recovery.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)

    def preserve(label):
        directory = work / "startup_recovery_attempts" / label
        directory.mkdir(parents=True, exist_ok=True)
        for name in ["vio.txt", "tracks.bin", "vio.log"]:
            if (work / name).exists():
                shutil.move(work / name, directory / name)
        return directory

    if not report["triggered"]:
        save()
        return report
    backup = preserve("primary")
    report["selected_attempt"] = None
    try:
        for skip in POLICY["retry_start_seconds"]:
            offset = int(np.searchsorted(timestamps, timestamps[0] + round(skip * 1e9)))
            offset = ((offset + stride - 1) // stride) * stride
            remaining = (
                deadline - POLICY["downstream_reserve_seconds"] - (time.monotonic() - started)
            )
            if offset >= len(rows) - stride or remaining <= 5:
                report["retry_stop_reason"] = "insufficient_frames_or_deadline_budget"
                break
            label = "retry_" + str(int(skip)) + "s"
            attempt = {
                "name": label,
                "requested_skip_seconds": skip,
                "first_frame_index": offset,
                "skip_seconds": float((timestamps[offset] - timestamps[0]) * 1e-9),
                "stride": stride,
                "timeout_seconds": min(POLICY["maximum_attempt_seconds"], remaining),
            }
            call_started = time.monotonic()
            rc, error = None, None
            try:
                frames.write_bytes("".join(rows[offset:]).encode())
                rc = run(
                    [root / "native/build/aria_vio", work, work / "vio.txt"],
                    timeout=attempt["timeout_seconds"],
                    extra={"THREADS": "4", "STRIDE": str(stride)},
                    log=work / "vio.log",
                )
            except (subprocess.TimeoutExpired, OSError) as exception:
                error = type(exception).__name__ + ": " + str(exception)
            finally:
                frames.write_bytes(original)
            attempt.update(
                seconds=time.monotonic() - call_started,
                health=health(work / "vio.txt", timestamps[offset:], stride, rc, error),
                full_frames_restored=frames.read_bytes() == original,
            )
            report["attempts"].append(attempt)
            if attempt["health"]["passed"]:
                report["selected_attempt"] = label
                break
            preserve(label)
        if report["selected_attempt"] is None and dense_enabled and stride > 1:
            remaining = (
                deadline - POLICY["downstream_reserve_seconds"] - (time.monotonic() - started)
            )
            if remaining > 5:
                label = "dense_primary"
                attempt = {
                    "name": label,
                    "requested_skip_seconds": 0.0,
                    "first_frame_index": 0,
                    "skip_seconds": 0.0,
                    "stride": 1,
                    "timeout_seconds": min(POLICY["maximum_attempt_seconds"], remaining),
                }
                call_started = time.monotonic()
                rc, error = None, None
                try:
                    frames.write_bytes(original)
                    rc = run(
                        [root / "native/build/aria_vio", work, work / "vio.txt"],
                        timeout=attempt["timeout_seconds"],
                        extra={"THREADS": "4", "STRIDE": "1"},
                        log=work / "vio.log",
                    )
                except (subprocess.TimeoutExpired, OSError) as exception:
                    error = type(exception).__name__ + ": " + str(exception)
                finally:
                    frames.write_bytes(original)
                attempt.update(
                    seconds=time.monotonic() - call_started,
                    health=health(work / "vio.txt", timestamps, 1, rc, error),
                    full_frames_restored=frames.read_bytes() == original,
                )
                report["attempts"].append(attempt)
                if attempt["health"]["passed"]:
                    report["selected_attempt"] = label
                else:
                    preserve(label)
            else:
                report["dense_retry_stop_reason"] = "insufficient_shared_deadline_budget"
        if report["selected_attempt"] is None and primary["passed"]:
            # Development-only forced retries must not discard a valid primary.
            for name in ["vio.txt", "tracks.bin", "vio.log"]:
                if (backup / name).exists():
                    shutil.copy2(backup / name, work / name)
            report["selected_attempt"] = "primary_after_forced_retry_failure"
    finally:
        frames.write_bytes(original)
        save()
    print(
        "Startup recovery:",
        json.dumps(
            {
                "selected_attempt": report["selected_attempt"],
                "development_force": forced,
                "attempts": len(report["attempts"]),
            }
        ),
        flush=True,
    )
    return report
