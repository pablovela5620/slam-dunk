"""Budget-bounded, recording-local retry of an unsettled joint solve.

The decision uses native stopping diagnostics only. It never reads recording
identifiers, reference trajectories, CP/GPS data or scores. Caller supplies the
current executable, unchanged sensor inputs and complete native environment.
"""

from pathlib import Path
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

POLICY = {
    "version": 3,
    "trigger_pass": "terminal_selected",
    "primary_iteration_cap": 250,
    "retry_iteration_cap": 2000,
    "minimum_unsettled_pose_step_m": 0.05,
    "minimum_retry_seconds": 10.0,
    "validation_reserve_seconds": 3.0,
}
STATE_SUFFIXES = (
    "",
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
    ".focals",
    ".points",
    ".observations",
)
FAMILY_SUFFIXES = STATE_SUFFIXES + tuple(".base" + s for s in STATE_SUFFIXES)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def parse_solver_log(text):
    passes = []
    pending = None
    reason = None
    reports = 0
    malformed = False
    for line in text.splitlines():
        if line.startswith("POSE_STABILITY "):
            if pending is not None:
                malformed = True
            try:
                values = list(map(float, line.split()[1:]))
                if len(values) != 3 or not all(math.isfinite(x) and x >= 0 for x in values):
                    raise ValueError()
                pending = values
            except ValueError:
                pending = None
                malformed = True
        elif line.startswith("STOP_REASON "):
            reason = line.removeprefix("STOP_REASON ")
        elif line.startswith("Ceres Solver Report: "):
            reports += 1
            m = re.search(r"Iterations:\s*(\d+).*Termination:\s*([A-Z_]+)", line)
            if m is None or pending is None or reason is None:
                malformed = True
            else:
                passes.append(
                    {
                        "iterations": int(m[1]),
                        "termination": m[2],
                        "max_pose_step_m": pending[0],
                        "mean_pose_step_m": pending[1],
                        "max_rotation_step_rad": pending[2],
                        "stop_reason": reason,
                    }
                )
            pending = None
            reason = None
    complete = (
        bool(passes)
        and not malformed
        and pending is None
        and reason is None
        and reports == len(passes)
    )
    return {"passes": passes, "complete": complete, "report_count": reports}


def parameter_trigger(parsed):
    if not parsed["complete"]:
        return False
    p = parsed["passes"][-1]
    return (
        p["termination"] == "CONVERGENCE"
        and "parameter tolerance reached" in p["stop_reason"].lower()
        and p["max_pose_step_m"] > POLICY["minimum_unsettled_pose_step_m"]
    )


def iteration_trigger(parsed, cap=250):
    if not parsed["complete"]:
        return False
    # Only the selected terminal pass can trigger. An early ceiling followed
    # by a settled later pass is a healthy primary, which stays unchanged.
    return parameter_trigger(parsed) or any(
        p["termination"] == "NO_CONVERGENCE"
        and p["iterations"] >= cap
        and "maximum number of iterations reached" in p["stop_reason"].lower()
        and p["max_pose_step_m"] > POLICY["minimum_unsettled_pose_step_m"]
        for p in parsed["passes"][-1:]
    )


def retry_converged(parsed):
    return (
        parsed["complete"]
        and all(p["termination"] in ["CONVERGENCE", "USER_SUCCESS"] for p in parsed["passes"])
        and parsed["passes"][-1]["max_pose_step_m"] <= POLICY["minimum_unsettled_pose_step_m"]
    )


def input_hashes(path):
    path = Path(path)
    return {p.name: sha(p) for p in sorted(path.parent.glob(path.name + "*")) if p.is_file()}


def snapshot_family(prefix):
    prefix = Path(prefix)
    backup = Path(tempfile.mkdtemp(prefix="optimizer-primary-", dir=prefix.parent))
    present = {}
    for suffix in FAMILY_SUFFIXES:
        source = Path(str(prefix) + suffix)
        if source.exists():
            if not source.is_file() or source.is_symlink():
                raise ValueError("nonregular optimizer state")
            target = backup / source.name
            shutil.copyfile(source, target)
            present[suffix] = sha(target)
    if "" not in present:
        raise ValueError("no complete primary pose file")
    return backup, present


def restore_family(prefix, backup, present):
    prefix = Path(prefix)
    for suffix in FAMILY_SUFFIXES:
        target = Path(str(prefix) + suffix)
        if suffix in present:
            temp = Path(str(target) + ".restore-tmp")
            shutil.copyfile(backup / target.name, temp)
            os.replace(temp, target)
            if sha(target) != present[suffix]:
                raise RuntimeError("primary state restoration failed")
        elif target.exists():
            if not target.is_file() or target.is_symlink():
                raise RuntimeError("nonregular new optimizer state")
            target.unlink()


def validate_selected_state(input_path, prefix, present, backup):
    input_path, prefix = Path(input_path), Path(prefix)
    with input_path.open("rb") as f:
        header = np.fromfile(f, np.int64, count=4)
        f.seek(400)
        times = np.fromfile(f, np.float64, count=int(header[0]))
    expected = np.rint(times * 1e9).astype(np.int64)
    poses = np.loadtxt(prefix, ndmin=2)
    velocity = np.loadtxt(str(prefix) + ".vel", ndmin=2)
    checks = {
        "complete_pose_grid": poses.shape == (len(expected), 8),
        "complete_velocity_grid": velocity.shape == (len(expected), 4),
    }
    if not all(checks.values()):
        return {"passed": False, "checks": checks}
    checks.update(
        exact_pose_times=bool(np.array_equal(poses[:, 0], expected)),
        exact_velocity_times=bool(np.array_equal(velocity[:, 0], expected)),
        finite_poses=bool(np.isfinite(poses).all()),
        finite_velocities=bool(np.isfinite(velocity).all()),
        unit_quaternions=bool(np.all(abs(np.linalg.norm(poses[:, 4:], axis=1) - 1) < 1e-5)),
    )
    for suffix in present:
        if suffix in ["", ".vel"]:
            continue
        target = Path(str(prefix) + suffix)
        try:
            if suffix.endswith((".points", ".observations")):
                width = 3 if suffix.endswith(".points") else 5
                values = np.fromfile(target, "<f8")
                checks["finite" + suffix] = bool(
                    values.size % width == 0 and np.isfinite(values).all()
                )
                if suffix == ".observations" and checks["finite" + suffix]:
                    values = values.reshape(-1, 5)
                    keys = values[:, :3]
                    points = Path(str(prefix) + ".points").stat().st_size // 24
                    checks["valid_observation_identities"] = bool(
                        np.equal(keys, np.floor(keys)).all()
                        and np.all((keys[:, 0] >= 0) & (keys[:, 0] < len(expected)))
                        and np.all((keys[:, 1] >= 0) & (keys[:, 1] < 2))
                        and np.all((keys[:, 2] >= 0) & (keys[:, 2] < points))
                    )
            else:
                values = np.loadtxt(target, ndmin=1)
                original = np.loadtxt(Path(backup) / target.name, ndmin=1)
                checks["finite" + suffix] = bool(
                    values.shape == original.shape and np.isfinite(values).all()
                )
        except (ValueError, OSError):
            checks["finite" + suffix] = False
    if ".base" in present:
        try:
            base = np.loadtxt(str(prefix) + ".base", ndmin=2)
            checks["valid_base_pose_fallback"] = bool(
                base.shape == (len(expected), 8)
                and np.array_equal(base[:, 0], expected)
                and np.isfinite(base).all()
                and np.all(abs(np.linalg.norm(base[:, 4:], axis=1) - 1) < 1e-5)
            )
        except (ValueError, OSError):
            checks["valid_base_pose_fallback"] = False
    return {"passed": all(checks.values()), "checks": checks}


def coherent_dump_branch(log):
    """Native9eaf overwrites debug dumps before restoring failed final state."""
    if "FINAL_CALIBRATION_FROZEN" not in log:
        return True
    markers = re.findall(r"^FINAL_SELECTION (\w+) reason=([^\n]+)", log, re.M)
    return bool(markers and markers[-1] == ("refined", "accepted_stable_cost_decrease"))


def intended_refinement_complete(log, extra):
    if "FINAL_REFINE" not in extra:
        return True
    markers = re.findall(r"^FINAL_SELECTION (\w+) reason=([^\n]+)", log, re.M)
    return bool(
        markers
        and markers[-1]
        in [("refined", "accepted_stable_cost_decrease"), ("baseline", "no_additions")]
    )


def preserve_rejected_family(prefix):
    prefix = Path(prefix)
    folder = Path(tempfile.mkdtemp(prefix="optimizer-rejected-", dir=prefix.parent))
    result = {}
    for suffix in FAMILY_SUFFIXES:
        source = Path(str(prefix) + suffix)
        if source.is_file() and not source.is_symlink():
            result[suffix] = sha(source)
            shutil.move(source, folder / source.name)
    return {"path": str(folder), "sha256": result}


def _record(log, report):
    target = Path(str(log) + ".iteration-recovery.json")
    temp = Path(str(target) + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temp, target)


def run_with_iteration_recovery(run, command, timeout, extra, log, clock=time.monotonic):
    """Return selected (native_exit, selected_timeout, report).

    The caller's single native budget covers primary, retry and validation.
    `run` has the existing pipeline signature (command,timeout,extra,log).
    Only the current native's ITERS and time ceilings change on retry.
    """
    start = clock()
    log = Path(log)
    input_path = Path(command[1])
    prefix = Path(command[2])
    native = Path(command[0])
    native_identity = sha(native)
    report = {
        "policy": POLICY,
        "native_budget_seconds": float(timeout),
        "native_sha256": native_identity,
        "primary_native_environment": dict(extra),
        "retried": False,
        "selected": "primary",
    }
    try:
        primary_rc = run(command, timeout=timeout, extra=extra, log=log)
        primary_timeout = False
    except subprocess.TimeoutExpired:
        primary_rc = None
        primary_timeout = True
    report.update(
        primary_exit=primary_rc, primary_timeout=primary_timeout, primary_seconds=clock() - start
    )
    primary_text = log.read_text(errors="replace") if log.exists() else ""
    parsed = (
        parse_solver_log(primary_text)
        if log.exists()
        else {"passes": [], "complete": False, "report_count": 0}
    )
    report["primary_solver"] = parsed
    cap = int(extra.get("ITERS", 250))
    if (
        primary_rc != 0
        or primary_timeout
        or cap != POLICY["primary_iteration_cap"]
        or not iteration_trigger(parsed, cap)
        or not coherent_dump_branch(primary_text)
    ):
        report["reason"] = "primary_not_eligible_for_iteration_retry"
        report["elapsed_seconds"] = clock() - start
        _record(log, report)
        return primary_rc, primary_timeout, report
    if (
        timeout - (clock() - start)
        <= POLICY["minimum_retry_seconds"] + POLICY["validation_reserve_seconds"]
    ):
        report["reason"] = "insufficient_remaining_native_budget"
        report["elapsed_seconds"] = clock() - start
        _record(log, report)
        return primary_rc, False, report
    if sha(native) != native_identity:
        report.update(reason="native_changed_before_retry", elapsed_seconds=clock() - start)
        _record(log, report)
        return primary_rc, False, report
    try:
        backup, present = snapshot_family(prefix)
    except (ValueError, OSError) as error:
        report.update(
            reason="primary_state_not_preservable",
            snapshot_error=repr(error),
            elapsed_seconds=clock() - start,
        )
        _record(log, report)
        return primary_rc, False, report
    original_inputs = input_hashes(input_path)
    primary_log = Path(str(log) + ".primary")
    shutil.copyfile(log, primary_log)
    remaining = timeout - (clock() - start) - POLICY["validation_reserve_seconds"]
    if remaining <= POLICY["minimum_retry_seconds"]:
        report["reason"] = "snapshot_used_remaining_retry_budget"
        report["elapsed_seconds"] = clock() - start
        _record(log, report)
        return primary_rc, False, report
    retry_extra = dict(
        extra,
        ITERS=str(POLICY["retry_iteration_cap"]),
        TOTAL_TIME=str(max(1.0, remaining - 3.0)),
        MAX_TIME=str(max(1.0, (remaining - 4.0) / 2)),
    )
    if parameter_trigger(parsed):
        retry_extra["PARAM_TOL"] = "0"
        report["retry_trigger"] = "unsettled_parameter_tolerance"
    else:
        report["retry_trigger"] = "terminal_iteration_ceiling"
    report["retry_native_environment"] = dict(retry_extra)
    retry_log = Path(str(log) + ".retry")
    report.update(
        retried=True,
        retry_timeout_seconds=remaining,
        primary_snapshot=str(backup),
        original_input_hashes=original_inputs,
    )
    # Retry from the same input seed, never from stale output/base sidecars.
    for suffix in FAMILY_SUFFIXES:
        target = Path(str(prefix) + suffix)
        if target.exists():
            target.unlink()
    retry_rc = None
    retry_timeout = False
    error = None
    validation = None
    interruption = None
    try:
        retry_rc = run(command, timeout=remaining, extra=retry_extra, log=retry_log)
        retry_text = retry_log.read_text(errors="replace")
        retry_solver = parse_solver_log(retry_text)
        report["retry_solver"] = retry_solver
        unchanged = input_hashes(input_path) == original_inputs
        report["original_inputs_unchanged"] = unchanged
        if not unchanged:
            raise RuntimeError("native retry changed original sensor input family")
        report["retry_diagnostic_branch_coherent"] = coherent_dump_branch(retry_text)
        report["retry_intended_refinement_complete"] = intended_refinement_complete(
            retry_text, extra
        )
        if (
            retry_rc == 0
            and retry_converged(retry_solver)
            and coherent_dump_branch(retry_text)
            and intended_refinement_complete(retry_text, extra)
        ):
            validation = validate_selected_state(input_path, prefix, present, backup)
        accepted = bool(
            retry_rc == 0 and validation and validation["passed"] and clock() - start <= timeout
        )
    except subprocess.TimeoutExpired:
        retry_timeout = True
        accepted = False
    except Exception as exc:
        error = repr(exc)
        accepted = False
    except BaseException as exc:
        interruption = exc
        error = repr(exc)
        accepted = False
    unchanged = input_hashes(input_path) == original_inputs
    report["original_inputs_unchanged"] = unchanged
    report["native_unchanged"] = sha(native) == native_identity
    accepted = accepted and unchanged and report["native_unchanged"]
    report.update(
        retry_exit=retry_rc,
        retry_timeout=retry_timeout,
        retry_error=error,
        retry_state_validation=validation,
    )
    if accepted:
        selected_log = Path(str(log) + ".selected-tmp")
        shutil.copyfile(retry_log, selected_log)
        os.replace(selected_log, log)
        report.update(selected="iteration_retry", reason="converged_retry_with_complete_state")
        selected_rc = retry_rc
    else:
        report["rejected_candidate_family"] = preserve_rejected_family(prefix)
        restore_family(prefix, backup, present)
        report.update(reason="retry_rejected_primary_restored", primary_state_restored_exactly=True)
        selected_rc = primary_rc
    report["elapsed_seconds"] = clock() - start
    _record(log, report)
    if interruption is not None:
        raise interruption
    if not unchanged:
        raise RuntimeError("native retry mutated original sensor inputs; primary outputs restored")
    # A rejected retry timeout does not relabel the selected complete primary
    # as a timed-out solve; the existing outer fallback must see its true exit.
    return selected_rc, False, report
