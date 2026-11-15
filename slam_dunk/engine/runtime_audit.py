"""Small persistent provenance/stop records; no decisions based on accuracy."""

import hashlib
import json
import os
import platform
import resource
import shutil
import time
from pathlib import Path


def file_identity(path):
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def cgroup():
    result = {}
    for name in (
        "cpu.max",
        "cpuset.cpus.effective",
        "memory.max",
        "memory.current",
        "memory.peak",
        "memory.events",
        "io.stat",
    ):
        path = Path("/sys/fs/cgroup") / name
        try:
            result[name] = path.read_text().strip()
        except OSError:
            pass
    return result


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


class RunAudit:
    def __init__(self, root, src, output, started, deadline):
        self.root, self.src, self.output = map(Path, (root, src, output))
        self.started = started
        parent = Path(str(output) + ".audit")
        self.directory = parent / (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(os.getpid())
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events = []
        runtime = list(root.glob("*.py")) + [root / "run.sh", root / "settings.json"]
        runtime += [
            p for p in (root / "native/build").iterdir() if p.is_file() and os.access(p, os.X_OK)
        ]
        inventory = root / "RUNTIME_MANIFEST.json"
        if inventory.exists():
            runtime.append(inventory)
        self.receipt = {
            "version": 1,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recording": str(src),
            "output": str(output),
            "root": str(root),
            "controller_budget_seconds": deadline,
            "native_budget_cap_seconds": 1200,
            "python": platform.python_version(),
            "runtime": [file_identity(p) for p in sorted(runtime)],
            "configuration": {k: v for k, v in os.environ.items() if k.startswith("ARIA_")},
            "allocation_and_resources_at_start": cgroup(),
            "input_identity": [],
        }
        # Sensor/calibration/index hashes identify the actual current input. Labels,
        # ground truth, GPS, survey files and previous estimates are never opened.
        names = ["calibration.json", "factory_calibration.json", "info.json"]
        for sensor in ("cam0", "cam1", "imu0", "imu1", "mag0", "baro0"):
            names.append(sensor + "/data.csv")
            if sensor.startswith("cam"):
                names.append(sensor + "/meta.csv")
        for name in names:
            identity = file_identity(src / name)
            if identity is not None:
                self.receipt["input_identity"].append(identity)
        # Camera members are exhaustively indexed by the hashed CSV. Keep a few
        # actual JPEG hashes too, explicitly marked as samples rather than full
        # image-content hashing. Runtime feature code consumes these indexed files.
        samples = []
        for camera in ("cam0", "cam1"):
            rows = [
                x.split(",")
                for x in (src / camera / "data.csv").read_text().splitlines()
                if x and not x.startswith("#")
            ]
            for index in sorted(set([0, len(rows) // 2, len(rows) - 1])):
                if rows:
                    samples.append(file_identity(src / camera / "data" / rows[index][1]))
        self.receipt["sample_camera_content_identity"] = samples
        self.receipt["camera_content_hash_scope"] = (
            "sampled; full member identity is the complete hashed camera CSV"
        )
        self.flush()

    def flush(self):
        self.receipt["stages"] = self.events
        self.receipt["elapsed_seconds"] = time.monotonic() - self.started
        write_json(self.directory / "run.json", self.receipt)

    def begin(self, label, args, timeout, extra):
        index = len(self.events)
        event = {
            "index": index,
            "stage": label,
            "command": list(map(str, args)),
            "timeout_seconds": timeout,
            "extra_environment": extra or {},
            "elapsed_at_start_seconds": time.monotonic() - self.started,
            "status": "running",
        }
        event["executable"] = file_identity(args[0])
        if len(args) > 1 and Path(args[0]).name.startswith("python"):
            event["script"] = file_identity(args[1])
        if label == "aria_vio" and len(args) > 1:
            native_work = Path(args[1])
            event["native_input_files"] = [
                file_identity(native_work / name)
                for name in ("frames.txt", "calib.json", "imu.bin", "cam0_lut.bin", "cam1_lut.bin")
            ]
        elif label in ("batch_joint", "batch_time") and len(args) > 1:
            native_input = Path(args[1])
            event["native_input_files"] = [
                file_identity(path)
                for path in sorted(native_input.parent.glob(native_input.name + "*"))
                if path.is_file()
            ]
        self.events.append(event)
        self.flush()
        return index, self.directory / (f"{index:02d}-" + label + ".log")

    def end(self, index, rc, timed_out, seconds, log, error=None):
        event = self.events[index]
        event.update(
            status="finished", returncode=rc, timed_out=timed_out, seconds=seconds, error=error
        )
        destination = self.directory / (f"{index:02d}-" + event["stage"] + ".log")
        if log is not None and Path(log).exists() and Path(log).resolve() != destination.resolve():
            shutil.copy2(log, destination)
        event["log"] = file_identity(destination)
        self.flush()

    def finish(self, work, selection):
        work = Path(work)
        retained = []
        # Retain every attempt log, including logs moved by startup recovery.
        for path in sorted(work.rglob("*")):
            if not path.is_file():
                continue
            if self.directory in path.parents:
                continue
            if (
                ".log" in path.name
                or path.name.endswith(".json")
                or path.name.startswith(("ba.txt", "vio.txt"))
            ):
                destination = self.directory / "work" / path.relative_to(work)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                retained.append(str(destination.relative_to(self.directory)))
        self.receipt["retained_work_records"] = retained
        self.receipt["derived_input_identity"] = [
            file_identity(p) for p in sorted(work.glob("input.bin*"))
        ]
        self.receipt["native_output_identity"] = [
            file_identity(p) for p in sorted(work.glob("ba.txt*")) if p.is_file()
        ]
        self.receipt["output_selection"] = selection
        self.receipt["output_identity"] = file_identity(self.output)
        native_log = work / "ba.log"
        native_text = native_log.read_text(errors="replace") if native_log.exists() else ""
        terminal_selection = next(
            (
                line
                for line in reversed(native_text.splitlines())
                if line.startswith("FINAL_SELECTION ")
            ),
            None,
        )
        final_started = "FINAL_CALIBRATION_FROZEN" in native_text
        dump_scope = (
            "selected_final_state"
            if terminal_selection and terminal_selection.startswith("FINAL_SELECTION refined")
            else (
                "rejected_or_partial_final_dumps_may_not_match_restored_main_poses"
                if final_started
                else "main_pass_state" if native_text else "no_native_state"
            )
        )
        sift_events = [event for event in self.events if event["stage"] == "sift_tracks.py"]
        native_events = [
            event for event in self.events if event["stage"] in ("batch_joint", "batch_time")
        ]
        iteration_path = work / "ba.log.iteration-recovery.json"
        iteration_report = (
            json.loads(iteration_path.read_text()) if iteration_path.exists() else None
        )
        selected_native_completed = bool(
            native_events
            and native_events[-1].get("returncode") == 0
            and not native_events[-1].get("timed_out")
        )
        if iteration_report is not None:
            selected_native_completed = (
                bool(
                    iteration_report.get("primary_exit") == 0
                    and not iteration_report.get("primary_timeout")
                )
                if iteration_report.get("selected") == "primary"
                else bool(
                    iteration_report.get("retry_exit") == 0
                    and not iteration_report.get("retry_timeout")
                )
            )
        matching_health = work / "sift-health.json"
        self.receipt["completion_evidence"] = {
            "visual_source": (
                "SIFT_plus_flow"
                if (work / "tracks_ba.bin").exists()
                and (work / "tracks_ba.bin").stat().st_size > 48000
                else "initializer_flow"
            ),
            "matching_stage_completed": bool(
                sift_events
                and sift_events[-1].get("returncode") == 0
                and not sift_events[-1].get("timed_out")
            ),
            "matching_health": (
                json.loads(matching_health.read_text()) if matching_health.exists() else None
            ),
            "selected_native_stage_completed": selected_native_completed,
            "iteration_recovery": iteration_report,
            "any_ceres_failure_report": "Termination: FAILURE" in native_text,
            "terminal_native_selection": terminal_selection,
            "native_debug_dump_scope": dump_scope,
            "native_pass_reports": [
                line for line in native_text.splitlines() if line.startswith("Ceres Solver Report")
            ],
            "note": "Successful exits/finite output do not establish intended-stage completion or physical accuracy; full logs and explicit fallbacks are retained.",
        }
        self.receipt["native_stops_and_selection"] = []
        for path in sorted(work.rglob("*")):
            if not path.is_file() or ".log" not in path.name or path.name.endswith(".json"):
                continue
            if self.directory in path.parents:
                continue
            for line in path.read_text(errors="replace").splitlines():
                if line.startswith(
                    (
                        "STOP_REASON",
                        "FINAL_",
                        "PASS_CONTINUATION_",
                        "POSE_STABILITY",
                        "Ceres Solver Report",
                        "NATIVE_EXIT",
                        "SIFT_EXIT",
                    )
                ):
                    self.receipt["native_stops_and_selection"].append(
                        {"log": str(path.relative_to(work)), "line": line}
                    )
        self.receipt["allocation_and_resources_at_end"] = cgroup()
        self.receipt["children_rusage_at_end"] = {
            "user_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
            "system_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        }
        try:
            import numpy as np

            expected = [
                int(x.split(",")[0])
                for x in (self.src / "cam0/data.csv").read_text().splitlines()
                if x and not x.startswith("#")
            ]
            rows = [
                x.split()
                for x in self.output.read_text().splitlines()
                if x and not x.startswith("#")
            ]
            stamps = [int(x[0]) for x in rows]
            values = np.array([[float(v) for v in x[1:]] for x in rows])
            self.receipt["output_check"] = {
                "expected_frames": len(expected),
                "output_rows": len(rows),
                "exact_timestamps": stamps == expected,
                "unique_timestamps": len(set(stamps)) == len(stamps),
                "all_finite": bool(np.isfinite(values).all()),
                "max_quaternion_norm_error": float(
                    np.max(np.abs(np.linalg.norm(values[:, 3:], axis=1) - 1))
                ),
            }
        except Exception as error:
            self.receipt["output_check"] = {"error": str(error)}
        self.flush()
