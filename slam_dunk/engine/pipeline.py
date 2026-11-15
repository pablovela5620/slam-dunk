import os, sys, time, tempfile, subprocess, struct, contextlib, json
from pathlib import Path
from startup_recovery import recover
from final_refinement import restore_base
from runtime_audit import RunAudit

ROOT = Path(__file__).resolve().parent
# One configuration is shared by every recording. Environment overrides are
# reserved for controlled development runs and do not depend on sequence names.
settings = ROOT / "settings.json"
if settings.exists():
    for key, value in json.loads(settings.read_text()).items():
        if key.startswith("ARIA_"):
            os.environ.setdefault(key, str(value))
src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
deadline = min(1700.0, float(os.environ.get("ARIA_TOTAL_SECONDS", "1700")))
env = dict(
    os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", THREADS="8"
)
env.setdefault("DETECT_SCALE", ".5")
env.setdefault("DETECT_FRACTION", ".8")
audit = RunAudit(ROOT, src, out, started, deadline)
stage_events = []


def run(args, timeout=None, extra=None, log=None):
    call_started = time.monotonic()
    e = env.copy()
    e.update(extra or {})
    print("Running", str(args[0]), "elapsed", round(time.monotonic() - started, 2), flush=True)
    label = (
        Path(args[1]).name
        if len(args) > 1 and Path(args[0]).name.startswith("python")
        else Path(args[0]).name
    )
    remaining = max(0.0, deadline - 5 - (time.monotonic() - started))
    timeout = min(float(timeout), remaining) if timeout is not None else remaining
    audit_index, audit_log = audit.begin(label, args, timeout, extra)
    if log is None:
        log = audit_log
    # Identity/hash/logging overhead consumes the same requested stage budget.
    timeout = max(0.0, timeout - (time.monotonic() - call_started) - 0.1)
    audit.events[audit_index]["applied_child_timeout_seconds"] = timeout
    audit.flush()
    stage_started = time.monotonic()
    try:
        with open(log, "w") as f:
            rc = subprocess.run(
                [str(x) for x in args], env=e, timeout=timeout, stdout=f, stderr=subprocess.STDOUT
            ).returncode
    except subprocess.TimeoutExpired:
        seconds = time.monotonic() - stage_started
        audit.end(audit_index, None, True, seconds, log)
        event = dict(stage=label, exit=None, timeout=True, seconds=seconds)
        stage_events.append(event)
        print("STAGE_EXIT", json.dumps(event), flush=True)
        raise
    except Exception as error:
        audit.end(audit_index, None, False, time.monotonic() - stage_started, log, str(error))
        raise
    audit.end(audit_index, rc, False, time.monotonic() - stage_started, log)
    event = dict(stage=label, exit=rc, timeout=False, seconds=time.monotonic() - stage_started)
    stage_events.append(event)
    print("STAGE_EXIT", json.dumps(event), flush=True)
    return rc


# Produce a complete file immediately, then replace it atomically as estimates improve.
run(["/opt/venv/bin/python", ROOT / "fallback.py", src, out])
selection = "camera_fallback"
workdir = os.environ.get("ARIA_WORK_DIR")
if workdir:
    Path(workdir).mkdir(parents=True, exist_ok=True)
with (
    contextlib.nullcontext(workdir) if workdir else tempfile.TemporaryDirectory(prefix="aria_")
) as d:
    w = Path(d)
    prefetch = None
    featurelog = None
    ba_step = ".6"
    feature_settings = dict(
        SIFT_FEATURES=os.environ.get("ARIA_FEATURES", "900"),
        SIFT_CONTRAST=os.environ.get("ARIA_CONTRAST", ".006"),
        SIFT_DARK_ADAPT=os.environ.get("ARIA_DARK_ADAPT", "0"),
        PRECISE_UPSCALE=os.environ.get("ARIA_PRECISE", "0"),
        ROOT_SIFT="1",
        TRACK_CONSISTENCY="1",
        MATCH_MAX_DISTANCE=".7",
    )
    try:
        if run(["/opt/venv/bin/python", ROOT / "prepare.py", src, w], timeout=60):
            raise RuntimeError("preprocessing failed")
        rows = (w / "frames.txt").read_text().splitlines()
        duration = (int(rows[-1].split()[0]) - int(rows[0].split()[0])) * 1e-9
        ba_step = str(float(os.environ.get("ARIA_BA_STEP", max(0.6, duration / 1500.0))))
        if os.environ.get("ARIA_PREFETCH", "1") == "1":
            featurelog = open(w / "features.log", "w")
            pe = env.copy()
            pe.update(feature_settings)
            pe.update(BA_STEP=ba_step, SIFT_THREADS="4", TEMPORAL_PREFETCH="1")
            prefetch = subprocess.Popen(
                ["/opt/venv/bin/python", str(ROOT / "prefetch_features.py"), str(src), str(w)],
                env=pe,
                stdout=featurelog,
                stderr=subprocess.STDOUT,
            )
        vio_started = time.monotonic()
        vio_error = None
        try:
            rc = run(
                [ROOT / "native/build/aria_vio", w, w / "vio.txt"],
                timeout=max(5, min(600, deadline - 300) - (time.monotonic() - started)),
                extra=dict(THREADS="4"),
                log=w / "vio.log",
            )
        except subprocess.TimeoutExpired as e:
            rc = None
            vio_error = str(e)
            print("VIO deadline", flush=True)
        recovery = recover(
            w, ROOT, run, env, started, deadline, rc, time.monotonic() - vio_started, vio_error
        )
        if recovery["selected_attempt"] is not None:
            rc = 0
        if rc:
            print((w / "vio.log").read_text()[-1500:], flush=True)
        if not (w / "vio.txt").exists():
            raise RuntimeError("VIO failed")
    except subprocess.TimeoutExpired:
        print("VIO deadline", flush=True)
    except Exception as e:
        print(str(e), flush=True)
    if prefetch is not None:
        try:
            prefetch.wait(timeout=20)
        except subprocess.TimeoutExpired:
            prefetch.terminate()
            try:
                prefetch.wait(timeout=20)
            except subprocess.TimeoutExpired:
                prefetch.kill()
                prefetch.wait()
        featurelog.close()
        print((w / "features.log").read_text()[-400:], flush=True)
    if (w / "vio.txt").exists() and (w / "vio.txt").stat().st_size > 300:
        try:
            if not run(
                ["/opt/venv/bin/python", ROOT / "densify.py", src, w / "vio.txt", out], timeout=10
            ):
                selection = "initial_vio"
        except Exception as e:
            print("Initial output:", str(e), flush=True)
        try:
            # A bounded visual pass supplies longer tracks and geometrically verified revisits.
            with (w / "frames.txt").open() as f:
                rows = f.readlines()
            duration = (int(rows[-1].split()[0]) - int(rows[0].split()[0])) * 1e-9
            ba_step = str(float(os.environ.get("ARIA_BA_STEP", max(0.6, duration / 1500.0))))
            if time.monotonic() - started < deadline - 200:
                budget = min(500, deadline - 200 - (time.monotonic() - started))
                try:
                    rc = run(
                        ["/opt/venv/bin/python", ROOT / "sift_tracks.py", src, w],
                        timeout=budget,
                        extra=dict(
                            feature_settings,
                            BA_STEP=ba_step,
                            SIFT_THREADS="8",
                            SIFT_BUDGET=str(budget - 2),
                        ),
                        log=w / "sift.log",
                    )
                except subprocess.TimeoutExpired:
                    print("Visual matching deadline; refining available tracks.", flush=True)
                print((w / "sift.log").read_text()[-1800:], flush=True)
            if os.environ.get("ARIA_DUAL", "0") == "1" and (src / "imu1/data.csv").exists():
                try:
                    rcdual = run(
                        ["/opt/venv/bin/python", ROOT / "dual_imu.py", src, w],
                        timeout=45,
                        log=w / "dual.log",
                    )
                    if (
                        not rcdual
                        and (w / "imu_fused.bin").stat().st_size == (w / "imu.bin").stat().st_size
                    ):
                        os.replace(w / "imu_fused.bin", w / "imu.bin")
                except Exception as e:
                    print("Second IMU:", str(e), flush=True)
            tracks = (
                "tracks_ba.bin"
                if (w / "tracks_ba.bin").exists() and (w / "tracks_ba.bin").stat().st_size > 48000
                else "tracks.bin"
            )
            if time.monotonic() - started < deadline - 30:
                rc = run(
                    ["/opt/venv/bin/python", ROOT / "export_tracks.py", src, w],
                    timeout=35,
                    extra=dict(
                        BA_STEP=ba_step,
                        TRACKS_FILE=tracks,
                        MIN_OBS="3",
                        TRI_ERR="5",
                        SAVE_META="1",
                        SAVE_CANDIDATES=(
                            "1"
                            if (
                                os.environ.get("ARIA_RECOVER", "0") == "1"
                                or os.environ.get("ARIA_FINAL_REFINE", "0") == "1"
                            )
                            else ""
                        ),
                        SIFT_PIXEL_BIAS="0",
                        MIX_FLOW=(
                            os.environ.get("ARIA_MIX_FLOW", "1")
                            if tracks == "tracks_ba.bin"
                            else "0"
                        ),
                        FLOW_PIXEL_BIAS=(
                            "0" if feature_settings["PRECISE_UPSCALE"] == "1" else ".25"
                        ),
                    ),
                    log=w / "export.log",
                )
                print((w / "export.log").read_text()[-500:], flush=True)
                budget = min(
                    1200.0,
                    float(os.environ.get("ARIA_SOLVE_SECONDS", "1200")),
                    deadline - 25 - (time.monotonic() - started),
                )
                enough = False
                if not rc:
                    with (w / "input.bin").open("rb") as f:
                        nposes, npoints, nobs, nimu = struct.unpack("qqqq", f.read(32))
                    enough = npoints >= 30 and nobs >= max(500, 10 * nposes)
                if not rc and enough and budget > 8:
                    extra = dict(
                        THREADS="6",
                        CHECKPOINT_INTERVAL="15",
                        OPENBLAS_NUM_THREADS="1",
                        BG_NORMALIZE="1",
                        BIAS_THERMAL="1",
                        BIAS_TEMP_SIG=".00005",
                        POSE_STOP="1",
                        TOTAL_TIME=str(max(1, budget - 3)),
                        FIRST_FRACTION=".52",
                        SOFT_CAUCHY="1",
                        PLS_FROZEN="1",
                        THERMAL="1",
                        OUTLIER="3",
                        KEEP_OBS="2",
                        CHUNK_DT="5",
                        SIGMA_G=".0005",
                        SIGMA_A=".003",
                        ACC_ALIGN="1",
                        ACC_SCALE="1",
                        REFINE_FOCAL="1",
                        FOCAL_EQUAL="1",
                        MAX_TIME=str(max(1, (budget - 4) / 2)),
                        FUN_TOL="0",
                        ITERS=os.environ.get("ARIA_ITERS", "125"),
                        OUTER="2",
                        FOCAL_RW=".035",
                        GYRO_REFINE="1",
                        TIME_REFINE="1",
                        BIAS_RW=".00002",
                        ACC_BIAS_RW=".005",
                    )
                    for option in [
                        "ACC_CROSS",
                        "OBS_MOTION",
                        "OBS_VELOCITY",
                        "RECOVER",
                        "SCALE_WEIGHT",
                        "ADAPTIVE_LOSS",
                        "FINAL_REFINE",
                        "WHITEN_VISUAL",
                        "LATE_PRECISION",
                        "PARAMETER_CONTINUATION",
                        "FINAL_TIMING_VELOCITY",
                    ]:
                        if os.environ.get("ARIA_" + option, "0") == "1":
                            extra[option] = "1"
                    if os.environ.get("ARIA_SOFT_CAUCHY", "1") == "0":
                        extra.pop("SOFT_CAUCHY", None)
                    extra["OUTER"] = os.environ.get("ARIA_OUTER", "2")
                    extra["ADAPTIVE_MIN"] = os.environ.get("ARIA_ADAPTIVE_MIN", ".6")
                    optimizer = (
                        "batch_joint" if os.environ.get("ARIA_JOINT", "0") == "1" else "batch_time"
                    )
                    print("NATIVE_FLAGS", json.dumps(extra, sort_keys=True), flush=True)
                    native_rc = None
                    timed_out = False
                    try:
                        command = [ROOT / "native/build" / optimizer, w / "input.bin", w / "ba.txt"]
                        if os.environ.get("ARIA_ITERATION_RECOVERY", "0") == "1":
                            from iteration_recovery import run_with_iteration_recovery

                            native_rc, timed_out, iteration_report = run_with_iteration_recovery(
                                run, command, budget, extra, w / "ba.log"
                            )
                            print(
                                "ITERATION_SELECTION",
                                iteration_report["selected"],
                                "retried",
                                iteration_report["retried"],
                                "reason",
                                iteration_report["reason"],
                                flush=True,
                            )
                        else:
                            native_rc = run(command, timeout=budget, extra=extra, log=w / "ba.log")
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        print(
                            "Optimization deadline; using latest complete checkpoint.", flush=True
                        )
                    print("NATIVE_EXIT", native_rc, "timeout", timed_out, flush=True)
                    optimizer_log = (w / "ba.log").read_text()
                    print(optimizer_log[-1500:], flush=True)
                    for line in optimizer_log.splitlines():
                        if line.startswith("FINAL_"):
                            print(line, flush=True)
                    enabled = os.environ.get("ARIA_FINAL_REFINE", "0") == "1"
                    last_final_selection = next(
                        (
                            line
                            for line in reversed(optimizer_log.splitlines())
                            if line.startswith("FINAL_SELECTION ")
                        ),
                        "",
                    )
                    accepted = enabled and last_final_selection.startswith(
                        "FINAL_SELECTION refined"
                    )
                    if enabled and (timed_out or native_rc != 0):
                        if restore_base(w):
                            print(
                                "FINAL_SELECTION baseline reason=controller_optimizer_failure",
                                flush=True,
                            )
                        accepted = False
                    if (w / "ba.txt").exists():
                        try:
                            rc = run(
                                ["/opt/venv/bin/python", ROOT / "fuse.py", src, w, out],
                                timeout=30,
                                log=w / "fuse.log",
                            )
                            if rc:
                                raise RuntimeError("dense reconstruction failed")
                            selection = "final_refinement" if accepted else "main_refinement"
                            for line in (w / "fuse.log").read_text().splitlines():
                                if "fallback" in line.lower():
                                    print(line, flush=True)
                        except Exception as error:
                            print("Final reconstruction failed:", str(error), flush=True)
                            if enabled and restore_base(w):
                                rc = run(
                                    ["/opt/venv/bin/python", ROOT / "fuse.py", src, w, out],
                                    timeout=30,
                                    log=w / "fuse_fallback.log",
                                )
                                if not rc:
                                    selection = "main_refinement_fallback"
                                print(
                                    "FINAL_SELECTION baseline reason=dense_reconstruction_failure exit",
                                    rc,
                                    flush=True,
                                )
        except Exception as e:
            print("Refinement:", str(e), flush=True)
    audit.finish(w, selection)
if workdir:
    (Path(workdir) / "stage-exits.json").write_text(json.dumps(stage_events, indent=2))
print("OUTPUT_SELECTION", selection, flush=True)
print("Finished in", round(time.monotonic() - started, 2), "seconds", flush=True)
