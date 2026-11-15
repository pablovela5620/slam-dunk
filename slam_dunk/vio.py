"""Run the preserved Full VIO startup-recovery policy in an isolated process."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .backend import _engine_module
from .runtime import ENGINE, write_json


def main():
    work = Path(sys.argv[1])
    started = time.monotonic() - float(sys.argv[2])
    attempts = []

    def invoke(command, timeout=None, extra=None, log=None):
        before = time.monotonic()
        env = dict(os.environ)
        env.update(extra or {})
        row = dict(
            command=list(map(str, command)), timeout_seconds=timeout, extra_environment=extra or {}
        )
        try:
            with Path(log).open("w") as stream:
                result = subprocess.run(
                    row["command"],
                    env=env,
                    timeout=timeout,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            row.update(exit=result.returncode, timeout=False)
            return result.returncode
        except subprocess.TimeoutExpired:
            row.update(exit=None, timeout=True)
            raise
        finally:
            row["seconds"] = time.monotonic() - before
            attempts.append(row)
            write_json(work / "vio-attempts.json", attempts)

    before = time.monotonic()
    error = None
    try:
        rc = invoke(
            [ENGINE / "native/build/aria_vio", work, work / "vio.txt"],
            timeout=600,
            extra={"THREADS": "4"},
            log=work / "vio.log",
        )
    except subprocess.TimeoutExpired as exception:
        rc, error = None, str(exception)
    report = _engine_module("startup_recovery").recover(
        work,
        ENGINE,
        invoke,
        dict(os.environ),
        started,
        1700.0,
        rc,
        time.monotonic() - before,
        error,
    )
    if report["selected_attempt"] is None:
        raise RuntimeError("No healthy VIO startup; see recovery.json and vio-attempts.json")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
