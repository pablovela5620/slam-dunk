"""Subprocess isolation and receipts shared by frontend and backend."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ENGINE = Path(__file__).resolve().parent / "engine"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def clean_environment():
    # Never inherit scientific flags from a previous run or the caller's shell.
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "TMPDIR")
        if key in os.environ
    }
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        NUMBA_CACHE_DIR="/tmp/numba",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


class Runner:
    def __init__(self, output, config=None):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.receipt = dict(status="running", commands=[], config=config, started_unix=time.time())
        self.flush()

    def flush(self):
        write_json(self.output / "receipt.json", self.receipt)

    def call(self, command, timeout=None, extra=None, log=None):
        command = list(map(str, command))
        log = Path(log) if log else self.output / f'stage-{len(self.receipt["commands"])}.log'
        env = clean_environment()
        env.update(extra or {})
        row = dict(command=command, timeout_seconds=timeout, environment=extra or {}, log=str(log))
        started = time.monotonic()
        try:
            with log.open("w") as stream:
                result = subprocess.run(
                    command, env=env, timeout=timeout, stdout=stream, stderr=subprocess.STDOUT
                )
            row.update(exit=result.returncode, timeout=False)
            return result.returncode
        except subprocess.TimeoutExpired:
            row.update(exit=None, timeout=True)
            raise
        finally:
            row["seconds"] = time.monotonic() - started
            self.receipt["commands"].append(row)
            self.flush()

    def checked(self, command, timeout, extra=None, log=None):
        rc = self.call(command, timeout, extra, log)
        if rc:
            raise RuntimeError(f"Stage failed with exit {rc}: {command[0]}; see {self.output}")
