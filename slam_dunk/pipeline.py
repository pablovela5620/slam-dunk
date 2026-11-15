"""Cold sensor inference with the same reusable frontend/BA interfaces."""

from pathlib import Path

from .config import Config
from .frontend import prepare
from .backend import run_backend


def run(recording, output, config=None, *, seed=None):
    """Prepare fresh tracks, refine, and return the dense trajectory path."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    packet = prepare(recording, output / "frontend", seed=seed)
    return run_backend(packet, output / "estimate", config or Config())
