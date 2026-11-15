"""SLAM Dunk: Full R17-C frontend packets, configurable BA, and dense export."""

from .config import Config


def prepare(recording, output, *, seed=None):
    """Create an unfitted frontend packet from prepared recording sensors."""
    from .frontend import prepare as implementation

    return implementation(recording, output, seed=seed)


def run_backend(packet, output, config=None, *, binary=None):
    """Refine a cold packet independently; return its dense trajectory path."""
    from .backend import run_backend as implementation

    return implementation(packet, output, config, binary=binary)


def run(recording, output, config=None, *, seed=None):
    """Prepare fresh sensor tracks and run the configured Full refinement."""
    from .pipeline import run as implementation

    return implementation(recording, output, config, seed=seed)


__all__ = ["Config", "prepare", "run_backend", "run"]
