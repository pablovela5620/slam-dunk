"""Command line entry point for preparation and controlled ablations."""

import argparse
from pathlib import Path

from .config import Config


def main():
    parser = argparse.ArgumentParser(description="SLAM Dunk Full R17-C recipe")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Create one cold frontend packet")
    prep.add_argument("recording", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument(
        "--seed", type=Path, help="Unrefined VIO trajectory for matched-packet reproduction"
    )
    backend = commands.add_parser(
        "refine", help="Refine a cold packet; writes trajectory.txt and receipts"
    )
    backend.add_argument("packet", type=Path)
    backend.add_argument("output", type=Path)
    backend.add_argument("--config", type=Path, required=True)
    raw = commands.add_parser("run", help="Prepare sensors and refine in one command")
    raw.add_argument("recording", type=Path)
    raw.add_argument("output", type=Path)
    raw.add_argument("--config", type=Path, required=True)
    raw.add_argument("--seed", type=Path)
    check = commands.add_parser("config", help="Validate a config and print its exact native flags")
    check.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        from .frontend import prepare

        print(prepare(args.recording, args.output, seed=args.seed))
    elif args.command == "refine":
        from .backend import run_backend

        print(run_backend(args.packet, args.output, Config.load(args.config)))
    elif args.command == "run":
        from .pipeline import run

        print(run(args.recording, args.output, Config.load(args.config), seed=args.seed))
    else:
        import json

        print(json.dumps(Config.load(args.path).native_environment(), indent=2))


if __name__ == "__main__":
    main()
