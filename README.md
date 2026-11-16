> **About this fork/branch.** SLAM Dunk is by **Jacob Van Meter and Ayush Paul**
> ([paper](https://www.alphaxiv.org/abs/2611.slam-dunk), [upstream code](https://github.com/jacobvm04/slam-dunk)).
> This `pixi` branch adds nothing to the method: it only adds a [pixi](https://pixi.sh) build
> (instead of Docker) and an independent reproduction of the paper's LaMAria training-set results
> ([REPRO.md](REPRO.md)). The upstream code is unchanged and its license
> ([AGPL-3.0-only](LICENSE)) and copyright are retained.

# SLAM Dunk

**Jacob Van Meter · Ayush Paul**

Standard VIO systems struggle to maintain accuracy with long trajectories in the wild, especially in egocentric settings, as evidenced by the gap between prior art and industry systems on the LaMAria benchmark. We find that a conventional offline bundle adjustment pipeline closes much of this gap, but plateaus at the level of strong public submissions, well below Aria's SLAM. We hypothesize that the remaining gap occurs because standard VIO models omit two physical phenomena: camera calibration drift and gyroscope intrinsic calibration. Modeling both in such a pipeline raises LaMAria's control-point score from 69.1 to 82.4 on the training set, and the resulting system, SLAM Dunk, matches Aria's SLAM on the LaMAria test set and is the first to score above it on three of five categories; it also ranks second on the UZH-FPV leaderboard.

## Reproduce with pixi

Linux x86-64-v3 (AVX2), [pixi](https://pixi.sh) ≥ 0.83. No Docker, venv or system packages are needed.
Environments: `default` (inference), `native` (C++ build), `extract` (VRS extraction),
`metrics` (Evo), `cp-eval` (the official LaMAria evaluator and its pycolmap 3.13 wheel),
`viz` (optional Rerun logging).

```sh
pixi install --locked -a          # all environments
pixi run -e native build-native   # aria_vio, libtrack_union.so, batch_joint (-march=x86-64-v3)
pixi run test                     # unit tests
```

Data (public, no login): the 13 LaMAria training recordings with control points. That is
about 92 GB of `.vrs` from <https://cvg-data.inf.ethz.ch/lamaria/>, plus the Aria calibration,
control points and pseudo-GT for each recording. `DATA_ROOT` (default `./data/lamaria`) and
`RUNS_ROOT` (default `./results/runs`) set where things go; see `repro/env.sh`.

```sh
pixi run download-cp-cohort                          # or: ... download-cp-cohort R_11_5cp
pixi run repro-seq R_11_5cp full baseline            # extract, one frontend packet, refine both, score
pixi run repro-cohort 2 full baseline                # all 13 recordings, 2 at a time
pixi run repro-cohort 2 camera_only gyro_only        # reuses the packets
pixi run repro-table results/runs results/repro-table   # ours vs paper table (markdown/html/json)
```

Each stage runs under `systemd-run --user --scope -p CPUQuota=800% -p MemoryMax=8G` (the paper's
`docker run --cpus 8 --memory 8g`); set `SLAM_DUNK_SCOPE=0` to run without it.

Expected mean CP over the 13 recordings (one run each; see [REPRO.md](REPRO.md) for every recording,
the removal ablations, the two gaps and the hardware caveats):

| config | ours | paper |
|---|---:|---:|
| baseline | 68.8 | 69.14 |
| gyro_only | 71.7 | 71.47 |
| camera_only | 77.1 | 77.13 |
| full | 82.9 | 82.42 |

The backend has wall-clock budgets: on a slow or busy machine, check each run's `receipt.json`
(`native_timeout`, `restored_base`) before comparing numbers.

## Run (upstream, Docker)

Download Raw data (.vrs) and Aria calibration (.json) for the same sequence
from [LaMAria](https://lamaria.ethz.ch/slam_datasets), saving them as `recording.vrs`
and `calibration.json`. Prepare the recording with Python 3.12:

```sh
python3.12 -m venv .venv-extract
.venv-extract/bin/pip install -r tools/requirements.txt
.venv-extract/bin/python tools/extract_recording.py recording.vrs data/recording --calibration calibration.json
```

Then run with Docker on Linux x86-64-v3 (AVX2):

```sh
docker build -t slam-dunk .
mkdir -p results
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --cpus 8 --memory 8g --memory-swap 8g \
  --read-only --tmpfs /tmp:rw,exec,size=512m \
  -v "$PWD/data/recording:/recording:ro" \
  -v "$PWD/results:/results" \
  slam-dunk run /recording /results/full --config configs/full.json
```

Output: `results/full/estimate/trajectory.txt`. Use a new output directory for each run.

## Ablations

Select a model with `--config`. The camera/gyro comparison uses
[`full.json`](configs/full.json), [`baseline.json`](configs/baseline.json),
[`camera_only.json`](configs/camera_only.json), and [`gyro_only.json`](configs/gyro_only.json).
Additional ablations are in [`configs/`](configs/). The `prepare` and `refine`
commands let multiple models reuse one frontend packet; see `--help`.

## Evaluation

See [`evaluation/`](evaluation/README.md) for scoring commands and dependencies.

## License

Original project code is licensed under [AGPL-3.0-only](LICENSE).
Copyright © 2026 Jacob Van Meter and Ayush Paul. Third-party licenses are retained.
For alternative commercial licensing, contact [jacobvm04@gmail.com](mailto:jacobvm04@gmail.com).
