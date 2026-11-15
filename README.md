# SLAM Dunk

**Jacob Van Meter · Ayush Paul**

Standard VIO systems struggle to maintain accuracy with long trajectories in the wild, especially in egocentric settings, as evidenced by the gap between prior art and industry systems on the LaMAria benchmark. We find that a conventional offline bundle adjustment pipeline closes much of this gap, but plateaus at the level of strong public submissions, well below Aria's SLAM. We hypothesize that the remaining gap occurs because standard VIO models omit two physical phenomena: camera calibration drift and gyroscope intrinsic calibration. Modeling both in such a pipeline raises LaMAria's control-point score from 69.1 to 82.4 on the training set, and the resulting system, SLAM Dunk, matches Aria's SLAM on the LaMAria test set and is the first to score above it on three of five categories; it also ranks second on the UZH-FPV leaderboard.

## Run

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
