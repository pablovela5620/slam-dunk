# LaMAria evaluation

Scoring is separate from estimation. No evaluator, reference trajectory, or control
point is imported by `slam_dunk`. These sources preserve the original benchmark
wrapper and official LaMAria evaluator at commit
`238b6cad307a10183277388c9c0a8406cde48197`. Original third-party licenses
are retained alongside the bundled evaluator source.

Use two separate CPython 3.12 environments because inference PyCOLMAP 4.2.1 and
the official scorer's PyCOLMAP 3.13 build are different APIs:

```sh
python3.12 -m venv /tmp/slam-dunk-metrics
/tmp/slam-dunk-metrics/bin/pip install -r evaluation/metrics-requirements.lock
python3.12 -m venv /tmp/slam-dunk-cp
/tmp/slam-dunk-cp/bin/pip install -r evaluation/official-evaluator-requirements.lock

/tmp/slam-dunk-metrics/bin/python evaluation/score_lamaria.py \
  /recording /estimate/trajectory.txt /new/metrics.json \
  --cp-python /tmp/slam-dunk-cp/bin/python
```

`/recording` must contain `cam0/data.csv`, `calibration.json`, and available
`groundtruth.txt` / `control_points.json` in the **prepared right-IMU frame**.
Missing labels produce null metrics, never invented zeroes. `--no-cp` explicitly
skips CP. The CP environment lock includes a public CPython 3.12 Linux x86-64
wheel URL for the official evaluator's PyCOLMAP build.

For author-provided **natural recording** reference files, add
`--public-pgt /labels/sequence_1_19.txt` and
`--public-control-points /labels/sequence_1_19.json`. This converts the public
camera0 reference to the right-IMU body frame before scoring. Merely renaming the
public reference to `groundtruth.txt` gives the wrong body frame. The wrapper
requires this explicit mode for `sequence_3_17`, whose historical training
annotation was mounted separately. Use the annotation convention that matches your prepared recording.

CP is the official triangulation / similarity alignment / surveyed-point score.
ATE is unit-scale SE(3) Evo error; coverage is reported separately. For the 23-record training cohort, CP has 13 recordings and ATE has 21.
Keep fixed cohorts when averaging.
The scoring wrapper does not submit anything to a leaderboard.
