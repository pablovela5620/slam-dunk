# Reproduction of SLAM Dunk on the LaMAria training set (pixi build)

This file reports an independent reproduction of the LaMAria training-set results of
*SLAM Dunk: A recipe for long horizon egocentric SLAM in the wild* (Jacob Van Meter, Ayush Paul;
[paper](https://www.alphaxiv.org/abs/2611.slam-dunk),
[upstream code](https://github.com/jacobvm04/slam-dunk)). We ran the authors' code from the
`pixi` build in this branch, not their Docker image. Everything below is one run per recording;
we did not sweep seeds.

**Result:** the headline, the camera/gyro 2×2 factorial and the paper's Fig. 6 "remove one term"
ablations all reproduce on the full 13-recording control-point (CP) cohort.

## Pins

| item | value |
|---|---|
| upstream code | `jacobvm04/slam-dunk` @ `a7aaf6d` (this branch only adds the pixi build and `repro/`) |
| official evaluator | frozen copy in `evaluation/` (LaMAria evaluator at `238b6cad307a10183277388c9c0a8406cde48197`, its pycolmap 3.13.0.dev0 wheel) |
| environments | `pixi.lock` in this branch (pixi 0.83); deviations from `requirements-full.lock` / `apt-full.lock` are listed below |
| data | public LaMAria archive <https://cvg-data.inf.ethz.ch/lamaria/>: `raw_data/training/*.vrs`, `aria_calibrations/training/*.json`, `ground_truth/sparse/*.json`, `ground_truth/pseudo_dense/*.txt` |
| paper reference values | Table 1 (13-recording means); per-recording baseline/full values digitized from Fig. 4c (the paper has no per-recording table; read precision ±0.1 CP; their means reproduce Table 1 exactly) |

## Protocol

- Cohort: the 13 training recordings that have control points (the paper's CP cohort): `R_11_5cp R_12_10cp R_13_15cp sequence_1_19 sequence_1_20 sequence_2_11 sequence_2_12 sequence_3_17 sequence_3_18 sequence_4_10 sequence_4_11 sequence_5_11 sequence_5_12`.
- Per recording, `prepare` builds one frontend packet; every config `refine`s that same packet (the paper's matched-packet protocol).
- Scoring: `evaluation/score_lamaria.py` with `--public-control-points`. For `sequence_*` it also takes `--public-pgt`, which converts the public camera-0 pGT to the right-IMU frame for ATE. CP is the official triangulate → Sim(3) → piecewise score. ATE is unit-scale SE(3) (Evo).
- Resources: each stage ran under `systemd-run --scope -p CPUQuota=800% -p MemoryMax=8G`, the equivalent of the paper's `docker run --cpus 8 --memory 8g`. The thread variables `OPENBLAS/OMP/MKL/NUMBA_NUM_THREADS=1` match the Dockerfile.

## 1. Headline and 2×2 factorial: mean CP over the 13 recordings

| config | camera calibration | gyro matrix G | ours | paper (Table 1) |
|---|---|---|---:|---:|
| `baseline` | static | off | **68.8** | 69.14 |
| `gyro_only` | static | on | **71.7** | 71.47 |
| `camera_only` | 5 s chunks | off | **77.1** | 77.13 |
| `full` | 5 s chunks | on | **82.9** | 82.42 |

## 2. Per recording (CP score, higher is better)

"VIO seed" is the frontend's Basalt-style VIO before any offline BA, scored with the same evaluator. Paper values come from Fig. 4c.

| recording | VIO seed | baseline (ours) | baseline (paper) | Δ | gyro_only | camera_only | full (ours) | full (paper) | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R_11_5cp | 44.5 | 87.0 | 86.2 | +0.8 | 91.6 | 97.7 | 96.9 | 97.4 | −0.5 |
| R_12_10cp | 32.3 | 81.2 | 81.1 | +0.1 | 80.6 | 78.5 | 82.1 | 82.1 | +0.0 |
| R_13_15cp | 39.2 | 76.0 | 83.0 | **−7.0** | 81.6 | 76.4 | 91.0 | 88.3 | +2.7 |
| sequence_1_19 | 48.5 | 81.0 | 81.1 | −0.1 | 89.7 | 83.8 | 94.4 | 94.3 | +0.1 |
| sequence_1_20 | 33.7 | 95.1 | 94.1 | +1.0 | 94.9 | 90.8 | 91.0 | 89.4 | +1.6 |
| sequence_2_11 | 49.9 | 84.4 | 83.7 | +0.7 | 86.3 | 96.4 | 95.3 | 95.5 | −0.2 |
| sequence_2_12 | 33.2 | 43.3 | 43.0 | +0.3 | 52.3 | 58.7 | 73.4 | 70.3 | **+3.1** |
| sequence_3_17 | 27.5 | 52.7 | 52.6 | +0.1 | 61.2 | 64.3 | 87.3 | 87.6 | −0.3 |
| sequence_3_18 | 31.7 | 63.2 | 63.1 | +0.1 | 65.1 | 85.2 | 88.6 | 88.8 | −0.2 |
| sequence_4_10 | 28.6 | 81.2 | 81.4 | −0.2 | 82.8 | 87.9 | 90.7 | 91.1 | −0.4 |
| sequence_4_11 | 21.9 | 61.0 | 60.9 | +0.1 | 60.4 | 90.6 | 93.6 | 93.3 | +0.3 |
| sequence_5_11 | 22.2 | 81.8 | 81.5 | +0.3 | 79.3 | 85.9 | 87.5 | 87.4 | +0.1 |
| sequence_5_12 | 3.8 | 7.0 | 7.0 | +0.0 | 6.8 | 5.8 | 5.9 | 6.0 | −0.1 |
| **mean (13)** | 32.1 | **68.8** | 69.14 | | **71.7** | **77.1** | **82.9** | 82.42 | |

- 11 of 13 recordings are within ±1 CP of the paper for both baseline and full.
- That includes the paper's two odd cases:
  - sequence_1_20, where full is *worse* than baseline;
  - sequence_5_12, which neither config solves (its CP Sim(3) needs a scale of 0.92).

ATE in metres (unit-scale SE(3)). The wrapper computes it only for the natural recordings with a public pGT; 5_11 and 5_12 have none.

| recording | baseline | full |
|---|---:|---:|
| sequence_1_19 | 0.527 | 0.179 |
| sequence_1_20 | 0.272 | 0.228 |
| sequence_2_11 | 0.669 | 0.276 |
| sequence_2_12 | 2.330 | 0.713 |
| sequence_3_17 | 2.307 | 0.785 |
| sequence_3_18 | 2.382 | 0.425 |
| sequence_4_10 | 1.657 | 0.620 |
| sequence_4_11 | 1.506 | 0.317 |
| **mean (8)** | **1.46** | **0.44** |

The paper's ATE means are over 21 recordings (1.018 → 0.430 m), so these 8 are not directly comparable. The ratio between configs is.

## 3. Removing one term from `full` (paper Fig. 6)

Mean CP lost against `full` over the 13 recordings:

| config | term removed | ours | paper |
|---|---|---:|---:|
| `no_accelerometer` | accelerometer scale, cross-axis and alignment | 3.8 | 3.3 |
| `no_clock` | camera–IMU clock offsets | 2.2 | 2.3 |
| `no_thermal` | thermal gain and bias terms | 0.5 | 0.5 |
| `no_auxiliary` | all three groups above | 5.8 | 5.6 |
| `no_final_refinement` | the third, calibration-frozen pass | 1.2 | not reported |

Per recording (CP):

| recording | full | no_accelerometer | no_clock | no_thermal | no_auxiliary | no_final_refinement |
|---|---:|---:|---:|---:|---:|---:|
| R_11_5cp | 96.9 | 95.8 | 95.6 | 96.7 | 93.9 | 96.6 |
| R_12_10cp | 82.1 | 81.0 | 81.8 | 82.0 | 81.0 | 80.8 |
| R_13_15cp | 91.0 | 82.1 | 91.2 | 90.1 | 77.8 | 88.8 |
| sequence_1_19 | 94.4 | 94.4 | 93.9 | 92.8 | 92.6 | 92.3 |
| sequence_1_20 | 91.0 | 89.8 | 90.1 | 91.3 | 89.4 | 89.8 |
| sequence_2_11 | 95.3 | 93.1 | 96.6 | 94.6 | 94.9 | 94.7 |
| sequence_2_12 | 73.4 | 71.0 | 68.8 | 73.4 | 71.9 | 72.9 |
| sequence_3_17 | 87.3 | 69.6 | 71.7 | 86.8 | 54.9 | 86.2 |
| sequence_3_18 | 88.6 | 85.9 | 83.5 | 84.9 | 79.8 | 85.8 |
| sequence_4_10 | 90.7 | 88.1 | 89.2 | 91.5 | 86.9 | 90.5 |
| sequence_4_11 | 93.6 | 83.6 | 94.7 | 93.6 | 87.6 | 91.8 |
| sequence_5_11 | 87.5 | 87.8 | 85.8 | 87.9 | 85.6 | 86.1 |
| sequence_5_12 | 5.9 | 5.8 | 5.7 | 5.9 | 6.1 | 5.6 |
| **mean (13)** | **82.9** | **79.1** | **80.7** | **82.4** | **77.1** | **81.7** |

## 4. Gaps

Two recordings differ from the paper by more than ±1 CP:

| recording | config | ours | paper | Δ |
|---|---|---:|---:|---:|
| R_13_15cp | baseline | 76.0 | 83.0 | −7.0 |
| sequence_2_12 | full | 73.4 | 70.3 | +3.1 |

What we checked:

- **No budget was hit.** In both runs (and in every cohort run):
  - no native timeout and no restored base state;
  - no SIFT pair skipped at its deadline;
  - every solver pass ended on convergence, the pose-step rule or the 250-iteration cap, never on time.
- **Not run-to-run noise.** We repeated R_13_15cp twice:
  - The same packet refined again with `baseline` gives 75.95, the same to 0.01.
  - A fresh frontend `prepare` gives 75.97 baseline and 90.97 full; its VIO seed differs only at float precision.
  - So the pipeline repeats within ±0.05 CP here, and the gap is systematic.
- **Error spread.** The R_13 baseline errors are spread over all 15 points: 0.1–1.0 m horizontally, against 0.1–0.35 m for full on the same packet. They sit on the steep part of the CP curve, where 0.3 m costs a point about 15 score.

**Unexplained.** The most likely sources are build differences that change the frontend features and the BA seed:

- OpenCV 4.13 (conda-forge) instead of 4.14 (SIFT, RANSAC);
- `aria_vio` and `libtrack_union` compiled with `-march=native` on Zen 5 instead of the paper's Zen 4.

The static-calibration baseline is the model most sensitive to such differences.

## 5. The two physical effects, as measured by `full`

These come from the saved states (`ba.txt.focals`, `ba.txt.diag`, `ba.txt.gyromat`; `repro/physics_summary.py`):

- **Focal length:** follows the camera-record temperature at 0.02–0.06 px/°C (paper medians 0.028 and 0.036), over 12–22 °C swings per walk.
- **Stereo rotation:** moves 1.6–3.7 mrad from its start (paper: 1–6 mrad).
- **Gyro matrix G:** groups by device. Its diagonals cluster into four groups: the `R_1x` walks; 1_19/1_20; 2_12/3_17/3_18; 4_10/4_11. This is our inference from the values, not from device IDs. It is consistent with the paper's claim that G is a property of the unit.

## 6. Hardware, load and timing caveats

- **CPU:** one AMD Ryzen 9 9950X3D (16 cores / 32 threads), Ubuntu 24.04. The paper used a Ryzen 9 7950X3D.
- **Shared host:** the host ran other jobs during the whole cohort. The 1-minute load average was 30–86, so each backend effectively got 2–3 of its 6 threads. Every stage's start and end are logged with the load average (`<run>/timing.tsv`).
- **Backend wall time:** 130–930 s per refine (paper: median 2.0 min, max 5.5 min on an idle machine).
- **Frontend wall time:** 1.5–5 min per recording.
- **Wall-clock budgets:** the backend has fixed budgets (598 s per pass, 1197 s in total; the SIFT matching deadline is 498 s). On a slower or busier machine these can bind and change results. Check `receipt.json` (`native_timeout`, `restored_base`, `final_selections`) and `packet/frontend/sift-health.json` (`deadline_skipped_pairs`) before comparing numbers.

## 7. Deviations from the upstream locks

| package | upstream lock | this branch | why |
|---|---|---|---|
| OpenCV (Python) | opencv-contrib-python-headless 4.14.0.94 | py-opencv 4.13.0 (conda-forge) | conda-forge first; only SIFT, PnP, findEssentialMat and imread are used |
| OpenCV (C++) | libopencv-dev 4.6.0 | libopencv 4.12 | the conda-forge 4.x that co-solves with Ceres 2.2 |
| numba | 0.68.0 | 0.68.1 | conda-forge patch release |
| fmt / tbb / glog | 9.1 / 2021.11 / 0.6 | 12.2 / 2021.13 / 0.7.1 | conda-forge builds; glog 0.7 needs `-DGLOG_USE_GLOG_EXPORT` |
| Eigen | 3.4.0 | 3.4.0 in `native`, 5.0.1 in `default` | conda-forge pycolmap 4.2.1 needs Eigen 5, so the C++ toolchain has its own `native` env (binaries carry an RPATH into it) |
| g++ | 13.3 (apt) | gxx 13 (conda-forge) | conda's default CXXFLAGS are dropped so that only the project's flags apply; `-pthread` added |
| projectaria-tools, vrs | PyPI | PyPI | not on conda-forge |
| evaluator env | full venv freeze | pycolmap wheel + pyceres 2.6 + evo 1.31.1 + projectaria-tools from PyPI | the LaMAria pycolmap build exists only as a wheel; pyceres must match its bundled Ceres; jupyter etc. are not needed |
