# Shared settings for the reproduction scripts. Override any of these in the environment.
#   DATA_ROOT        LaMAria data: $DATA_ROOT/raw/<seq>/ (downloads), $DATA_ROOT/prepared/<seq>/ (extracted)
#   RUNS_ROOT        per-sequence outputs: $RUNS_ROOT/<seq>/{packet,<config>,score-<config>.json,...}
#   SLAM_DUNK_SCOPE  1 (default) = run each stage in a systemd user scope with CPUQuota=800% and
#                    MemoryMax=8G, mirroring the paper's `docker run --cpus 8 --memory 8g`; 0 = no cap
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-$REPO/data/lamaria}
RUNS_ROOT=${RUNS_ROOT:-$REPO/results/runs}
SLAM_DUNK_SCOPE=${SLAM_DUNK_SCOPE:-1}
# The 13 LaMAria training recordings with control points (the paper's CP cohort), smallest first.
CP_COHORT="R_11_5cp sequence_1_19 R_12_10cp sequence_2_11 sequence_1_20 sequence_5_11 sequence_5_12 sequence_4_11 sequence_2_12 R_13_15cp sequence_4_10 sequence_3_18 sequence_3_17"
scope() {
  if [ "$SLAM_DUNK_SCOPE" = 1 ] && command -v systemd-run > /dev/null; then
    systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=8G -p MemorySwapMax=0 -- "$@"
  else
    "$@"
  fi
}
