#!/usr/bin/env bash
# Run the CP cohort with N parallel workers (waits for each download to finish).
# usage: run_cohort.sh <workers> [config ...]   e.g. run_cohort.sh 2 full baseline
set -uo pipefail
source "$(dirname -- "$0")/env.sh"
W=$1; shift; CONFIGS=("$@"); [ $# -eq 0 ] && CONFIGS=(full baseline)
QUEUE=${QUEUE:-$CP_COHORT}
LOGS=$RUNS_ROOT/logs; CLAIMS=$LOGS/claims; mkdir -p "$CLAIMS"
tag=$(echo "${CONFIGS[*]}" | tr ' ' _)
worker() {
  for s in $QUEUE; do
    mkdir "$CLAIMS/$s.$tag" 2>/dev/null || continue   # atomic claim
    until [ -f "$DATA_ROOT/raw/$s/.vrs_ok" ]; do sleep 30; done
    echo "$(date -Is) worker$1 start $s"
    "$REPO/repro/run_seq.sh" "$s" "${CONFIGS[@]}" > "$LOGS/run-$s-$tag.log" 2>&1
    echo "$(date -Is) worker$1 end $s rc=$?"
  done
}
for i in $(seq 1 "$W"); do worker "$i" & done
wait
echo "$(date -Is) COHORT DONE ${CONFIGS[*]}"
