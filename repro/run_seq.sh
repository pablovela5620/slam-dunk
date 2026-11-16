#!/usr/bin/env bash
# One LaMAria recording: extract (if needed) -> one frontend packet -> refine each config in
# parallel from that packet -> official CP score (+ Evo ATE where a public pGT exists) ->
# CP geometry dump. usage: run_seq.sh <seq> [config ...]   (default: full baseline)
# Each stage's timing.tsv row carries the 1/5/15-min load average.
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
S=$1; shift; CONFIGS=("$@"); [ $# -eq 0 ] && CONFIGS=(full baseline)
RAW=$DATA_ROOT/raw/$S; PREP=$DATA_ROOT/prepared/$S; RUN=$RUNS_ROOT/$S
mkdir -p "$RUN"; cd "$REPO"
stamp() { printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$S" "$1" "$(cut -d' ' -f1-3 /proc/loadavg)" >> "$RUN/timing.tsv"; }
if [ ! -f "$PREP/extraction.json" ]; then
  stamp extract-start; rm -rf "$PREP"
  pixi run -e extract python tools/extract_recording.py "$RAW/$S.vrs" "$PREP" --calibration "$RAW/calibration.json" > "$RUN/extract.log" 2>&1
  stamp extract-end
fi
if [ ! -f "$RUN/packet/receipt.json" ] || ! grep -q '"status": "complete"' "$RUN/packet/receipt.json"; then
  rm -rf "$RUN/packet"; stamp prepare-start
  scope pixi run python -m slam_dunk prepare "$PREP" "$RUN/packet" > "$RUN/prepare.log" 2>&1
  stamp prepare-end
fi
PACKET=$RUN/packet/work
pids=()
for c in "${CONFIGS[@]}"; do
  [ -f "$RUN/$c/trajectory.txt" ] && continue
  rm -rf "$RUN/$c"
  ( stamp "refine-$c-start"; scope pixi run python -m slam_dunk refine "$PACKET" "$RUN/$c" --config "configs/$c.json" > "$RUN/refine-$c.log" 2>&1; stamp "refine-$c-end rc=$?" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
CPPY=$REPO/.pixi/envs/cp-eval/bin/python
for c in "${CONFIGS[@]}"; do
  [ -f "$RUN/$c/trajectory.txt" ] || { echo "no trajectory for $c"; continue; }
  rm -f "$RUN/score-$c.json"
  extra=(--public-control-points "$RAW/control_points.json")
  [[ $S == sequence_* && -f $RAW/pgt.txt ]] && extra+=(--public-pgt "$RAW/pgt.txt")
  stamp "score-$c-start"
  pixi run -e metrics python evaluation/score_lamaria.py "$PREP" "$RUN/$c/trajectory.txt" "$RUN/score-$c.json" \
    --cp-python "$CPPY" "${extra[@]}" > "$RUN/score-$c.log" 2>&1 || echo "score failed $c"
  stamp "score-$c-end"
done
# Official-evaluator geometry (Sim(3), triangulated CPs), incl. the frontend's VIO seed.
for c in vio "${CONFIGS[@]}"; do
  est=$RUN/$c/trajectory.txt; [ "$c" = vio ] && est=$RUN/packet/frontend/vio.txt
  [ -f "$est" ] && [ ! -f "$RUN/cp-$c.json" ] && pixi run -e cp-eval python repro/cp_dump.py "$est" "$RAW/control_points.json" "$PREP/calibration.json" "$RUN/cp-$c.json" > "$RUN/cp-$c.log" 2>&1
done
grep -H cp_score "$RUN"/score-*.json || true
