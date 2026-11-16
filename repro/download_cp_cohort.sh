#!/usr/bin/env bash
# Download the raw .vrs, Aria calibration, control points and pseudo-dense GT for the
# 13-recording LaMAria training CP cohort (public archive, no login, ~92 GB of .vrs)
# into $DATA_ROOT/raw/<seq>/. Resumable. usage: download_cp_cohort.sh [seq ...]
set -uo pipefail
source "$(dirname -- "$0")/env.sh"
B=https://cvg-data.inf.ethz.ch/lamaria
SEQS=("$@"); [ $# -eq 0 ] && SEQS=($CP_COHORT)
for s in "${SEQS[@]}"; do
  d=$DATA_ROOT/raw/$s; mkdir -p "$d"
  wget -q -c -O "$d/calibration.json" "$B/aria_calibrations/training/$s.json"
  wget -q -c -O "$d/control_points.json" "$B/ground_truth/sparse/$s.json"
  wget -q -c -O "$d/pgt.txt" "$B/ground_truth/pseudo_dense/$s.txt" || rm -f "$d/pgt.txt"  # none for 5_11, 5_12
  [ -f "$d/.vrs_ok" ] && continue
  echo "$(date -Is) start $s"
  for try in 1 2 3; do wget -q -c -O "$d/$s.vrs.part" "$B/raw_data/training/$s.vrs" && break; sleep 10; done
  remote=$(curl -sIL "$B/raw_data/training/$s.vrs" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r' | tail -1)
  if [ "$remote" = "$(stat -c %s "$d/$s.vrs.part")" ]; then mv "$d/$s.vrs.part" "$d/$s.vrs"; touch "$d/.vrs_ok"; echo "$(date -Is) done $s"
  else echo "$(date -Is) SIZE MISMATCH $s"; fi
done
