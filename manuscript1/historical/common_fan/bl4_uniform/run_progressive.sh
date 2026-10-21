#!/usr/bin/env bash
# Unattended fixed-fan -> adaptive certificate pipeline for one body.
set -euo pipefail

body=${1:?usage: run_progressive.sh BODY [FIXED_LEVEL] [ADAPTIVE_BASE] [ADAPTIVE_ROUNDS] [FRACTION] [WORKERS]}
fixed_level=${2:-6}
adaptive_base=${3:-5}
adaptive_rounds=${4:-6}
fraction=${5:-0.15}
workers=${6:-16}

mkdir -p certificates logs
progress_log="logs/${body}_progress.log"
status_file="logs/${body}_STATUS.txt"

progress() {
  timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  message="$timestamp [$body] $*"
  printf '%s\n' "$message" | tee -a "$progress_log"
  printf '%s\n' "$message" > "$status_file"
}

failed() {
  code=$?
  if [ "$code" -ne 0 ]; then
    progress "FAILED with exit code $code; inspect the latest log"
  fi
}
trap failed EXIT

fixed_cert="certificates/${body}_k${fixed_level}.json"
progress "STEP 1/2: entering fixed level $fixed_level generation"
python3 -u generate_certificate.py --body "$body" --level "$fixed_level" \
  --workers "$workers" --out "$fixed_cert" 2>&1 | tee "logs/${body}_k${fixed_level}_generate.log"
progress "STEP 1/2: entering independent verification"
python3 -u verify_certificate.py "$fixed_cert" \
  2>&1 | tee "logs/${body}_k${fixed_level}_verify.log"
if python3 check_threshold.py "$fixed_cert"; then
  progress "COMPLETE: strict threshold crossed at fixed level $fixed_level"
  trap - EXIT
  exit 0
fi

adaptive_cert="certificates/${body}_adaptive_b${adaptive_base}_r${adaptive_rounds}.json"
progress "ENTERING NEXT STEP 2/2: adaptive refinement, base=$adaptive_base, fraction=$fraction, max_rounds=$adaptive_rounds"
python3 -u generate_adaptive_certificate.py --body "$body" \
  --base-level "$adaptive_base" --fraction "$fraction" \
  --max-rounds "$adaptive_rounds" --workers "$workers" --out "$adaptive_cert" \
  2>&1 | tee "logs/${body}_adaptive_generate.log"
progress "STEP 2/2: entering independent adaptive verification"
python3 -u verify_certificate.py "$adaptive_cert" \
  2>&1 | tee "logs/${body}_adaptive_verify.log"
if python3 check_threshold.py "$adaptive_cert"; then
  progress "COMPLETE: strict threshold crossed during adaptive refinement"
  trap - EXIT
  exit 0
fi

progress "NEEDS REVIEW: configured adaptive rounds finished below the threshold; inspect oracle tightness before allocating more memory"
trap - EXIT
exit 2
