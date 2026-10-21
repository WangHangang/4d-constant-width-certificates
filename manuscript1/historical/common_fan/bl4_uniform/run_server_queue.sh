#!/usr/bin/env bash
# Supervisor-facing unattended queue.  Large AMO computation remains separate
# because it uses the witness generator and checkpoint/resume.
set -u
set -o pipefail
mkdir -p certificates logs
workers=${WORKERS:-16}
queue_log="logs/server_queue_progress.log"
announce(){ printf '%s [queue] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$queue_log"; }
run_body(){
  body=$1; fixed=$2; adaptive_base=$3
  announce "STARTING $body"
  rounds=${4:-6}
  bash run_progressive.sh "$body" "$fixed" "$adaptive_base" "$rounds" 0.15 "$workers"
  code=$?
  if [ "$code" -eq 0 ]; then announce "FINISHED $body: certified above threshold"
  elif [ "$code" -eq 2 ]; then announce "FINISHED $body: needs review; continuing queue"
  else announce "ABORTING queue: $body failed with exit code $code"; exit "$code"; fi
}

run_body M4 3 3 0
run_body RaisedMeissnerFace 6 5 8
run_body Body5 6 5 8
# RaisedU3 is proof-ready, but adaptive refinement is likely preferable to a
# memory-heavy global level 7 if the fixed run is close.
run_body RaisedU3 6 5 10

announce "PREDICATE-ADAPTER QUEUE COMPLETE"
announce "NEXT: run AMO with run_amo_progressive.sh (checkpointed witness generator)"
