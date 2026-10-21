#!/usr/bin/env bash
# AMO V1 levels 4->6, then V2 adaptive fallback from base level 5.
set -euo pipefail
mkdir -p certificates logs cache
radius_exp=${1:-24}
workers=${2:-16}
shared_cache=${3:-cache/AMO_shared_ray_cache_r24.json}
for level in 4 5 6; do
  cert="certificates/AMO_k${level}.json"
  ck="certificates/AMO_k${level}.checkpoint.json"
  echo "$(date -u '+%FT%TZ') [AMO] V1 level $level"
  python3 -u generate_amo_certificate.py --level "$level" --radius-exp "$radius_exp" \
    --workers "$workers" --shared-cache "$shared_cache" --out "$cert" --checkpoint "$ck" 2>&1 | tee "logs/AMO_k${level}_generate.log"
  python3 verify_certificate.py "$cert" 2>&1 | tee "logs/AMO_k${level}_verify.log"
  if python3 check_threshold.py "$cert"; then
    echo "$(date -u '+%FT%TZ') [AMO] COMPLETE at V1 level $level"; exit 0
  fi
done

echo "$(date -u '+%FT%TZ') [AMO] switching to V2 adaptive"
python3 -u generate_amo_adaptive_certificate.py --base-level 5 --fraction 0.20 \
  --max-rounds 6 --radius-exp "$radius_exp" --workers "$workers" \
  --shared-cache "$shared_cache" --out certificates/AMO_adaptive.json \
  --checkpoint certificates/AMO_adaptive.checkpoint.json \
  2>&1 | tee logs/AMO_adaptive_generate.log
python3 verify_certificate.py certificates/AMO_adaptive.json \
  2>&1 | tee logs/AMO_adaptive_verify.log
python3 check_threshold.py certificates/AMO_adaptive.json
