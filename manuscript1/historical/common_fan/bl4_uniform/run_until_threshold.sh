#!/usr/bin/env bash
set -eu
body=${1:?usage: run_until_threshold.sh BODY [START_LEVEL] [MAX_LEVEL]}
start=${2:-6}
maximum=${3:-7}
mkdir -p certificates logs
level=$start
while [ "$level" -le "$maximum" ]; do
  cert="certificates/${body}_k${level}.json"
  echo "[$body] generating level $level"
  python3 generate_certificate.py --body "$body" --level "$level" --out "$cert" \
    2>&1 | tee "logs/${body}_k${level}_generate.log"
  python3 verify_certificate.py "$cert" \
    2>&1 | tee "logs/${body}_k${level}_verify.log"
  if python3 check_threshold.py "$cert"; then
    echo "[$body] strict threshold crossed at level $level"
    exit 0
  fi
  level=$((level+1))
done
echo "[$body] no strict crossing through level $maximum" >&2
echo "Do not start a global level-8 run blindly; inspect witness/oracle tightness and use adaptive refinement." >&2
exit 2

