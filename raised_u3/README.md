# Raised U3 (auxiliary)

This directory contains an auxiliary validated computation for the
canonically raised `U_3`. **It is not used in the certified comparison**
`Vol_4(U4^(1)) < Vol_4(K_RM)` documented at the repository root.

## Files

| File | Role |
|---|---|
| `src/raised_u3_certify_centres.py` | Interval-arithmetic verifier for the raised-`U_3` center certificate. Uses only `mp.iv.mpf`, four basic operations, `mp.iv.sqrt`, and three-valued interval comparison — no `findroot`, numerical integration, trigonometric functions, or other higher-level `iv` algorithms. |

## Status

Only the source script is included here for now. The corresponding
certificate output/log found alongside it in the working directory
(`raised_u3_certify_centres.log`, `raised_u3_s4_centres_certificate.json`)
predate the current version of the script (log/certificate timestamped
2026-08-02, script last modified 2026-08-03), so they are not included
until re-run against the current script and re-verified. This will be
added as `certificates/` once regenerated and hash-checked.
