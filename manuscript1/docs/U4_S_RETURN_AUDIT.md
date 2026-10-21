# U4-S: returned production certificate and verification scope

## Provenance

Andriy Prymak supplied the production return archive on 21 October 2026.
The archive SHA-256 is
`508c95230a2644972bc7b7da9ecc25b581b7d75b4c4557b55a6bba96cdc1921b`.
Its five computational files are preserved byte for byte under
`certificates/u4_s_20261021/`; individual hashes are in `manifest.json`
and the original `RETURN_MANIFEST.json`.

This is a new full run, starting at row 0, not recovery of the historical
JSON with digest `b787db8a5fadc05b895e8c2b57567fd19139fb4982cbacb19f4f17a0c426f1b7`.
The current certificate digest is
`a5b0a863c50de0a262e6596df7098b93e6955a3232f4aa4c743b0b3c0896bd5f`;
the completed checkpoint digest is
`5ccfe5a3c878044d703707aa74464630adb07dcf9ad5ce4e5410b807b2324ebb`.

## Computation and result

The run used Python 3.14.7, python-flint 0.9.0, 50 decimal digits, a
4096-by-4096 base grid, and 4-by-4 refinement of fallback cells. The production
log reports 4971.08246421814 seconds (about 83 minutes), all 4096 rows complete,
and the six-decimal success token. The exact counters cover 16,777,216 base
cells, including 506,466 refined base cells and 8,103,456 refinement subcells.

The ten-piece volume computation freshly evaluates six 1D pieces and the
adaptive I2_04 piece. Three other 2D intervals (I2_01, I3_01, I3_02) come from
the preserved U4-G records. All four frozen driver/dependency hashes agree.

The outward decimal enclosure is

    0.2218357659033942 < Vol_4(U4^(1)) < 0.2218363538061581.

Both endpoints lie strictly inside (0.2218355, 0.2218365), so the correctly
rounded six-decimal value is **0.221836**. This is a deterministic interval
certificate, not a statistical confidence interval. Seven decimal places
are not certified. The result concerns the baseline U4, not an optimized body.

## Checks performed locally

1. Standard-library audit (`src/audit_u4_return.py`): returned-file and source
   hashes; completed checkpoint and coverage-counter consistency; exact
   dyadic checkpoint endpoints; outward inclusion of the three imported
   intervals against the original exact U4-G piece records; rational
   ten-piece aggregation; the displayed enclosure and six-decimal rounding.
   Its output is `results/u4_s_return_audit.json`.
2. Completed-checkpoint replay with python-flint 0.9.0: the frozen driver
   recomputed the six 1D integrals and the final sum using a copy of the
   supplied completed I2_04 checkpoint. The mathematical fields match the
   returned certificate. Scope is recorded in
   `results/u4_s_checkpoint_replay.json`.

Neither local check repeats the full adaptive 2D integration. The returned
production log documents that full run. Hashes, counters and success flags
alone are not proofs of the integration kernel; the mathematical formulas,
frozen interval code and Arb arithmetic remain the trust boundary. Automated
checks do not enlarge the author's stated human verification scope.

## Reproduce

Run from `manuscript1/`:

```text
python reproduce.py quick
python -m pip install -r requirements-u4.txt
python reproduce.py u4-resume
```

The resume mode creates a new `runs/` directory and copies the supplied
checkpoint before running; original records are never overwritten.

To repeat the full adaptive production computation, use a new, empty directory
and omit `--resume`:

```text
python src/u4/u4_certify_6dp_adaptive.py --base-n 4096 --refine 4 --dps 50 --outdir runs/u4_s_fresh
```

The three older imported 2D pieces are still reused by that command.
Their full U4-G regeneration command is given in `CERTIFICATES.md`.
Run metadata will change file hashes; compare mathematical endpoints and
parameters, while retaining the new file's own provenance.
