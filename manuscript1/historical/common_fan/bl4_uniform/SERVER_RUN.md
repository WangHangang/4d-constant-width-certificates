# Server instructions: common rational inner-polytope proof

The intended machine is Andriy's i9 workstation (20 cores, 32 GB RAM).  Long
runs of one or two weeks are acceptable.  The goal is not to obtain many
correct decimals; it is to produce the easy-to-check strict separators

```text
Vol_4(U4) < 4437/20000 < Vol_4(K).
```

The large production calculation is intentionally performed on that machine,
not in the ChatGPT sandbox.

## 0. Environment and proof boundary

Python 3.10+ is sufficient for the independent verifier.  The AMO *generator*
uses NumPy/SciPy:

```bash
python3 -m pip install -r requirements-generator.txt
```

NumPy/SciPy are proposal-only.  Final verification is standard-library exact
arithmetic plus the local `bl4` package.

On Windows, the AMO generators automatically request `ES_SYSTEM_REQUIRED` so
the machine does not sleep during the run.  On Linux, use `nohup`/`tmux`.

## 1. Smoke tests before any long computation

The quickest full package check is

```bash
python3 preflight_handoff.py
```

It checks the frozen Raised-U3 hash, all exact fan/oracle tests, the packaged
certificates, and the shared-cache schema.  The individual commands are:

```bash
python3 tests/test_fan.py
python3 tests/test_adaptive_fan.py
python3 tests/test_oracles.py
python3 tests/test_raised_u3.py
python3 tests/test_amo_witness.py

python3 verify_certificate.py certificates/M4_k2.json
python3 verify_certificate.py certificates/M4_k3.json
python3 verify_certificate.py certificates/RaisedMeissnerFace_k2.json
python3 verify_certificate.py certificates/RaisedMeissnerFace_k3.json
python3 verify_certificate.py certificates/Body5_k2.json
python3 verify_certificate.py certificates/Body5_k3.json
python3 verify_certificate.py certificates/RaisedU3_k2.json
python3 verify_certificate.py certificates/RaisedU3_k3.json
python3 verify_certificate.py certificates/AMO_k2.json
python3 verify_certificate.py certificates/AMO_k3.json
python3 verify_certificate.py certificates/AMO_adaptive_smoke_b2_r24.json
```

The packaged AMO V1 level-3 certificate has lower bound
`0.198407195743731`; the packaged V2 smoke certificate has
`0.199047896824984`.  Both are below the target and are framework checks, not
final comparisons.

The warm production cache is `cache/AMO_shared_ray_cache_r24.json`
(`radius_den=2^24`).  If it ever needs to be rebuilt, run

```bash
python3 seed_amo_cache.py --radius-exp 24 \
  --cache cache/AMO_shared_ray_cache_r24.json \
  certificates/AMO_k2.json certificates/AMO_k3.json \
  certificates/AMO_adaptive_smoke_b2_r24.json
```

Every source certificate is independently verified before seeding, and each
endpoint is exactly reverified after any denominator rescaling.

## 1.25 Cheap common level-4 probe before a long run

Before committing to a multi-day resolution, obtain one third common-resolution
point for every comparison body:

```bash
python3 -u run_level4_probe.py --workers 16
```

This runs `k=4` for `M4`, `RaisedMeissnerFace`, `Body5`, `RaisedU3` and AMO,
and independently verifies every result.  It writes
`logs/level4_probe_summary.json`.  This is deliberately a scheduling probe:
an empirical fit from `k=2,3,4` can guide the next resolution, but no fit or
extrapolation is part of the proof.

The four predicate bodies use independent exact ray workers.  AMO uses the
checkpointed witness generator and the shared exact ray cache.

## 1.5 Preferred unattended AMO driver

For the i9 workstation the simplest production command is

```bash
python3 -u run_amo_server.py --workers 16
```

The driver first runs the full package preflight, requests a Windows anti-sleep
guard when applicable, streams each stage to a persistent log, preserves the
generators' checkpoint/resume state, independently verifies every completed
certificate, and stops as soon as the exact separator is crossed.  By default
it runs substantive V1 levels 4, 5 and 6 and then switches to V2 base level 5
with six 20% refinement rounds.  The level-6 stage was added because a coarse
pre-asymptotic extrapolation from the packaged AMO `k=2,3` certificates suggests
that level 5 may still miss the separator.  The extrapolation is only a resource
planning heuristic; the exact certificate decides the result.  The exact production command can be audited without
starting a computation via

```bash
python3 run_amo_server.py --dry-run --skip-preflight
```

The older PowerShell/Bash launchers remain as transparent alternatives.

## 2. V1 = uniform fixed fan

For the four predicate adapters use the common generator, for example

```bash
python3 generate_certificate.py --body M4 --level 3 --workers 16 \
  --out certificates/M4_k3.json
python3 generate_certificate.py --body RaisedU3 --level 5 --workers 16 \
  --out certificates/RaisedU3_k5.json
python3 verify_certificate.py certificates/RaisedU3_k5.json
```

For AMO use the witness generator.  It checkpoints automatically and can be
resumed by rerunning the same command:

```bash
python3 -u generate_amo_certificate.py --level 3 --radius-exp 24 --workers 16 \
  --shared-cache cache/AMO_shared_ray_cache_r24.json \
  --out certificates/AMO_k3.json \
  --checkpoint certificates/AMO_k3.checkpoint.json \
  2>&1 | tee logs/AMO_k3_generate.log
python3 verify_certificate.py certificates/AMO_k3.json \
  2>&1 | tee logs/AMO_k3_verify.log
python3 check_threshold.py certificates/AMO_k3.json
```

Level 3 is already packaged, so the first substantive server run is normally
level 4, followed by levels 5 and 6 only if necessary.  Pass the same `--shared-cache` to
each level; inherited rays are reverified and reused.  Level 4 has 11,008
rays / 65,536 cells; level 5 has 87,552 rays / 524,288 cells.  Do **not** jump
directly to the 33.5-million-cell global level-7 fan.

The AMO checkpoint loader asserts its schema, 0-based ray indexing, fan hash,
direction scale, radius denominator and ray-index range before reusing any
saved witness.

## 3. V2 = adaptive refinement

V2 is the preferred fallback when V1 is close to the threshold.  It has the
same proof kernel; only the choice of cells to subdivide is heuristic.

Predicate adapters:

```bash
python3 generate_adaptive_certificate.py --body RaisedU3 \
  --base-level 5 --fraction 0.15 --max-rounds 10 --workers 16 \
  --out certificates/RaisedU3_adaptive.json
python3 verify_certificate.py certificates/RaisedU3_adaptive.json
```

AMO:

```bash
python3 -u generate_amo_adaptive_certificate.py \
  --base-level 5 --fraction 0.20 --max-rounds 6 --radius-exp 24 --workers 16 \
  --shared-cache cache/AMO_shared_ray_cache_r24.json \
  --out certificates/AMO_adaptive.json \
  --checkpoint certificates/AMO_adaptive.checkpoint.json \
  2>&1 | tee logs/AMO_adaptive_generate.log
python3 verify_certificate.py certificates/AMO_adaptive.json \
  2>&1 | tee logs/AMO_adaptive_verify.log
python3 check_threshold.py certificates/AMO_adaptive.json
```

The verifier rebuilds the adaptive fan and checks the exact coverage invariant
before it sums determinants.  Therefore a bug in the heuristic score cannot
create a proof.

## 4. Suggested unattended order

The cheap/already-ready comparisons should not block the AMO run. A practical
order is:

1. verify the packaged smoke certificates;
2. run `python3 -u run_level4_probe.py --workers 16` once for all five bodies;
3. inspect the verified `k=4` lower bounds and use them only to choose the next
   computational resolution;
4. `M4` needs no further work because packaged `k=3` already crosses;
5. run `RaisedMeissnerFace`, `Body5`, and `RaisedU3` with V1/V2 as needed;
6. run AMO through V1 levels 4, 5, 6 with early stopping, then V2 base level 5
   if the fixed fan still misses.

The coarse `k=2 -> k=3` convergence exponents are pre-asymptotic. They may be
conservative and are not theorem statements. The level-4 probe is specifically
intended to prevent an unnecessary global level-7 allocation.

The existing manuscript interval-domain proof already establishes
`4437/20000 < Vol_4(AMO)`.  The server AMO run is therefore a *unification and
exposition* task: it replaces a body-specific integral certificate by the same
rational-inscribed-polytope architecture used for the other bodies.

## 5. Resource cautions

The predicate generators and the AMO generators support `--workers N`.
For predicate bodies, each worker proposes and exact-checks independent rational
ray endpoints; ordered collection in the parent means worker scheduling cannot
change the certificate.  For AMO, each worker constructs an independent
boundary witness which is then rechecked by the verifier.

AMO V1 and V2 should use the same `--shared-cache` file, so a V2 fallback
reuses all exact endpoints already found by V1. Each cache hit is rechecked by
the exact endpoint verifier.

Each worker owns a NumPy/SciPy
proposal library and certifies independent rays; the process pool keeps only a
bounded number of tasks in flight.  Worker processes also request one
BLAS/OpenMP thread to reduce oversubscription.  On a 20-core machine,
`--workers 16` is a conservative first production setting.  It is still not
correct to infer a 16x or 20x speedup: patch-seam rays have variable solve
times, process startup has a cost, and exact certificate verification remains
serial.

For a first timing sample, run level 3 with 8 workers and compare its log with
16 workers before committing to a week-long level 4/5/6 run.  Changing
`--workers` does not invalidate or change the checkpoint schema; the saved
mathematical data are keyed by ray, not by worker.

AMO witness certificates are much larger than the predicate certificates.  In
the current JSON format the smoke files use about 464 bytes per certified ray.
As a rough disk budget, uniform level 6 (about 699,000 rays) is about 325 MB,
and uniform level 7 (about 5.6 million rays) is about 2.6 GB, before
checkpoints and logs.  These figures are empirical storage estimates only.
Reserve several additional GB of free disk space.

A global level 7 has 33,554,432 cells and can put substantial pressure on a
32-GB machine. Adaptive refinement is the default response to a near miss.
Always keep the `.checkpoint.json`, final `.json`, generator log and verifier
log together.

For native Windows PowerShell, `run_amo_progressive.ps1` is the prepared AMO
V1-to-V2 launcher.  `run_amo_progressive.sh` is the corresponding Bash/WSL
launcher.
