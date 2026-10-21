# Dimension-four B–L volume comparison: code handoff

This package separates two things that must not be confused.

1. `bl4/` is a common rational inner-polytope framework. It fixes a
   midpoint-refined triangulation of the boundary of the 16-cell, certifies
   one point on each rational ray, and sums exact 4-simplex determinants.
   `verify_certificate.py` rebuilds the fan, rechecks every point and repeats
   the determinant and threshold comparisons using the Python standard
   library only.
2. `reference_methods/` contains the existing body-specific certificate
   programs from the manuscript. They are included as reference and
   cross-check implementations; they do not constitute one common numerical
   method.

The comparison threshold is

```text
T = 4437/20000 = 0.22185.
```

The fixed-fan proof uses only two geometric facts. A simplex whose vertices
belong to a convex body belongs to that body. The fan simplices lie in
pairwise disjoint cones and cover all directions. `audit_fan` checks the
integer invariant

```text
sum(abs(det(direction columns))) = 16 * direction_scale**4,
```

which certifies the second fact independently of the body.

## Proof-ready common adapters

- `M4`: exact Cartesian integer predicate.
- `RaisedMeissnerFace`: exact positive preserved-arc predicate plus a
  conservative rigorous negative-half witness family.  **All** preserved arcs
  are checked.  The separate exact equal-volume theorem transfers the result
  to the other noncongruent raised-Meissner variant.
- `Body5`: exact implementation of the spherical-patch support theorem in the
  regular-simplex coefficient chart.
- `RaisedU3`: every positive ellipse constraint is cleared to an integer
  quartic on `q in [0,1]` and proved nonnegative by exact Bernstein
  coefficients plus dyadic de Casteljau subdivision.  The negative half uses
  a rigorous finite subfamily of the frozen `L_-` inner model.
- `AMO`: there is deliberately **no fake global membership predicate**.
  Instead each nonzero endpoint stores a source-boundary witness.  The
  standard-library verifier encloses that algebraic boundary point with exact
  rational intervals and applies the inward-shrink lemma from the certified
  rational inball `B(G,1/3)`.  The regular-4-simplex coefficient chart has
  metric

  ```text
  |a|^2 = (5*sum(a_i^2) - (sum a_i)^2)/10
  ```

  and Jacobian `sqrt(5)/20`, which is included in the exact final volume
  comparison.

The AMO proposal layer (`bl4/amo_generate.py`) uses NumPy/SciPy only to locate
a convenient boundary chart and rational parameter point. Those floating
steps are not proof data. `bl4/amo_parallel.py` can distribute independent
AMO ray searches over multiple Python processes.  The predicate generators
also accept `--workers N` through `bl4/radial_parallel.py`; every worker still
returns only a radius that has passed the exact body predicate, and ordered
collection in the parent makes worker completion order irrelevant to the
certificate. `verify_certificate.py` does not import NumPy or SciPy.

## V1 and V2 execution variants

The mathematics is the same in both variants.

- **V1 uniform:** `generate_certificate.py` for predicate adapters and
  `generate_amo_certificate.py` for AMO.
- **V2 adaptive:** `generate_adaptive_certificate.py` for predicate adapters
  and `generate_amo_adaptive_certificate.py` for AMO.

Both store the exact fan metadata, one certified rational endpoint per final
ray and an exact determinant sum.  V2 additionally stores every selected cell
index.  The independent verifier reconstructs the nonuniform fan and checks
`sum(abs(det)) = 16*scale**4`.  Thus adaptive selection affects efficiency,
not the proof.

The AMO generators include checkpoint/resume, progress/ETA reporting,
optional `--workers N` multiprocessing, and a Windows
`SetThreadExecutionState` guard to prevent automatic sleep during long runs.
The process pool keeps only a bounded number of ray tasks in flight and limits
BLAS/OpenMP thread counts in workers to reduce oversubscription. Before a
checkpoint is reused, its schema, indexing convention, fan hash, scale,
radius denominator and index ranges are asserted explicitly.

V1 and V2 may also share `cache/AMO_shared_ray_cache_r24.json`.  Its keys are
primitive exact rational directions, so an inherited ray has the same key
after refinement.  A cached endpoint is **never trusted merely because it is
in the cache**: before reuse, the current generator re-runs the exact AMO
endpoint verifier for the current direction/denominator.  The cache is only a
proposal-time optimization and is absent from the proof kernel.  The included
cache uses `radius_den = 2^24` and can be rebuilt from verified certificates
with `seed_amo_cache.py`.

Local architecture checks now include the following independent round trips:

```text
AMO V1 level 2: 192 rays, 1024 cells,  L = 0.161611471306101
AMO V1 level 3: 1408 rays, 8192 cells, L = 0.198407195743731
AMO V2 base 2 + two 20% rounds:
    1235 rays, 5903 cells,             L = 0.199047896824984
```

All are below the separator and are intentionally only framework validation.

Packaged predicate smoke certificates now include:

```text
M4:                 k2 = 0.198487903237575 < T
                    k3 = 0.233244525156270 > T
RaisedMeissnerFace: k2,k3 < T
Body5:              k2 = 0.176426523887951 < T
                    k3 = 0.208292228646337 < T
RaisedU3:            k2,k3 < T
```

Only `M4_k3.json` presently crosses the separator in the unified common-fan
files.  The other proof-side adapters are verified, but their production
common-fan refinements are still pending; the manuscript's existing
body-specific validated certificates remain load-bearing for those strict
comparisons until a common-fan file crosses `T`.
The V2 smoke run gives a slightly larger certified lower bound than V1 level 3
with fewer cells, which is useful engineering evidence for the adaptive
fallback but is not itself a theorem about convergence.  The production runs
are assigned to Andriy's i9 / 20-core / 32-GB machine.

Raised Robert remains experimental and is not part of the present unified
comparison target.

Run `python3 preflight_handoff.py` after unpacking for a one-command audit of
the manifest, frozen dependency, fan/oracle tests, packaged certificates and
shared-cache schema.  Before a long run, `python3 -u run_level4_probe.py --workers 16` obtains and
independently verifies one `k=4` common-resolution point for all five comparison
bodies.  It is a scheduling diagnostic only: any convergence fit from `k=2,3,4`
is heuristic and never part of the proof.

For the unattended AMO production computation,
`python3 -u run_amo_server.py --workers 16` is the preferred cross-platform
driver.  It runs V1 levels 4, 5 and 6, independently verifies each completed
certificate, stops immediately after a strict separator, and otherwise switches
to V2 from base level 5 while reusing the same exact ray cache and checkpoints.
The additional level-6 stage is conservative because the coarse `k=2,3` AMO
trend suggests level 5 may still miss the line; that trend is pre-asymptotic
and is used only for resource planning.

AMO JSON witnesses are relatively large: the current smoke format is about
464 bytes per ray, corresponding roughly to 325 MB at uniform level 6 and
2.6 GB at uniform level 7, before checkpoints/logs.  A global level 7 also has
33,554,432 cells, so V2 adaptive refinement is preferred after a near miss on
a 32-GB machine.

See `POINT_SELECTION.md` for the direction and adaptive-selection rules,
`AMO_WITNESS.md` for the short proof contract, `SERVER_RUN.md` for commands,
`STATUS.md` for the exact claim boundary, and `reference_methods/README.md`
for the older certificate routes.
