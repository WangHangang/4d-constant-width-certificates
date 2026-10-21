# How the rational points are selected

The certificate uses a deterministic rational direction fan and a body-specific
*proof* of one endpoint on every ray. Random or quasi-Monte Carlo points are
not used in the proof.

There are two endpoint mechanisms:

1. **predicate adapters** (`M4`, `RaisedMeissnerFace`, `Body5`, `RaisedU3`);
2. the **AMO witness adapter**, which deliberately does *not* claim a global
   membership oracle.

Both feed the same exact determinant-volume verifier.

## Fixed direction fan

Start from the boundary of the four-dimensional cross-polytope (the 16-cell).
Its vertices are `+/- e_i`, and its 16 tetrahedral facets correspond to the 16
sign choices. One midpoint refinement replaces every tetrahedron by eight
tetrahedra. After level `k`, every direction vertex is an integer vector with
common scale `2**k`.

The exact audit

```text
sum(abs(det(direction columns))) = 16 * direction_scale**4
```

is checked by the verifier. Together with the explicit recursive midpoint
subdivision, this certifies that the final leaf cones cover all directions and
have disjoint interiors.

## Predicate adapters: one certified point on each ray

For a direction vector `u`, the generic generator may propose a radius using
floating arithmetic. It rounds the proposal down to the rational grid and
calls the body's exact containment predicate. If necessary, it backs off or
bisects until the endpoint

```text
p = radius_num * u / (radius_den * direction_scale)
```

is accepted. The stored endpoint is checked once more by the generator and
again by the independent verifier.

For `RaisedU3`, the positive-half predicate is the exact integer
quartic/Bernstein-de Casteljau test, while the negative half uses a finite
rational subfamily of the frozen rigorous `L_-` inner model. Floating
arithmetic chooses where to look; it never establishes membership.

## AMO: boundary witness plus inward shrink

The direct AMO four-dimensional peabody is different. The source paper proves
a complete boundary assembled from pair-wedge, triple-wedge and spherical-cap
patches, but this boundary description is not silently promoted to a global
membership predicate.

For every rational fan ray the AMO generator therefore performs two separate
steps.

### Proposal step — not part of the proof

NumPy/SciPy nearest-neighbour and least-squares searches identify a source
boundary chart and a parameter point whose boundary direction is close to the
target fan ray. The chosen parameters are rationalized to a dyadic rational
point.

This step may fail, be changed, or be parallelized without changing the proof
statement.

### Witness step — proof-producing

The certificate stores:

- patch type: `pair`, `triple`, or `cap`;
- one permutation of the five regular-simplex vertices;
- one rational parameter point (serialized as a degenerate rational interval
  box);
- one rational inward factor `lambda`.

The independent standard-library verifier then:

1. reevaluates the exact algebraic AMO boundary formula with rational interval
   arithmetic;
2. obtains a rational midpoint `c` and a rigorous rational enclosure radius
   `rho`, so that a source-proved boundary point `x` satisfies
   `|x-c| <= rho`;
3. uses the exact inball
   `B(G,1/3) subset AMO`;
4. checks
   ```text
   |p - lambda*c| + lambda*rho <= (1-lambda)/3;
   ```
5. concludes `p in AMO` by convexity.

The cap path additionally runs the exact first-component audit that excludes
the remote positive branch of the cone quadratic. The current audit closes in
55 rational boxes.

Thus AMO still produces an ordinary rational endpoint on the prescribed fan
ray; only the membership witness is body-specific.

## Exact simplex volume

For each leaf cone with four rays, the origin and the four certified endpoints
form a 4-simplex. Convexity puts that simplex inside the body. The cones have
disjoint interiors, so the exact lower volume is the sum of the integer
determinants divided by the common fourth-power denominator and by `4!`, with
the body-specific chart Jacobian applied at the end.

For AMO the regular-simplex coefficient chart has

```text
|a|^2 = (5*sum(a_i^2) - (sum a_i)^2)/10,
Jacobian = sqrt(5)/20.
```

The final threshold comparison against `4437/20000` is exact algebraic
arithmetic.

## V1: uniform refinement

V1 refines every fan cell. It is the simplest execution path and should be
tried first. AMO uses `generate_amo_certificate.py`; predicate bodies use
`generate_certificate.py`.

The local V1 checks are:

```text
level 2: rays=192,  cells=1024, L_AMO=0.161611471306101...
level 3: rays=1408, cells=8192, L_AMO=0.198407195743731...
```

Both are independently verified and below `4437/20000`; no AMO comparison
conclusion is inferred from them.  Level 3 is packaged mainly to validate the
full V1 path and to warm the shared production cache before the supervisor run.

## V2: adaptive refinement

If the exact lower bound has not crossed `4437/20000`, each leaf cone receives
a heuristic score using:

- its determinant weight;
- angular spread of its four direction vectors; and
- spread of the four certified radii.

Only the top requested fraction is midpoint-refined. The certificate records
the selected cell indices. The verifier reconstructs the resulting nonuniform
fan, checks its exact coverage invariant, rechecks every endpoint, and
recomputes the exact volume.

The score affects efficiency only. It is not part of the proof.  A local
base-level-2 run with two 20% rounds gives 1235 rays / 5903 cells and the
independently verified lower bound `0.199047896824984`.  This is slightly above
the V1 level-3 bound despite using fewer cells; it is engineering evidence for
V2, not a convergence theorem.

## Checkpoint and multicore execution

The AMO V1 and V2 generators checkpoint exact witnesses. Before reading saved
ray data they assert the checkpoint schema, indexing convention, fan metadata
and index ranges. A corrupted or incompatible checkpoint is rejected rather
than silently reused.

For predicate adapters, `generate_certificate.py` and
`generate_adaptive_certificate.py` also accept `--workers N`.  Each worker uses
the same body oracle and returns only an exactly accepted rational radius;
ordered collection in the parent process preserves deterministic ray order.
For AMO, `--workers N` parallelizes only the proposal/witness construction over
independent rays. The process pool is bounded and numeric-library thread counts
are limited to avoid oversubscription. Completion order has no mathematical
meaning: the final certificate is serialized by ray index/cache key and then
verified from scratch.

The intended large production run is on Andriy's 20-core, 32-GB workstation;
`--workers 16` is a conservative starting point. The proof verifier itself is
kept simple and serial.

The optional shared ray cache is keyed by the primitive rational direction
`u/scale`.  Therefore inherited V1 rays and V2 rays can reuse the same
generator result.  Cache reuse is guarded twice: strict schema/radius-denominator
preflight, followed by a fresh exact endpoint-witness verification in the
current fan.  The cache is thus outside the logical proof boundary.


## Production-resolution diagnostic

Before a long production run, the package provides

```text
python -u run_level4_probe.py --workers 16
```

which generates and independently verifies a common V1 level-4 certificate
for all five comparison bodies.  The purpose is to obtain a third coarse
resolution point before deciding whether level 6, adaptive refinement, or a
larger fixed fan is computationally justified.  Any fitted convergence exponent
is a heuristic scheduling diagnostic only.  It is not a bound and is not used
by the verifier.

For AMO the default unattended escalation is now

```text
V1 k=4 -> V1 k=5 -> V1 k=6 -> V2 base k=5,
```

with independent verification and early stopping after every stage.  The
current AMO JSON witness format is roughly 464 bytes per ray, so uniform k=6
is about 325 MB and k=7 about 2.6 GB before checkpoints/logs.  This is why a
near miss should normally go to V2 rather than directly to global k=7.

## Stopping rule and trust boundary

Stop only after the independent verifier proves the exact strict inequality

```text
lower_bound > 4437/20000.
```

Failure to cross after substantial refinement should trigger an
oracle/witness-tightness review; more rays cannot repair a systematically
conservative inner model.

Exact determinant arithmetic downstream does not repair an uncertified
endpoint upstream. The proof boundary is therefore:

```text
heuristic search
    -> serialized rational endpoint witness
    -> exact endpoint verification
    -> exact fan audit
    -> exact determinant sum
    -> exact threshold comparison.
```
