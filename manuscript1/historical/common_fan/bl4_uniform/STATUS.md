# Status on 2026-09-07

Common separator:

```text
T = 4437/20000 = 0.22185.
```

The proof target is `Vol_4(U4) < T < Vol_4(K)` for each named comparison
body. The common framework has two execution variants with the same proof
kernel:

- **V1 (uniform):** fixed midpoint-refined rational 16-cell fan;
- **V2 (adaptive):** exact local refinements of the same fan, with the final
  coverage/nonoverlap invariant independently rechecked.

The phrase **adapter verified** below means that the body-specific endpoint
membership proof is implemented and independently testable. It does **not** mean
that the common-fan numerical lower bound has already crossed `T`.

| body | proof-side adapter | packaged common-fan status |
|---|---|---|
| `M4` | verified exact predicate | `k2=0.198487903237575<T`; `k3=0.233244525156270>T` |
| `RaisedMeissnerFace` | verified exact positive preserved-arc test + rigorous negative witness balls | `k2,k3<T`; production refinement pending; equal-volume theorem transfers to the noncongruent companion |
| `Body5` | verified exact quadratic-field predicate | `k2=0.176426523887951<T`; `k3=0.208292228646337<T`; production refinement pending |
| `RaisedU3` | verified integer quartics + Bernstein/de Casteljau; rigorous negative inner model | `k2,k3<T`; production refinement pending |
| `AMO` | verified boundary-parameter witness + rational interval enclosure + inward shrink | V1 `k2,k3<T`; V2 smoke `<T`; production refinement pending |
| `RaisedRobert` | experimental only | deferred; current exact lower bound does not settle the comparison |

Thus, **among the unified rational-fan certificates currently packaged, only
`M4_k3.json` already proves the strict separator**. For the other bodies, the
older body-specific validated certificates in the manuscript remain
load-bearing until the common-fan production runs cross `T`.

## AMO evidence boundary

The AMO generator uses NumPy/SciPy only to propose a boundary patch and
parameters. The saved proof data contain a rational parameter point, a simplex
permutation and an inward factor. The independent verifier uses only the
Python standard library and checks:

1. the source boundary chart and parameter domain;
2. an exact rational interval enclosure of the corresponding algebraic
   boundary point;
3. the rational inball radius `r0 = 1/3` and the inward-shrink inequality;
4. the exact rational fan endpoint;
5. the full fan determinant coverage invariant and exact volume sum.

No floating Qhull combinatorics and no asserted global AMO membership oracle
are used in the proof.

The precise shrink statement is the following. If `B(G,r0) subset K`,
`x in K`, `|x-c| <= rho`, and `0 <= lambda <= 1`, then

```text
B(G + lambda*(c-G), (1-lambda)*r0 - lambda*rho) subset K
```

whenever the displayed radius is nonnegative. In the centered AMO chart
`G=0`, `r0=1/3`, the endpoint verifier checks the equivalent condition

```text
|p - lambda*c| + lambda*rho <= (1-lambda)/3.
```

This corrects an earlier handoff-summary typo that omitted the `-lambda*rho`
loss in the ball radius; the manuscript lemma and verifier code were already
correct.

Local certified checks produced in this handoff:

```text
AMO V1 level 2: 192 rays / 1024 cells
    0.161611471306101   PASS independent verifier
AMO V1 level 3: 1408 rays / 8192 cells
    0.198407195743731   PASS independent verifier
AMO V2 base level 2, two 20% refinement rounds:
    1235 rays / 5903 cells
    0.199047896824984   PASS independent verifier
```

These remain below `T`; they validate V1, V2, checkpoint/resume, exact adaptive
fan reconstruction and AMO witness serialization, not the final comparison.
A shared exact ray cache with `radius_den=2^24` contains 1458 unique verified
AMO directions. It was assembled from independently verified V1/V2
certificates and every rescaled endpoint was reverified exactly. V1 and V2
reverify a cache hit again before reuse, so the cache can only save generator
time; it cannot create a proof.

## Supervisor production order

The heavy calculation is intentionally left to Andriy's i9 / 20-core / 32-GB
machine. Before committing to a multi-day level, the cheapest useful diagnostic
is one **common level-4 probe** for all five comparison bodies:

```bash
python3 -u run_level4_probe.py --workers 16
```

This generates and independently verifies `k4` for the four predicate adapters
and AMO. The resulting third common-resolution point can be used to reassess
empirical convergence before allocating level 6/7 resources. Such a fit is
only a scheduling heuristic; it is not part of the proof.

For AMO, the unattended production driver now defaults to

```text
V1 level 4 -> V1 level 5 -> V1 level 6 -> V2 base level 5
```

with exact verification and early stopping after every completed stage:

```bash
python3 -u run_amo_server.py --workers 16
```

The extension to level 6 is deliberate: extrapolation from only the coarse
`k2,k3` AMO certificates suggested that level 5 may still miss `T`. That
extrapolation is pre-asymptotic and must not be treated as a theorem. If level
4 already shows substantially faster convergence, the schedule can be shortened
before the larger run.

For the predicate adapters, `generate_certificate.py` and
`generate_adaptive_certificate.py` now accept `--workers N`. Parallelism only
changes wall-clock time: each returned endpoint is exact-checked by the body
oracle, and output order is deterministic. `run_server_queue.sh` passes 16
workers by default and gives `RaisedU3` more adaptive rounds instead of jumping
directly to a memory-heavy uniform level 7.

## Resource budget

AMO witnesses are much larger than predicate certificates. The current smoke
files are about 464 bytes per certified ray, so a rough storage estimate is:

```text
uniform level 6: about 699,000 rays -> about 325 MB certificate
uniform level 7: about 5.6 million rays -> about 2.6 GB certificate
```

These are empirical file-size estimates, not mathematical quantities. Keep the
checkpoint, final certificate, generator log and verifier log together and
reserve several additional GB of free disk space. A global level 7 also has
33,554,432 cells and is not the default on a 32-GB machine; V2 adaptive
refinement is preferred after a near miss.

## Frozen Raised-U3 dependency

`reference_methods/raised_u3/raised_u3_negative_inner_validated_lower.py`
has SHA-256

```text
e8d7b5b8e9a9feb60b6b6ac65c04f0694a768bd50d0d34736cd0e29c01b495ef
```

Do not modify this file: the manuscript's frozen legacy certificate binds to
that byte hash.
