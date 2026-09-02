# Trust boundaries

Where floating-point or otherwise non-exact computation enters each
pipeline, and why it does not affect the certified conclusion.

## Raised Meissner

```
floating-point geometric construction (Qhull, upstream, not in this repo)
              |
              | candidate combinatorics generation ONLY
              | (proposed tetrahedron list; not trusted as a convexity oracle)
              v
exact rational vertex data (parent .npz)
              |
              v
exact membership + exact reduced-subset identity checks
              |
              v
exact topology checks (unique facets, triangle incidence = 2,
facet-dual connectivity, vertex-link Euler characteristic = 2)
              |
              v
exact local convexity per edge star
(Rybnikov, "An Efficient Local Approach to Convexity Testing of
Piecewise-Linear Hypersurfaces," arXiv:cs/0703030, Lemma 10 cone-sign test;
every sign is an exact Python-integer 4x4 determinant sign)
              |
              v
Rybnikov Theorem 8 (bounded n=4 local-to-global PL convexity)
              |
              v
exact determinant volume sum + exact rational threshold comparison
              v
rigorous lower bound
```

Floating point appears only in: (a) the upstream Qhull run that proposes a
candidate triangulation (not part of this repository, not trusted), and (b)
`scipy.optimize.linprog`, used only to *find* a numerically plausible
origin-interior 5-point simplex — the resulting simplex is then certified
exactly with `fractions.Fraction` before being used, so a wrong numerical
search would cause the exact certification step to fail, not produce a
false-positive certificate.

## U4

```
analytic reduction (exact, by hand / symbolic — not in this repo)
        |
        v
finite exact domain decomposition (piecewise, sine-square endpoint maps)
        |
        v
python-flint / Arb rigorous ball arithmetic
(every sqrt/asin/acos/etc. call returns an interval guaranteed
to contain the true value; domain-clamping only ever narrows toward
the already-proved sub-interval, never widens it)
        |
        v
directed lower/upper endpoint summation (exact binary man_exp serialization
across checkpoints — no decimal round-trip loss)
        v
validated enclosure
```

No step converts an Arb interval to a Python `float` before the final
comparison; the comparison `total_upper < target.lower()` is itself an Arb
comparison.

## What "exact" and "validated" mean here, precisely

- **Exact**: computed with Python arbitrary-precision integers or
  `fractions.Fraction`; the result is exactly the value it claims to be, not
  an approximation with an error bound.
- **Validated**: computed with rigorous ball/interval arithmetic (Arb); the
  result is an interval *proved* to contain the true value, even though the
  true value itself is irrational and not exactly representable.

Neither is a floating-point point-estimate or a statistical/Monte-Carlo
result. See [`evidence_levels.md`](evidence_levels.md) for how this compares
to the historical/exploratory scripts that are deliberately excluded from
this repository.
