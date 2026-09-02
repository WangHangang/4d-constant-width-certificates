# Evidence levels

A shared vocabulary used throughout this repository's READMEs and
certificate maps, from weakest to strongest:

1. **Heuristic / floating-point point estimate.** A single double-precision
   (or similar) numerical value with no rigorous error bound. Useful for
   exploration; never used as a certified conclusion in this repository.

2. **Statistical / QMC (quasi-Monte Carlo).** A numerical estimate with a
   confidence interval or discrepancy bound, not a proof. Where such
   computations exist in the source project they are explicitly labeled
   as diagnostics, not as the theorem endpoint, and are not included here.

3. **Validated interval computation.** Computed with rigorous ball/interval
   arithmetic (e.g., Arb via python-flint), where every returned interval is
   *proved* to contain the true value. This is a genuine proof of an
   enclosure, not an estimate — see `u4/`.

4. **Exact arithmetic.** Computed with arbitrary-precision integers or
   exact rationals; the result is exactly the claimed value, not an
   approximation. See `raised_meissner/`.

5. **Fully rigorous computer-assisted proof.** A combination of (3) and (4)
   (or either alone) where every step in the chain from input data to final
   inequality has been independently re-verified by the code in this
   repository — not merely produced once and trusted.

## Important caveat

Exact arithmetic downstream does not retrospectively certify upstream
floating-point combinatorics unless the latter are independently verified.
Concretely: the fact that `raised_meissner`'s determinant sums and
local-convexity checks are exact integers does not, by itself, certify that
Qhull's upstream triangulation was combinatorially correct — that is why
the verifier independently recomputes topology (facet uniqueness, triangle
incidence, dual connectivity, vertex-link Euler characteristic) rather than
trusting Qhull's output. See [`trust_boundaries.md`](trust_boundaries.md).
