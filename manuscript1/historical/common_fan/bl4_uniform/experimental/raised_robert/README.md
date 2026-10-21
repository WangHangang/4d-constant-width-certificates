# Raised Robert: experimental status

This directory is outside every theorem claim.

The existing Stage-2 outer covers have spatial resolution of order `0.3` and
produce a much smaller inner surrogate before any fan aggregation. A floating
diagnostic estimates that surrogate near `0.076`; this number is diagnostic,
not a rigorous upper bound. The older exact box certificate gives only the
partial lower bound near `0.0533`.

Possible continuations:

1. Enumerate exact vertices of the Stage-2 rational H-polytopes in dimension
   three, then maximize each squared-distance constraint over those vertices.
2. Replace the negative half by explicit rational witness balls, as in the
   raised-Meissner adapter.
3. Deeper axis-aligned box covers are not recommended: reducing the geometric
   scale from about `0.3` to about `1.2e-4` would require roughly eleven more
   bisections per axis and an infeasible leaf count.

