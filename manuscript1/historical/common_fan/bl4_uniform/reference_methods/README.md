# Existing body-specific certificate programs

These files are copied from the manuscript project for code review,
reproducibility checks, and comparison with the common rational-fan method.
They use different proof methods and third-party dependencies. No program in
this directory is called by the independent common fixed-fan verifier.

## Contents

- `u4/`: Arb-based upper-bound scripts (`python-flint` / Arb required). The
  full production run was historically long.
- `raised_meissner/`: legacy rational-point membership and hull-audit programs.
- `raised_u3/`: the earlier certified simplicial/cubature implementation and
  its auxiliary files. NumPy, SciPy, Numba and `mpmath` are used by different
  legacy scripts.
- `body5/`: deterministic directed-rounding certificate and stratified
  boundary evaluator.
- `amo/`: direct four-dimensional peabody boundary-orbit interval certificate,
  which is the already-completed body-specific proof of
  `4437/20000 < Vol_4(AMO)`.

## Frozen Raised-U3 dependency

The file

```text
raised_u3/raised_u3_negative_inner_validated_lower.py
```

is included in this handoff and must remain byte-for-byte unchanged. Its
SHA-256 is

```text
e8d7b5b8e9a9feb60b6b6ac65c04f0694a768bd50d0d34736cd0e29c01b495ef
```

The manuscript's earlier strict Raised-U3 certificate is explicitly bound to
that digest. The new common adapter uses the same established negative-half
`L_-` mathematics; modifying the frozen file would break the provenance of
the older certificate.

## Relation to the new common proof

The new proof does **not** certify a floating Qhull or retrofit exact
determinants onto an uncertified point cloud. It independently constructs
rational ray endpoints and verifies membership/witnesses before summing exact
simplex determinants.

Large legacy NPZ/Qhull data are not proof inputs to the common verifier.
Where legacy data files are present in this directory, they are reference
artifacts only and their schemas must be checked before any future program
reads them.

For AMO, the legacy interval-domain volume certificate remains a separate
load-bearing proof until Andriy's production run returns a common-fan
certificate above the separator. The common AMO adapter instead uses
source-proved boundary patches plus an exact inward-shrink witness, avoiding
any claim of a global AMO membership oracle.
