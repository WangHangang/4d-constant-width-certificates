# 4D Constant-Width Certificates

Source code and certificate metadata for computer-assisted results concerning
four-dimensional bodies of constant width, supporting the paper *"Search for
Small Four-Dimensional Bodies of Constant Width."*

## Main certified comparison

```
Vol_4(U4^(1)) <= 0.22183829946417434497071609358086433207758599642028
             <  0.22185
             <  Vol_4(K_RM)
```

- **Upper bound** on `Vol_4(U4^(1))`: validated Python/Arb ball-arithmetic
  enclosure (deterministic, not statistical). See [`u4/README.md`](u4/README.md).
- **Lower bound** on `Vol_4(K_RM)` (the raised Meissner body), via the exact
  rational threshold `4437/20000 = 0.22185`: exact-integer/rational
  certificate. See [`raised_meissner/README.md`](raised_meissner/README.md).

Neither bound is a floating-point approximation used as the final proof step;
see [`docs/trust_boundaries.md`](docs/trust_boundaries.md) for exactly where
floating-point computation enters each pipeline and why it does not affect
the certified conclusion.

## Large data files

The exact-rational certificate data (`.npz`, tens of MB) that the
Raised-Meissner verifier consumes are **not** stored in this git history.
They are archived immutably on Zenodo:
[10.5281/zenodo.22245760](https://doi.org/10.5281/zenodo.22245760) (concept
DOI, always resolves to the latest version); see
[`raised_meissner/data/README.md`](raised_meissner/data/README.md) for the
version-specific DOI and per-file SHA-256 hashes.

## Layout

| Path | Contents |
|---|---|
| [`raised_meissner/`](raised_meissner/) | Exact local-convexity / volume certificate for the raised Meissner body |
| [`u4/`](u4/) | Validated Arb enclosure for `Vol_4(U4^(1))` |
| [`raised_u3/`](raised_u3/) | Auxiliary certificate for the raised `U_3` (not part of the U4 vs. K_RM comparison) |
| [`docs/`](docs/) | Certificate map, trust-boundary notes, evidence-level taxonomy, data schema |

## Reproduction

Each subdirectory's `README.md` gives the exact command used to regenerate
its certificate from the corresponding `src/` script and (for Raised
Meissner) the Zenodo-archived input data.

## Status

This is a working reproducibility archive accompanying an in-progress paper.
Contents will be extended as further certificates are added; existing files
are not renamed or silently replaced — superseded files are removed rather
than kept alongside newer versions, to avoid ambiguous citations.
