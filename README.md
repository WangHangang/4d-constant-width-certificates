# 4D Constant-Width Certificates

Source code, data and certificate records for two manuscripts by Hangang Wang:

1. *Volume Computation and Comparison for Four-Dimensional Bodies of Constant Width*.
2. *Symmetric Volume Descent from U4 and a Four-Dimensional Lower Bound*.

Most of the work was completed during a Mitacs internship. The author thanks
Mitacs for support and Andriy Prymak for supervision and guidance in research
and learning methods. Both manuscripts remain drafts for mathematical review.

## October 2026 manuscript supplements

- [`manuscript1/`](manuscript1/): the five-volume-class comparison and AMO
  comparison, frozen verifier sources, compact raised-body NPZ inputs,
  historical evidence, and the explicit theorem-to-file index.
- [`manuscript2/lower_bound_checks/`](manuscript2/lower_bound_checks/): the
  portable scalar checks for the independent spectral bound and the
  literature-dependent projection bound. This does not check the analytic
  descent proof or certify an optimized numerical candidate.
- [`docs/manuscript_supplements.md`](docs/manuscript_supplements.md): scope,
  reference to each paper, commands, and current limitations.

Quick commands from the repository root:

```text
python manuscript1/reproduce.py quick
python -m pip install -r manuscript2/lower_bound_checks/requirements.txt
python manuscript2/lower_bound_checks/certify_lower_gap.py
```

The first command verifies frozen-file identities and stored arithmetic;
it does not repeat geometric certification. The final U4 six-decimal terminal
certificate is still unavailable. The existing wider U4 interval suffices for
the main strict comparisons. See the supplement's `CERTIFICATES.md`.

The original directories below are retained to preserve their existing links.

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

The two large exact-rational `.npz` files consumed by the original
`raised_meissner/` verifier are **not** stored in this git history. The
separate compact inputs for `manuscript1/` are included with that supplement.
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
