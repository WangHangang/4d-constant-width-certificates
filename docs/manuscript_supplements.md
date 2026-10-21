# Companion manuscripts and verification scope

## Manuscript 1

Hangang Wang. *Volume Computation and Comparison for Four-Dimensional Bodies
of Constant Width*. Revised discussion draft, 21 October 2026.

The compact raised-body and Body5 production replays passed on 16 October
2026. Their recorded logs and exact inputs are retained under `manuscript1/`.
The 21 October packaging check verifies file identities and stored endpoint
arithmetic; it is not a new production replay. The five-entry volume ordering
also has an AMO comparison, whose precise evidence level is recorded in
`manuscript1/CERTIFICATES.md`. A new full U4-S production run returned on
21 October supplies the six-decimal certificate and completed checkpoint.
The local audit checks their identities, endpoint provenance and rounding;
the local checkpoint replay recomputes six 1D integrals and aggregation.
Neither local check repeats the adaptive 2D integration. Details and commands
are in `manuscript1/docs/U4_S_RETURN_AUDIT.md`.

```text
python manuscript1/reproduce.py quick
python manuscript1/reproduce.py body5
python -m pip install -r manuscript1/requirements-raised.txt
python manuscript1/reproduce.py raised
```

The large original raised-Meissner datasets remain on the existing Zenodo
record linked by `raised_meissner/data/README.md`. The new compact verifier
uses different, included inputs; do not interchange these datasets.

## Manuscript 2

Hangang Wang. *Symmetric Volume Descent from U4 and a Four-Dimensional Lower
Bound*. Revised discussion draft, 21 October 2026.

This paper contains an analytic symmetric descent. Its small scalar supplement
checks rational constants and interval enclosures for the lower bounds only:

```text
python -m pip install -r manuscript2/lower_bound_checks/requirements.txt
python manuscript2/lower_bound_checks/certify_lower_gap.py
```

The independent spectral constant is 0.1781840422363513...; the projection
constant is 0.1787220399103946.... The latter additionally depends on Jiang's
Theorem 1.1, arXiv:2610.13964v1. Neither script proves the analytic theorems,
certifies an optimized candidate, or identifies the unknown minimum volume.

## Attribution and human checking

Most work was completed during the author's Mitacs internship, under Andriy
Prymak's supervision. GPT-5.6 Sol, GPT-6 Sol and Astra assisted selected steps,
checks and coding; Claude gave limited editorial help. The author has checked
the main methods, but not every proof detail or program. Neither the release
nor the automated checks assert independent peer review or supervisor approval.

Original source and data notices are retained. See the licensing notice within
each added supplement; the archived collection is not being relicensed as a
uniformly authored body of software. No private correspondence or manuscript
PDFs are included in this update.
