# Constant-width volume comparison

Computational supplement to Volume Computation and Comparison for
Four-Dimensional Bodies of Constant Width
by Hangang Wang, supervised by Andriy Prymak. This is a computational supplement to a discussion draft.

The paper compares the original five volume classes and additionally compares
U4 with the Arelio--Montejano--Oliveros peabody (AMO).
See CERTIFICATES.md for the exact link between each claim and its inputs.

Fresh complete Body5 and compact raised-body replays passed on 16 October
2026. Exact raised-body fractions and Body5 component intervals match their
historical records. Their full logs are included under results/.

## Start here

The quick check needs only the Python standard library (locally tested with Python 3.12):

    python reproduce.py quick

This checks file identities, stored exact arithmetic, and endpoint separation.
It does not rerun the geometric predicates or integrations.

A complete fresh Body5 replay needs only the standard library:

    python reproduce.py body5

A complete fresh compact raised-body replay:

    python -m pip install -r requirements-raised.txt
    python reproduce.py raised

The raised checker rechecks both bounds for each raised body. It takes U4 and
Body5 endpoint bounds as external inputs; Body5 has its separate replay above.
Each run writes to a new runs/ directory, so old success records are not reused.

## U4 and the common fan

The uniform-grid U4 source and its original exact endpoint records are included.
The full grid run is expensive (the archived run took about 14 hours).
The sharper six-decimal production certificate and completed checkpoint
returned on 21 October 2026 are included under `certificates/u4_s_20261021/`.
The production log records a full run of about 83 minutes. The current `quick`
command audits exact aggregation, imported endpoints, completion counters,
file identities and six-decimal rounding to 0.221836.

A local completed-checkpoint replay also passed. To repeat that limited check:

    python -m pip install -r requirements-u4.txt
    python reproduce.py u4-resume

This recomputes the six 1D integrals and the final sum, using a copy of the
completed adaptive 2D checkpoint. It does not rerun the adaptive 2D grid.
For a fresh adaptive run and the precise scope, see
`docs/U4_S_RETURN_AUDIT.md`. The new record has a different hash from the
unrecovered historical JSON; both identities are documented.

The original common-fan code archive, extracted source, level-5 records,
and AMO verification log are included as historical evidence. The quick
check does not rerun all of these geometric certificates.
Instructions and scope are in CERTIFICATES.md.

## Layout

- src/body5/: standalone interval verifier.
- src/raised_comparison/: unmodified compact verifier, both NPZ inputs,
  and the original manifest/README; this layout preserves its default paths.
- src/u4/: frozen Arb kernels and uniform/hybrid refinement drivers.
- certificates/: original component/endpoint records and common-fan archives.
- results/: fresh audit output, with execution scope stated in each record.
- historical/: older records and the common-fan source archive.
- docs/: mathematical kernel notes, review scope, and publication plan.
- manifest.json: frozen-file hashes, current verification status and historical provenance.

## Attribution and checking

GPT-5.6 Sol, GPT-6 Sol, and Astra assisted selected mathematical steps,
checks, and programming; Claude provided limited LaTeX/editorial suggestions.
The author has understood and checked the main methods, but has not checked
every proof detail or every program. Automated replays do not enlarge that
human verification statement. No supervisor approval or independent peer
review is implied.

The public repository is https://github.com/WangHangang/4d-constant-width-certificates.
The existing Zenodo DOI covers the original raised-Meissner datasets, not all
files in this supplement. See LICENSE.md for the added archive's reuse status.
