# Raised-body certificates for the full comparison - revision 2

Use this revision instead of the earlier email attachments. The earlier
lower-only certificates remain valid. This package additionally verifies
the upper bounds and a stronger U3 lower bound needed for the full ordering.
The accompanying six-page explanation introduces the objects and gives
precise references to the updated manuscript.

## Run

Install Python packages `numpy`, `numba`, and `mpmath`, then extract all files
into one directory and run:

```
python -u certify_comparison.py
```

There is one self-contained Python checker and two data files. No SciPy,
Qhull, local helper module, or proposal/search program is required. Initial
Numba compilation is included in the runtime. The program prints progress,
prevents automatic Windows system sleep while running, and writes a new
`verification_result.json` only after all final comparisons pass.

## What is independently rechecked

1. Raised Meissner lower: all 105,654 point memberships; all 708,899 boundary
   tetrahedra; topology; all 814,553 exact local convexity tests; exact volume.
2. Raised Meissner upper: complete exact domain coverage; all 716,800 positive
   and 1,650,688 negative cubature leaves; continuous-arc centre membership;
   fully enclosed Gauss nodes, integrands, weights and fourth-order remainders.
3. Raised U3 lower: 800,442 rational-ray endpoints per half; continuous positive
   arc constraints; exact negative witnesses; exact integer simplex-volume sum.
4. Raised U3 upper: all 435 centre seeds and their exact 10,141-point symmetry
   orbit; coverage and every one of 86,016 cubature leaves.

All global upper sums and all U3 simplex sums use exact integer/rational
reductions. The checker does not accept stored successful predicates, old
aggregate totals, old per-box bounds, or floating combinatorics as proofs.
Delta encoding and LZMA compression are storage only; NPZ remains a standard
ZIP of NPY arrays, decoded by Python's standard library. Witness precision was
reduced for a small attachment and every resulting point is rechecked anew.

## Scope and existing inputs

The raised certificates combine with the existing validated U4 upper and
Body 5 enclosure to establish:

```
Vol(U4) < Vol(Raised Meissner) < Vol(Raised U3) < Vol(Body 5) < Vol(M4).
```

The following conservative endpoints are supplied by those existing
certificates, not recomputed by this raised-body checker:

- U4 upper: 0.221838300.
- Body 5 lower: 0.222877646477730.
- Body 5 upper: 0.222991176426990.

The Body 5 source is `body5_independent_validated_certificate.py` from the
manuscript handoff; its source SHA-256 is
`644097e6488a8e8200c1d7b77aa36611e9382400cca56d67bab2775e98ef26e9`.
The M4 endpoint is checked against its established exact volume formula.
AMO is not part of this original five-candidate ordering; its existing
independent common-fan certificate still establishes U4 < AMO.

Thus this package also supplies the raised lower certificates needed for
the principal U4-smallest comparison. The previous small lower-only package
remains valid, but this package adds the stronger U3 lower and both uppers.

See the accompanying six-page mathematical note for the proof and explicit
arithmetic trust boundary. `CERTIFICATE_MANIFEST.json` records hashes and the
successful full replay on the preparing host; the checker recomputes the proof
instead of trusting that manifest. A second fresh replay on 29 September
passed every check in 536.6 seconds (about nine minutes on the preparing host).
The executable mathematics and both data files are unchanged from the
repaired full-comparison package; a misleading header concerning the RM
lower input has been corrected. RM lower is independently rechecked here.

## Suggested reading

Read Sections 1-2 of the note for the definitions and common inner-ball
argument; Sections 3-4 construct the lower certificates. Section 5 explains
the principal U4-smallest comparison. Section 6 is needed only for the
stronger original ordering; Section 7 gives the reproduction command and
manuscript locations. The manuscript's current full ordering is Theorem 105,
equation (768), and its compact proof is Section 13.6. Appendix J.1 records
this checker; Appendices J.2 and J.3 record the external U4 and Body 5 inputs.

The common-fan k=5 bounds miss the threshold for both raised bodies.
The supplied email establishes an adaptive memory failure for Raised U3,
not an adaptive failure for raised Meissner. Neither observation proves
that the rational-sample method is impossible; the present certificates
make further fan-only refinement unnecessary for these comparisons.
