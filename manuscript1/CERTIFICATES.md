# Proof-to-file index

The identifiers agree with the manuscript's reproducibility table.
Hashes are in manifest.json; they establish file identity, not correctness.

| ID | Claim | Files and verification route |
|---|---|---|
| U4-G | 0.2218338853720453 < V(U4) < 0.2218382994641744 | src/u4/u4_human_checkable_checkpoint_arb.py and its two kernels; certificates/u4_grid/ contains the 60-digit exact records. |
| U4-S | Sharper six-decimal enclosure reported in the paper | src/u4/u4_certify_6dp_adaptive.py plus frozen dependencies; the final terminal JSON is pending recovery. Do not substitute the five-decimal record. |
| RM-L | Exact inner-polytope lower bound for raised Meissner | src/raised_comparison/certify_comparison.py and rm_lower.npz. Complete membership, topology, all 814,553 edge checks, and determinant sum are recomputed. |
| R-C | Both raised-body upper bounds and final n=80 raised-U3 lower bound | src/raised_comparison/certify_comparison.py and raised_comparison_compact.npz, together with rm_lower.npz. Use python reproduce.py raised. |
| B5 | Boundary-flux interval enclosing Body5 | src/body5/body5_independent_validated_certificate.py; use python reproduce.py body5. |
| FAN | Common-fan lower thresholds, including AMO | historical/common_fan_code_20260907.zip and extracted historical/common_fan/; certificates/common_fan_k5.zip, AMO_k5.zip, and AMO_k5_verify.log. |

## Exact comparisons

The raised-Meissner inner polytope has volume S sqrt(2)/(192 D^4),
where S=36415211368949752100591187 and D=1048576.
The separator 4437/20000 is checked by squaring positive quantities and
comparing integers. The final raised-U3 lower uses rational rays with
denominator 2^42, mesh n=80, and an arc grid of 2048.
The older historical/raised_u3_earlier_n80.json uses a different
interval-determinant construction and is not this final certificate.

The compact upper sums and final U3 lower sum are exact fractions in its
output. Decimal display values are never used to decide their separation.

The original Body5 driver compares against 0.222868116845342.
The paper uses the slightly more conservative 0.222868116845350.
The quick check also verifies separation from that current endpoint.

## Full uniform-grid replay (optional, expensive)

    python -m pip install -r requirements-u4.txt
    python src/u4/u4_human_checkable_checkpoint_arb.py --n1 1048576 --n2 4096 --dps 60 --boundary-layers 1 --outdir runs/u4_grid_fresh

Use a new output directory for a fresh replay. The original grid checkpoint
is kept under certificates/ and is never silently passed as a new run.
The ten pieces consist of three I1 pieces, five I2 pieces, and two I3 pieces.

## Recovering or recomputing U4-S

Missing filename: u4_6dp_adaptive_certificate.json
Recorded SHA-256:
b787db8a5fadc05b895e8c2b57567fd19139fb4982cbacb19f4f17a0c426f1b7

The source first checks three frozen dependency hashes. A new computation
could produce a different JSON hash because it records run metadata;
it must be identified as a new certificate, not a recovered historical file.
The driver checks correct rounding and emits a success token only if the
validated enclosure is sufficiently narrow. A recomputation has not been
performed in this packaging pass.

## Common-fan replay

Consult the original README and verifier interfaces under
historical/common_fan/bl4_uniform/. Those September instructions retain
their historical status; they do not describe the later compact raised-body
proof. Preserve both coordinate normalization and dyadic leaf ordering.
The current quick check validates original archive identities only.
It does not infer a geometric proof from a stored success flag.
