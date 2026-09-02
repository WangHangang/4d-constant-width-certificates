# Raised Meissner: exact lower-bound certificate

Certifies, by exact integer/rational arithmetic only, that

```
Vol_4(K_RM) > 4437/20000 = 0.22185
```

## Proof chain

```
parent rational vertex data (rm_rational_v4_stage4_m16_exact.npz)
        |
        v
reduced rational certificate (rm_reduced_certificate_v2.npz)
        |
        v
exact local-convexity / PL-manifold audit  (src/rm_reduced_exact_local_convexity_audit.py)
        |
        v
exact determinant volume sum
        |
        v
exact rational threshold comparison  ->  rigorous lower bound
```

Qhull is used only as a *candidate* combinatorics generator upstream (not
included in this repository). Every property the proof actually needs
(closed/pure PL hypersurface, ridge incidence, dual connectivity, exact
local convexity of every edge star, boundedness, origin interior, exact
determinant sum) is independently recomputed and checked by
`rm_reduced_exact_local_convexity_audit.py` using exact Python integers only.
See [`../docs/trust_boundaries.md`](../docs/trust_boundaries.md).

## Files

| File | Role |
|---|---|
| `src/rm_reduced_exact_local_convexity_audit.py` | Verifier. Loads the certificate data, recomputes all topology/convexity checks, prints `EXACT_LOCAL_CONVEXITY_CERTIFICATE_SUCCESS` on success. |
| `certificates/rm_reduced_local_convexity_report.json` | Output of a successful run: point/facet/edge counts, exact determinant sum `S`, SHA-256 of both input `.npz` files, pass/fail. |
| `certificates/rm_reduced_local_convexity_checkpoint.json` | Resume checkpoint from the completed run (814,553/814,553 edges). |
| `certificates/rm_reduced_certificate_v2_report.txt` | Human-readable summary of the reduced certificate. |
| `data/` | Pointer to the Zenodo-archived `.npz` inputs (not stored here). |

## Reproduction

```bash
python rm_reduced_exact_local_convexity_audit.py \
    rm_reduced_certificate_v2.npz \
    --parent rm_rational_v4_stage4_m16_exact.npz
```

Both `.npz` files are obtained from the Zenodo release referenced in
[`data/README.md`](data/README.md); their SHA-256 hashes are recorded in
`certificates/rm_reduced_local_convexity_report.json` and must match before
the audit proceeds (the script checks this itself).
