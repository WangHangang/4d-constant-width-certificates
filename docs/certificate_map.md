# Certificate map

Each row: mathematical claim -> verifier -> input -> output -> evidence level
(see [`evidence_levels.md`](evidence_levels.md) for the level definitions).

## Raised-Meissner lower bound

```
Claim:    Vol_4(K_RM) > 4437/20000 = 0.22185
Verifier: raised_meissner/src/rm_reduced_exact_local_convexity_audit.py
Input:    rm_reduced_certificate_v2.npz  (+ parent rm_rational_v4_stage4_m16_exact.npz)
Output:   raised_meissner/certificates/rm_reduced_local_convexity_report.json
Evidence: exact arithmetic (Python integers / Fraction only)
```

## U4 volume upper bound

```
Claim:    Vol_4(U4^(1)) < 0.2219
Verifier: u4/src/u4_human_checkable_checkpoint_arb.py
          (imports u4_validated_interval_arb.py, u4_i2_04_centered_arb.py)
Input:    deterministic exact domain decomposition (no external input file)
Output:   u4/certificates/human_checkable_n4096/manifest.json
Evidence: validated interval (Arb ball) arithmetic
```

## Combined theorem-level comparison

```
Vol_4(U4^(1)) <= 0.22183829946417434497071609358086433207758599642028
             <  0.22185
             <  Vol_4(K_RM)
```

The middle inequality (`0.221838... < 0.22185`) is a plain decimal
comparison between two already-rigorous endpoints; it introduces no
additional numerical risk.

## Auxiliary (not part of the above comparison)

```
Object:   raised U_3, center certificate
Verifier: raised_u3/src/raised_u3_certify_centres.py
Status:   source-level checked, certificate output not yet re-verified
          against the current script version (see raised_u3/README.md)
```
