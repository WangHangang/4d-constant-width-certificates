# U4: validated upper-bound certificate

Certifies, by rigorous ball (interval) arithmetic — not statistics, not a
floating-point point estimate — that

```
Vol_4(U4^(1)) in [0.22183388537204538611501634927673329882847033767573,
                   0.22183829946417434497071609358086433207758599642028]
             <  0.2219
```

using [python-flint](https://python-flint.readthedocs.io/) / Arb, which
represents every real number as an interval guaranteed to contain the true
value.

## Proof chain

```
exact support-function integral
        |  analytic reduction (piecewise, exact domain split)
        v
I1 + I2 + I3   (3 one-dimensional pieces + 7 two-dimensional pieces)
        |  Arb rigorous range enclosure on a uniform grid,
        |  with a centered first-derivative refinement on interior boxes
        v
directed lower/upper endpoint summation
        v
rigorous enclosure of Vol_4(U4^(1))
```

## Files

| File | Role |
|---|---|
| `src/u4_validated_interval_arb.py` | Kernel: exact constants, branch boundaries, endpoint-regularizing maps, the ten proof pieces, natural-interval box integration. Imported by the other two scripts. |
| `src/u4_i2_04_centered_arb.py` | Centered first-derivative refinement for the single hardest 2D piece (`I2_04`), with natural-interval fallback near boundaries/singularities. |
| `src/u4_human_checkable_checkpoint_arb.py` | The certificate driver used for the theorem endpoint above. Uniform-grid, same centered-derivative kernel on all seven 2D pieces, checkpoint/resume with exact binary (`man_exp`) endpoint serialization. |
| `certificates/human_checkable_n4096/` | Completed run (`n1=1048576`, `n2=4096`, `dps=60`): `checkpoint.json` (final state, `phase: "done"`) and `manifest.json` (final enclosure, per-component bounds, source-file SHA-256, `python-flint` version). This is the theorem-level certificate. |
| `certificates/i2_04_centered_n1024/manifest.json` | Independent standalone run of `u4_i2_04_centered_arb.py` at `n=1024`, tighter than the n=4096 driver's contribution for that piece — kept as a stronger cross-check. |
| `cross_checks/i2_04_centered_n256/manifest.json` | Same script at `n=256`, a coarser independent consistency check, not used as the theorem endpoint. |

## Dependency closure

`u4_human_checkable_checkpoint_arb.py` imports exactly:

```
import u4_validated_interval_arb as K
import u4_i2_04_centered_arb as C
```

No other local module is on the import path of the certified run.

## Reproduction

```bash
python u4_human_checkable_checkpoint_arb.py \
    --n1 1048576 --n2 4096 --dps 60 \
    --checkpoint-every-rows 16 \
    --outdir u4_human_checkable_n4096
```

Re-running with `--resume` continues from `certificates/human_checkable_n4096/checkpoint.json`
if parameters and source-file hashes still match (the script refuses to
resume otherwise).
