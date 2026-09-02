# Data schema

`.npz` files in this project are proof objects, not incidental numerical
data: their keys, shapes, dtypes, and indexing conventions are part of the
certificate's contract with its verifier. This document records the schema
enforced by each verifier's own preflight checks (see the `load_*` /
preflight functions in the corresponding `src/` script) plus the concrete
counts from the one certified run. If you regenerate a `.npz`, re-derive
counts from your own run's report rather than copying the numbers below.

## `rm_reduced_certificate_v2.npz` (raised_meissner, "reduced")

Verified by `raised_meissner/src/rm_reduced_exact_local_convexity_audit.py`
(`load_reduced`).

| key | dtype | shape | meaning |
|---|---|---|---|
| `points_num` | integer | `(N, 4)` | Numerators of the N rational vertex coordinates in R^4 |
| `denominator` | integer, scalar | `()` | Common denominator `D` for all `points_num` (must be > 0) |
| `source_indices` | integer | `(N,)` | Index of each reduced point in the *parent* file's `numerators` array; must be duplicate-free |
| `qhull_simplices` | integer | `(F, 4)` | Zero-based local vertex indices (into `points_num`) of each boundary tetrahedron; every value in `[0, N-1]`, no repeated vertex within a row |
| `exact_det_sum_S` | integer (as string), optional | scalar | Precomputed exact determinant sum, cross-checked against the recomputed value |

Additional preflight invariants enforced before any geometric work begins:
every point in `points_num` must appear in `qhull_simplices` (no orphan
vertices), and the used vertex labels must be exactly `0, ..., N-1`.

Certified run (2026-08-11): `N = 105,654` points, `F = 708,899` facets,
`E = 814,553` unique edges, `D = 1,048,576`.

## `rm_rational_v4_stage4_m16_exact.npz` (raised_meissner, "parent")

| key | dtype | shape | meaning |
|---|---|---|---|
| `numerators` | integer | `(N_parent, 4)` | Numerators of all candidate exact-rational vertices |
| `denominator` | integer, scalar, optional | `()` | Must equal the reduced file's `denominator` if present |

Cross-check performed by the reduced-certificate verifier:
`points_num == numerators[source_indices]` exactly, element-wise.

## U4 certificate JSON (not `.npz` — plain JSON)

`u4/certificates/human_checkable_n4096/checkpoint.json` (see
`fresh_state` / `store_acc` in `u4_human_checkable_checkpoint_arb.py`):

```
{
  "format_version": 1,
  "meta": { n1, n2, dps, boundary_layers, target,
            script_sha256, kernel_sha256, centered_kernel_sha256, python_flint },
  "phase": "1d" | "2d" | "done",
  "piece_index": int,
  "next_index": int,
  "pieces": {
    "<piece name>": {
      "lower_exact": [mantissa_str, exponent_int],   # x = mantissa * 2**exponent, exact
      "upper_exact": [mantissa_str, exponent_int],
      "centered_boxes": int, "natural_boxes": int, "ad_fallbacks": int,
      "complete": bool
    }, ...
  }
}
```

`manifest.json` additionally records the final `total_lower`/`total_upper`
(both as decimal strings and as exact `[mantissa, exponent]` pairs),
per-component (`I1`/`I2`/`I3`) bounds, and `strict_upper_target_proved`.

Resume safety: `verify_resume_meta` refuses to continue a checkpoint whose
`n1`/`n2`/`dps`/`boundary_layers`/`target`/source-file SHA-256 values differ
from the current invocation's — a resumed run cannot silently mix
parameters or code versions.
