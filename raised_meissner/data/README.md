# Raised Meissner: certificate data

The two `.npz` files consumed by `../src/rm_reduced_exact_local_convexity_audit.py`
are archived immutably on Zenodo rather than committed to this git history
(they total ~79 MB and are proof *data*, not source).

- `rm_reduced_certificate_v2.npz` — final reduced rational certificate (~32 MB)
- `rm_rational_v4_stage4_m16_exact.npz` — parent exact-rational fan (~47 MB),
  needed because the reduced audit also verifies that every reduced point's
  numerators match `parent[numerators][source_indices]` exactly.

**Zenodo DOI: to be added once the release is minted.**

Until then, expected SHA-256 hashes (from
`../certificates/rm_reduced_local_convexity_report.json`):

```
reduced (rm_reduced_certificate_v2.npz):
  4baaa9e5a9da601992be574ff52fa84200f99d300176e00d36f9f46410468df6

parent (rm_rational_v4_stage4_m16_exact.npz):
  15158ac64ede3f650af85a10f9277c42db30bb09fb704f2d042f8c368fe79a7f
```

Do not substitute files with matching names but different hashes; the
verifier checks the hashes but a reader fetching files manually should too.
