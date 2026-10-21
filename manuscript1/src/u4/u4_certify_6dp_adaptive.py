#!/usr/bin/env python3
"""
Adaptive rigorous six-decimal certificate attempt for Vol_4(U_4^(1)).

Purpose
-------
The existing hybrid certificate already proves the correctly rounded
five-decimal value 0.22184.  Its uncertainty is dominated by the middle-A3
piece I2_04.  This script keeps the already-sharp proof ingredients and
recomputes only I2_04 with a rigorously refined two-dimensional scheme:

  * start from the established 4096 x 4096 transformed grid;
  * use the same centered first-derivative Arb enclosure on a base cell
    whenever its derivative formula is regular;
  * when a base cell would have used the coarse natural-range fallback,
    subdivide that cell into refine x refine exact dyadic subcells and apply
    the centered enclosure again on regular subcells, with natural Arb range
    only on the residual singular/boundary subcells.

Thus every contribution remains an outward Arb enclosure.  The refinement
changes efficiency/width only; it does not change the exact ten-piece
mathematical decomposition.

The script then:
  1. freshly recomputes the six one-dimensional pieces using the frozen
     semianalytic 5dp verifier;
  2. reuses the established rigorous 4096^2 intervals for I2_01, I3_01,
     and I3_02;
  3. inserts the new adaptive I2_04 interval;
  4. performs directed Arb summation and tests correct rounding to 6 decimals.

Evidence level
--------------
EXACT:
  ten-piece split and the full-A3 theta antiderivative used by the six 1D
  pieces.

VALIDATED HERE:
  the adaptive I2_04 enclosure and the six fresh one-dimensional Arb
  integrations.

IMPORTED VALIDATED:
  I2_01, I3_01, I3_02 from the established 4096^2 human-checkable
  certificate.

The final decimal statement is a rigorous hybrid corollary if and only if
the script prints "RIGOROUS 6-DECIMAL CERTIFICATE: SUCCESS".

Dependencies / source identity
------------------------------
Place this file in the SAME directory as the following frozen sources:
  u4_certify_5dp_semianalytic.py
  u4_validated_interval_arb.py
  u4_i2_04_centered_arb.py

Their SHA-256 hashes are checked BEFORE Python imports them.

Recommended first run
---------------------
  python u4_certify_6dp_adaptive.py --base-n 4096 --refine 4 --dps 50 \
      --outdir u4_6dp_adaptive_r4

Resume after interruption
-------------------------
  python u4_certify_6dp_adaptive.py --base-n 4096 --refine 4 --dps 50 \
      --outdir u4_6dp_adaptive_r4 --resume

If refine=4 is still too wide, use a NEW outdir and --refine 8.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Frozen proof-source hashes: verify BEFORE import/execution.
# ---------------------------------------------------------------------------

EXPECTED_SOURCE_HASHES = {
    "u4_certify_5dp_semianalytic.py":
        "c6c7e969c696b69f643e797c953baefb4180b3185bda5ce54b4d19efe3043063",
    "u4_validated_interval_arb.py":
        "a28bd08c9b4b5b6934216b5fde3aa28a87c1c3e00f23893033c82a646f5d4c75",
    "u4_i2_04_centered_arb.py":
        "6e098be65efaa67746e0093887ccd49d4f013345c67f2309b56ec3e497906b2b",
}

ESTABLISHED_OTHER_MIDDLE_2D = {
    "I2_01": (
        "0.005305474445252067648",
        "0.005305524269440644134",
    ),
    "I3_01": (
        "0.0001180335765627251934",
        "0.0001180371686580424704",
    ),
    "I3_02": (
        "0.002053385336922711151",
        "0.002053408608216891725",
    ),
}
OLD_I2_04 = (
    "0.02026090383070103131",
    "0.02026294671750725316",
)
OLD_GLOBAL = (
    "0.2218338853720453",
    "0.2218382994641744",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


HERE = Path(__file__).resolve().parent
for _name, _expected in EXPECTED_SOURCE_HASHES.items():
    _path = HERE / _name
    if not _path.is_file():
        raise SystemExit(
            f"Required frozen source is missing: {_path}\n"
            "Keep this script beside the three source files listed in its header."
        )
    _actual = sha256_file(_path)
    if _actual != _expected:
        raise SystemExit(
            f"REFUSING TO RUN: source hash mismatch for {_name}\n"
            f"  expected: {_expected}\n"
            f"  actual  : {_actual}\n"
            "Use the frozen audited source, or start a new proof audit for the modified file."
        )

# Import only after source bytes have passed the hash gate.
sys.path.insert(0, str(HERE))
try:
    import flint
    from flint import arb, ctx
except Exception as exc:
    raise SystemExit(
        "python-flint is required in this Python environment.\n"
        "Install it with: python -m pip install python-flint\n"
        f"Original import error: {exc}"
    )

P = importlib.import_module("u4_certify_5dp_semianalytic")
K = importlib.import_module("u4_validated_interval_arb")
C = importlib.import_module("u4_i2_04_centered_arb")

# Module-location preflight: no shadow module is allowed.
for _module, _name in (
    (P, "u4_certify_5dp_semianalytic.py"),
    (K, "u4_validated_interval_arb.py"),
    (C, "u4_i2_04_centered_arb.py"),
):
    _resolved = Path(_module.__file__).resolve()
    _expected_path = (HERE / _name).resolve()
    if _resolved != _expected_path:
        raise SystemExit(
            f"REFUSING TO RUN: imported {_name} from unexpected path\n"
            f"  expected: {_expected_path}\n"
            f"  imported: {_resolved}"
        )

ZERO, ONE, TWO = C.ZERO, C.ONE, C.TWO
D2 = C.D2
DerivativeDomainError = C.DerivativeDomainError


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def prevent_windows_sleep():
    if sys.platform != "win32":
        yield
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040
    k32 = ctypes.windll.kernel32
    old = k32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    )
    if old == 0:
        print("WARNING: Windows sleep prevention could not be enabled.", flush=True)
    try:
        yield
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)


def exact_arb_pack(x: arb) -> list[str | int]:
    """Serialize an exact finite Arb x as mantissa * 2**exponent."""
    m, e = x.man_exp()
    return [str(m), int(e)]


def exact_arb_unpack(data: Any) -> arb:
    """Strict inverse of exact_arb_pack after schema validation."""
    if not (
        isinstance(data, list)
        and len(data) == 2
        and isinstance(data[0], str)
        and isinstance(data[1], int)
    ):
        raise ValueError(f"bad exact-Arb encoding: {data!r}")
    int(data[0])  # lexical integer check
    return arb((int(data[0]), data[1]))


def directed_add_lower(a: arb, b: arb) -> arb:
    return (a + b).lower()


def directed_add_upper(a: arb, b: arb) -> arb:
    return (a + b).upper()


def interval_radius(x: arb) -> arb:
    return (x.upper() - x.lower()) / TWO


def endpoint_str(x: arb, digits: int = 55) -> str:
    return x.str(digits, radius=False)


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def imported_interval(lo: str, hi: str) -> arb:
    L, U = arb(lo), arb(hi)
    if L.upper() > U.lower():
        raise AssertionError(f"invalid imported interval [{lo}, {hi}]")
    return L.union(U)


# ---------------------------------------------------------------------------
# Checkpoint schema and preflight
# ---------------------------------------------------------------------------

CHECKPOINT_SCHEMA_VERSION = 1
PIECE_NAME = "I2_04_mid_zB_zC_nu_r2"
INDEXING_CONVENTION = "0-based; next_row is the first unfinished base row in [0, base_n]"
SENTINEL_CONVENTION = "no sentinel values; all counters are explicit nonnegative integers"


def current_meta(args, script_path: Path) -> dict:
    return {
        "base_n": args.base_n,
        "refine": args.refine,
        "dps": args.dps,
        "boundary_layers_refined": args.boundary_layers_refined,
        "script_sha256": sha256_file(script_path),
        "source_sha256": dict(EXPECTED_SOURCE_HASHES),
        "piece": PIECE_NAME,
        "indexing": INDEXING_CONVENTION,
        "sentinels": SENTINEL_CONVENTION,
    }


def fresh_checkpoint(meta: dict) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "meta": meta,
        "next_row": 0,
        "lower_exact": exact_arb_pack(ZERO),
        "upper_exact": exact_arb_pack(ZERO),
        "counts": {
            "base_centered": 0,
            "base_refined": 0,
            "base_derivative_fallbacks": 0,
            "refined_centered": 0,
            "refined_natural": 0,
            "refined_derivative_fallbacks": 0,
        },
        "complete": False,
        "started_unix": time.time(),
        "updated_unix": time.time(),
    }


def validate_checkpoint_schema(state: Any, meta: dict) -> None:
    """
    Explicit preflight for every checkpoint field used by the proof.

    This verifies keys/types, 0-based row range, exact-Arb encoding,
    nonnegative counters, indexing/sentinel conventions, and source hashes.
    """
    if not isinstance(state, dict):
        raise SystemExit("Checkpoint preflight failed: top level must be a JSON object.")

    required_top = {
        "schema_version", "meta", "next_row", "lower_exact", "upper_exact",
        "counts", "complete", "started_unix", "updated_unix",
    }
    missing = required_top - set(state)
    if missing:
        raise SystemExit(f"Checkpoint preflight failed: missing keys {sorted(missing)}")

    if state["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit(
            f"Checkpoint preflight failed: schema_version={state['schema_version']!r}, "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )

    saved_meta = state["meta"]
    if not isinstance(saved_meta, dict):
        raise SystemExit("Checkpoint preflight failed: meta must be an object.")
    for key in (
        "base_n", "refine", "dps", "boundary_layers_refined",
        "script_sha256", "source_sha256", "piece", "indexing", "sentinels",
    ):
        if key not in saved_meta:
            raise SystemExit(f"Checkpoint preflight failed: meta.{key} missing.")
        if saved_meta[key] != meta[key]:
            raise SystemExit(
                "Checkpoint preflight failed: parameter/source mismatch\n"
                f"  {key}: checkpoint={saved_meta[key]!r}\n"
                f"       current   ={meta[key]!r}"
            )

    nr = state["next_row"]
    if not isinstance(nr, int) or not (0 <= nr <= meta["base_n"]):
        raise SystemExit(
            f"Checkpoint preflight failed: next_row={nr!r} is outside "
            f"0..{meta['base_n']} under the documented 0-based convention."
        )

    try:
        lo = exact_arb_unpack(state["lower_exact"])
        hi = exact_arb_unpack(state["upper_exact"])
    except Exception as exc:
        raise SystemExit(f"Checkpoint preflight failed: bad exact endpoints: {exc}")
    if lo > hi:
        raise SystemExit("Checkpoint preflight failed: cumulative lower > upper.")

    counts = state["counts"]
    expected_count_keys = {
        "base_centered", "base_refined", "base_derivative_fallbacks",
        "refined_centered", "refined_natural",
        "refined_derivative_fallbacks",
    }
    if not isinstance(counts, dict) or set(counts) != expected_count_keys:
        raise SystemExit(
            "Checkpoint preflight failed: counts keys differ from schema.\n"
            f"  got     : {sorted(counts) if isinstance(counts, dict) else type(counts)}\n"
            f"  expected: {sorted(expected_count_keys)}"
        )
    for k, v in counts.items():
        if not isinstance(v, int) or v < 0:
            raise SystemExit(f"Checkpoint preflight failed: counts.{k}={v!r}.")

    if not isinstance(state["complete"], bool):
        raise SystemExit("Checkpoint preflight failed: complete must be boolean.")
    if state["complete"] and nr != meta["base_n"]:
        raise SystemExit(
            "Checkpoint preflight failed: complete=true but next_row != base_n."
        )


# ---------------------------------------------------------------------------
# Rigorous cell kernels
# ---------------------------------------------------------------------------

def centered_i204_cell(i: int, j: int, n: int) -> tuple[arb, arb]:
    """
    Rigorous centered first-derivative enclosure on one transformed square.

    Raises DerivativeDomainError / ValueError when the derivative form reaches
    a singular endpoint; the caller then uses a rigorous natural-range fallback.
    """
    wb = K.dyadic_cell(i, n)
    tb = K.dyadic_cell(j, n)
    wc = arb(f"{2*i+1}/{2*n}")
    tc = arb(f"{2*j+1}/{2*n}")

    dval = C.transformed_i2_04_d(D2.w(wb), D2.t(tb))
    rw = interval_radius(dval.dw)
    rt = interval_radius(dval.dt)

    delta = ONE / arb(n)
    area = delta * delta
    half = delta / TWO

    fc = C.natural_i2_04(wc, tc)
    center_contrib = area * fc

    # Centered rectangle lemma:
    # |integral_B f - Delta^2 f(c)| <= Delta^3/4 * (rw+rt).
    err = area * half * (rw + rt) / TWO
    e = err.upper()
    lo = center_contrib.lower() - e
    hi = center_contrib.upper() + e
    return lo, hi


def natural_i204_cell(i: int, j: int, n: int) -> tuple[arb, arb]:
    wb = K.dyadic_cell(i, n)
    tb = K.dyadic_cell(j, n)
    area = (ONE / arb(n)) ** 2
    val = C.natural_i2_04(wb, tb) * area
    return val.lower(), val.upper()


def refine_base_cell(
    i: int,
    j: int,
    base_n: int,
    refine: int,
    boundary_layers_refined: int,
) -> tuple[arb, arb, dict[str, int]]:
    """
    Partition one base cell exactly into refine^2 subcells.

    Global refined indices are
        gi = i*refine + a,  gj = j*refine + b,
    so the refined grid is a disjoint exact partition with denominator
        fine_n = base_n*refine.
    """
    fine_n = base_n * refine
    lo_total = ZERO
    hi_total = ZERO
    counts = {
        "refined_centered": 0,
        "refined_natural": 0,
        "refined_derivative_fallbacks": 0,
    }

    for a in range(refine):
        gi = i * refine + a
        for b in range(refine):
            gj = j * refine + b
            force_natural = (
                gi < boundary_layers_refined
                or gi >= fine_n - boundary_layers_refined
                or gj < boundary_layers_refined
                or gj >= fine_n - boundary_layers_refined
            )
            if not force_natural:
                try:
                    lo, hi = centered_i204_cell(gi, gj, fine_n)
                    counts["refined_centered"] += 1
                except (DerivativeDomainError, ZeroDivisionError, ValueError):
                    force_natural = True
                    counts["refined_derivative_fallbacks"] += 1

            if force_natural:
                lo, hi = natural_i204_cell(gi, gj, fine_n)
                counts["refined_natural"] += 1

            lo_total = directed_add_lower(lo_total, lo)
            hi_total = directed_add_upper(hi_total, hi)

    return lo_total, hi_total, counts


def run_adaptive_i204(args, checkpoint_path: Path, meta: dict) -> tuple[arb, arb, dict]:
    if args.resume:
        if not checkpoint_path.is_file():
            raise SystemExit(f"--resume requested but checkpoint not found: {checkpoint_path}")
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        validate_checkpoint_schema(state, meta)
        if state["complete"]:
            print(f"Checkpoint already complete: {checkpoint_path}", flush=True)
            return (
                exact_arb_unpack(state["lower_exact"]),
                exact_arb_unpack(state["upper_exact"]),
                state,
            )
        print(
            f"RESUME: next base row={state['next_row']} / {args.base_n}",
            flush=True,
        )
    else:
        if checkpoint_path.exists():
            raise SystemExit(
                f"Checkpoint already exists: {checkpoint_path}\n"
                "Use --resume, or choose a new --outdir."
            )
        state = fresh_checkpoint(meta)
        atomic_write_json(checkpoint_path, state)

    start_row = state["next_row"]
    total_lo = exact_arb_unpack(state["lower_exact"])
    total_hi = exact_arb_unpack(state["upper_exact"])
    counts = dict(state["counts"])

    t0 = time.time()
    rows_this_invocation = 0

    print("\n=== adaptive rigorous I2_04 refinement ===")
    print(f"base grid      : {args.base_n} x {args.base_n}")
    print(f"fallback refine: {args.refine} x {args.refine}")
    print(f"fine denominator: {args.base_n * args.refine}")
    print(f"resume row     : {start_row}")
    print(flush=True)

    for i in range(start_row, args.base_n):
        row_lo = ZERO
        row_hi = ZERO

        for j in range(args.base_n):
            base_force_refine = (
                i == 0 or i == args.base_n - 1
                or j == 0 or j == args.base_n - 1
            )

            if not base_force_refine:
                try:
                    lo, hi = centered_i204_cell(i, j, args.base_n)
                    counts["base_centered"] += 1
                except (DerivativeDomainError, ZeroDivisionError, ValueError):
                    base_force_refine = True
                    counts["base_derivative_fallbacks"] += 1

            if base_force_refine:
                lo, hi, subcounts = refine_base_cell(
                    i, j,
                    args.base_n,
                    args.refine,
                    args.boundary_layers_refined,
                )
                counts["base_refined"] += 1
                for k, v in subcounts.items():
                    counts[k] += v

            row_lo = directed_add_lower(row_lo, lo)
            row_hi = directed_add_upper(row_hi, hi)

        total_lo = directed_add_lower(total_lo, row_lo)
        total_hi = directed_add_upper(total_hi, row_hi)
        rows_this_invocation += 1

        done = i + 1
        if (
            args.checkpoint_every_rows
            and (done % args.checkpoint_every_rows == 0 or done == args.base_n)
        ):
            state["next_row"] = done
            state["lower_exact"] = exact_arb_pack(total_lo)
            state["upper_exact"] = exact_arb_pack(total_hi)
            state["counts"] = dict(counts)
            state["complete"] = (done == args.base_n)
            state["updated_unix"] = time.time()
            atomic_write_json(checkpoint_path, state)

        if (
            args.report_every_rows
            and (done % args.report_every_rows == 0 or done == args.base_n)
        ):
            elapsed = time.time() - t0
            rate = rows_this_invocation / elapsed if elapsed > 0 else 0.0
            remaining = args.base_n - done
            eta = remaining / rate if rate > 0 else float("inf")
            width_now = total_hi - total_lo
            print(
                f"rows {done:5d}/{args.base_n}; "
                f"base-centered={counts['base_centered']}; "
                f"base-refined={counts['base_refined']}; "
                f"fine-centered={counts['refined_centered']}; "
                f"fine-natural={counts['refined_natural']}; "
                f"width(partial)={endpoint_str(width_now, 10)}; "
                f"elapsed={elapsed/60:.1f} min; "
                f"rough ETA={eta/60:.1f} min",
                flush=True,
            )

    # Final state even if checkpoint cadence was disabled.
    state["next_row"] = args.base_n
    state["lower_exact"] = exact_arb_pack(total_lo)
    state["upper_exact"] = exact_arb_pack(total_hi)
    state["counts"] = dict(counts)
    state["complete"] = True
    state["updated_unix"] = time.time()
    atomic_write_json(checkpoint_path, state)

    return total_lo, total_hi, state


# ---------------------------------------------------------------------------
# Fresh six 1D pieces and final six-decimal test
# ---------------------------------------------------------------------------

def fresh_one_dimensional_pieces(args) -> dict[str, arb]:
    tol = 10.0 ** (-args.tol_digits)
    jobs = [
        ("I1_01", lambda x, a: P.i1_piece(x, P.zD, P.zC, "lens", a)),
        ("I1_02", lambda x, a: P.i1_piece(x, P.zC, P.zE, "linear", a)),
        ("I1_03", lambda x, a: P.i1_piece(x, P.zE, P.ONE, "full", a)),
        ("I2_02", lambda x, a: P.i2_full_piece(x, P.zB, P.zD, "I2_02", a)),
        ("I2_03", lambda x, a: P.i2_full_piece(x, P.zD, P.zC, "I2_03", a)),
        ("I2_05", lambda x, a: P.i2_full_piece(x, P.zC, P.ONE, "I2_05", a)),
    ]

    fresh = {}
    print("\n=== fresh rigorous 1D pieces ===", flush=True)
    for idx, (name, fun) in enumerate(jobs, 1):
        ts = time.time()
        val = P.integrate_1d(fun, tol, args)
        real = P.verify_real_integral(name, val)
        fresh[name] = real
        print(
            f"[{idx}/6] {name}: {real.str(55)}; "
            f"width={P.width(real).str(20, radius=False)}; "
            f"time={time.time()-ts:.3f}s",
            flush=True,
        )
    return fresh


def final_hybrid(args, i204_lo: arb, i204_hi: arb) -> tuple[dict, dict]:
    # Validate the newly computed bottleneck against the established theorem.
    old_i204 = imported_interval(*OLD_I2_04)
    new_i204 = i204_lo.union(i204_hi)
    if not (
        new_i204.lower() >= old_i204.lower()
        and new_i204.upper() <= old_i204.upper()
    ):
        raise RuntimeError(
            "New adaptive I2_04 enclosure is not contained in the established "
            "4096^2 interval. Stop and audit before using it."
        )

    fresh = fresh_one_dimensional_pieces(args)
    other = {
        name: imported_interval(*bounds)
        for name, bounds in ESTABLISHED_OTHER_MIDDLE_2D.items()
    }

    required = {
        "I1_01", "I1_02", "I1_03",
        "I2_01", "I2_02", "I2_03", "I2_04", "I2_05",
        "I3_01", "I3_02",
    }
    pieces = {
        **fresh,
        **other,
        "I2_04": new_i204,
    }
    if set(pieces) != required:
        raise AssertionError("ten-piece accounting failed")

    I1 = fresh["I1_01"] + fresh["I1_02"] + fresh["I1_03"]
    I2 = (
        other["I2_01"]
        + fresh["I2_02"]
        + fresh["I2_03"]
        + new_i204
        + fresh["I2_05"]
    )
    I3 = other["I3_01"] + other["I3_02"]
    total = I1 + I2 + I3

    old_global = imported_interval(*OLD_GLOBAL)
    if not (
        total.lower() >= old_global.lower()
        and total.upper() <= old_global.upper()
    ):
        raise RuntimeError(
            "Final hybrid interval is not contained in the previous global "
            "rigorous theorem interval."
        )

    cert5 = P.certify_rounding(total, 5)
    cert6 = P.certify_rounding(total, 6)
    cert7 = P.certify_rounding(total, 7)

    print("\n=== rigorous adaptive hybrid sum ===")
    print(f"I1    : {I1.str(60)}")
    print(f"I2    : {I2.str(60)}")
    print(f"I3    : {I3.str(60)}")
    print(f"TOTAL : {total.str(70)}")
    print(f"width : {P.width(total).str(50, radius=False)}")
    print()
    for d, cert in ((5, cert5), (6, cert6), (7, cert7)):
        print(
            f"{d} decimals: {cert['candidate']}  "
            f"cell=({cert['left']}, {cert['right']})  "
            f"certified={'YES' if cert['certified'] else 'NO'}"
        )

    result = {
        "components": {
            "I1": I1.str(90),
            "I2": I2.str(90),
            "I3": I3.str(90),
        },
        "total": {
            "ball": total.str(100),
            "lower": total.lower().str(100, radius=False),
            "upper": total.upper().str(100, radius=False),
            "width": P.width(total).str(100, radius=False),
        },
        "rounding_5dp": cert5,
        "rounding_6dp": cert6,
        "rounding_7dp": cert7,
        "fresh_1d": {
            k: {
                "ball": v.str(90),
                "lower": v.lower().str(90, radius=False),
                "upper": v.upper().str(90, radius=False),
                "width": P.width(v).str(90, radius=False),
            }
            for k, v in fresh.items()
        },
        "imported_other_middle_2d": ESTABLISHED_OTHER_MIDDLE_2D,
    }
    return result, {"5": cert5, "6": cert6, "7": cert7}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Adaptive rigorous attempt to certify Vol_4(U4^(1)) to six decimals."
    )
    ap.add_argument("--base-n", type=int, default=4096)
    ap.add_argument("--refine", type=int, default=4)
    ap.add_argument("--dps", type=int, default=50)
    ap.add_argument("--boundary-layers-refined", type=int, default=1)
    ap.add_argument("--checkpoint-every-rows", type=int, default=8)
    ap.add_argument("--report-every-rows", type=int, default=32)
    ap.add_argument("--tol-digits", type=int, default=13)
    ap.add_argument("--eval-limit", type=int, default=500000)
    ap.add_argument("--depth-limit", type=int, default=200)
    ap.add_argument("--deg-limit", type=int, default=120)
    ap.add_argument("--outdir", default="u4_6dp_adaptive_r4")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # Parameter preflight.  Powers of two preserve the exact dyadic partition
    # used by the audited kernel.
    if args.base_n <= 0 or (args.base_n & (args.base_n - 1)):
        raise SystemExit("--base-n must be a positive power of two.")
    if args.refine <= 0 or (args.refine & (args.refine - 1)):
        raise SystemExit("--refine must be a positive power of two.")
    if args.dps < 35:
        raise SystemExit("Use --dps >= 35.")
    if args.boundary_layers_refined < 1:
        raise SystemExit("Use --boundary-layers-refined >= 1.")
    if 2 * args.boundary_layers_refined >= args.base_n * args.refine:
        raise SystemExit("Refined boundary layer is too large.")
    if args.tol_digits < 10:
        raise SystemExit("Use --tol-digits >= 10.")

    ctx.dps = args.dps

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "u4_i2_04_adaptive_checkpoint.json"
    result_path = outdir / "u4_6dp_adaptive_certificate.json"

    script_path = Path(__file__).resolve()
    meta = current_meta(args, script_path)

    print("=== U4 adaptive rigorous 6-decimal verifier ===")
    print(f"Python       : {sys.version.split()[0]}")
    print(f"python-flint : {getattr(flint, '__version__', 'unknown')}")
    print(f"platform     : {platform.platform()}")
    print(f"Arb dps      : {args.dps}")
    print(f"base grid    : {args.base_n}^2")
    print(f"refine       : {args.refine}^2 on base fallback cells")
    print(f"script SHA256: {meta['script_sha256']}")
    print("Verified source hashes:")
    for name, digest in EXPECTED_SOURCE_HASHES.items():
        print(f"  {name}: {digest}")
    print()
    print("Checkpoint schema preflight:")
    print(f"  schema version : {CHECKPOINT_SCHEMA_VERSION}")
    print(f"  indexing       : {INDEXING_CONVENTION}")
    print(f"  sentinels      : {SENTINEL_CONVENTION}")
    print(f"  checkpoint     : {checkpoint_path}")
    print()
    print("Trust boundary:")
    print("  exact      : ten-piece split + full-A3 theta antiderivative")
    print("  fresh Arb  : six 1D pieces + adaptively refined I2_04")
    print("  prior Arb  : I2_01, I3_01, I3_02 from established 4096^2 certificate")
    print(flush=True)

    t0 = time.time()
    with prevent_windows_sleep():
        i204_lo, i204_hi, checkpoint_state = run_adaptive_i204(
            args, checkpoint_path, meta
        )

        print("\n=== completed rigorous I2_04 enclosure ===")
        print(f"lower = {endpoint_str(i204_lo, 70)}")
        print(f"upper = {endpoint_str(i204_hi, 70)}")
        print(f"width = {endpoint_str(i204_hi - i204_lo, 55)}")
        print(f"counts = {checkpoint_state['counts']}")
        print(flush=True)

        final_result, certs = final_hybrid(args, i204_lo, i204_hi)

    manifest = {
        "schema_version": 1,
        "statement": "adaptive rigorous decimal certificate attempt for Vol_4(U_4^(1))",
        "script": script_path.name,
        "script_sha256": meta["script_sha256"],
        "source_sha256": EXPECTED_SOURCE_HASHES,
        "python": sys.version,
        "python_flint": getattr(flint, "__version__", "unknown"),
        "platform": platform.platform(),
        "parameters": {
            "base_n": args.base_n,
            "refine": args.refine,
            "fine_n": args.base_n * args.refine,
            "dps": args.dps,
            "boundary_layers_refined": args.boundary_layers_refined,
            "tol_digits": args.tol_digits,
        },
        "checkpoint_schema": {
            "version": CHECKPOINT_SCHEMA_VERSION,
            "indexing": INDEXING_CONVENTION,
            "sentinels": SENTINEL_CONVENTION,
        },
        "i2_04": {
            "lower": endpoint_str(i204_lo, 100),
            "upper": endpoint_str(i204_hi, 100),
            "width": endpoint_str(i204_hi - i204_lo, 100),
            "counts": checkpoint_state["counts"],
        },
        **final_result,
        "elapsed_this_invocation_seconds": time.time() - t0,
        "trust_boundary": {
            "exact": "ten-piece split and full-A3 theta antiderivative",
            "fresh_validated": "six 1D Arb pieces and adaptive I2_04",
            "imported_validated": "I2_01, I3_01, I3_02 from established 4096^2 certificate",
            "numerical_seed": (
                "high-precision point value is used only to select/report a rounding cell; "
                "it is not part of the proof"
            ),
        },
    }
    atomic_write_json(result_path, manifest)

    print(f"\nSaved result : {result_path.resolve()}")
    print(f"Result SHA256: {sha256_file(result_path)}")
    print(f"Checkpoint   : {checkpoint_path.resolve()}")

    if certs["6"]["certified"]:
        print("\n============================================================")
        print("RIGOROUS 6-DECIMAL CERTIFICATE: SUCCESS")
        print(
            f"Vol_4(U4^(1)) = {certs['6']['candidate']}  "
            "(correctly rounded to 6 decimals)"
        )
        print("============================================================")
        return 0

    print("\nSix-decimal rounding is not yet certified.")
    print(
        "The new interval is still rigorous.  If its width is the only issue, "
        "rerun from a NEW outdir with --refine 8."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
