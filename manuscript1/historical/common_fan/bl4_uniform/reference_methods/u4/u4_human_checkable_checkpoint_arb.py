#!/usr/bin/env python3
"""
Human-checkable rigorous Python/Arb verifier for Vol_4(U4^(1)) < target.

Purpose
-------
This is the deliberately simple, uniform-grid certificate intended for the
"easy to check by a human; longer computation is acceptable" version of the
U4 proof.

It uses only two enclosure lemmas after the exact ten-piece reduction already
implemented in u4_validated_interval_arb.py:

(1) One-dimensional natural interval enclosure.
    If J is a cell and f(J) is an Arb enclosure of f on J, then

        |J| * lower(f(J)) <= int_J f <= |J| * upper(f(J)).

(2) Two-dimensional centered first-derivative enclosure.
    For a square B of side Delta and centre c, let D_w and D_t enclose the
    two first partial derivatives of F on B.  If r_w and r_t are the radii of
    those derivative intervals, symmetry of B gives

        | int_B F - |B| F(c) | <= Delta^3/4 * (r_w + r_t).

    If a derivative formula reaches a transformed-domain singular endpoint,
    that box falls back to the natural interval enclosure from (1), now in 2D.

The same centered proof kernel is used for ALL seven two-dimensional pieces.
No semi-analytic special treatment is needed in this verifier.  This makes the
certificate slower, but the mathematical logic is uniform and short.

Checkpoint / resume
-------------------
The run is checkpointed atomically after configurable row/chunk intervals.
The checkpoint stores every cumulative rigorous endpoint as an EXACT binary
floating-point number using arb.lower()/upper() + man_exp().  Thus resuming
introduces no decimal serialization loss.

Typical overnight run on the user's current machine:

  & "D:\\桌面搬运\\Canada001\\Presentation\\.venv311\\Scripts\\python.exe" `
    "D:\\桌面搬运\\Canada001\\Presentation\\u4_human_checkable_checkpoint_arb.py" `
    --n1 1048576 --n2 4096 --dps 60 `
    --checkpoint-every-rows 16 `
    --outdir "D:\\桌面搬运\\Canada001\\Presentation\\u4_human_checkable_n4096"

If interrupted, run the SAME command with --resume.  The program refuses to
resume if parameters or source-file hashes have changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

try:
    import flint
    from flint import arb, ctx
except Exception as exc:
    raise SystemExit(
        "python-flint is required in THIS Python environment.\n"
        f"Original import error: {exc}"
    )

try:
    import u4_validated_interval_arb as K
    import u4_i2_04_centered_arb as C
except Exception as exc:
    raise SystemExit(
        "Keep this file beside u4_validated_interval_arb.py and "
        "u4_i2_04_centered_arb.py.\n"
        f"Original import error: {exc}"
    )

# Reuse the already-audited first-order interval-AD primitive.
D2 = C.D2
DerivativeDomainError = C.DerivativeDomainError
ZERO, ONE, TWO = C.ZERO, C.ONE, C.TWO
THREE, FOUR, FIVE, SIX = C.THREE, C.FOUR, C.FIVE, C.SIX
TEN, SIXTY_FOUR = C.TEN, C.SIXTY_FOUR
PI, SQRT2 = C.PI, C.SQRT2
SQRT_6_OVER_5 = C.SQRT_6_OVER_5


# ---------------------------------------------------------------------------
# Hashing and exact checkpoint serialization
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exact_arb_pack(x: arb) -> list[str | int]:
    """Serialize an exact finite Arb x as x = mantissa * 2**exponent."""
    m, e = x.man_exp()
    return [str(m), int(e)]


def exact_arb_unpack(data: list[str | int]) -> arb:
    """Inverse of exact_arb_pack; arb(tuple) accepts exact binary data."""
    return arb((int(data[0]), int(data[1])))


def directed_add_lower(a: arb, b: arb) -> arb:
    """Exact directed lower endpoint of a+b at current Arb precision."""
    return (a + b).lower()


def directed_add_upper(a: arb, b: arb) -> arb:
    """Exact directed upper endpoint of a+b at current Arb precision."""
    return (a + b).upper()


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def endpoint_str(x: arb, digits: int = 50) -> str:
    # x is exact in this driver, but str(radius=False) is only presentation.
    return x.str(digits, radius=False)


# ---------------------------------------------------------------------------
# First-order AD formulas for all seven 2D pieces
# ---------------------------------------------------------------------------


def ell2_d(z: D2) -> D2:
    return (THREE * C.qfun_d(z)).sqrt_pos().acos_strict()


def nu2_d(z: D2) -> D2:
    return C.qfun_d(z).sqrt_pos().asin_strict()


def r2_d(z: D2) -> D2:
    return C.r2_bound_d(z)


def ell3_d(z: D2) -> D2:
    # K.ell3(z) = acos(2*sqrt(q(z))) = acos(sqrt(4q(z))).
    return (FOUR * C.qfun_d(z)).sqrt_pos().acos_strict()


def r3_d(z: D2) -> D2:
    x = (FIVE - FOUR * z * z) / (TEN * (ONE - z * z))
    return x.sqrt_pos().acos_strict()


def zero_d(_: D2) -> D2:
    return D2(ZERO, ZERO, ZERO)


def halfpi_d(_: D2) -> D2:
    return D2(PI / TWO, ZERO, ZERO)


def theta_map_d(
    t: D2,
    z: D2,
    lo: Callable[[D2], D2],
    hi: Callable[[D2], D2],
) -> tuple[D2, D2]:
    L = lo(z)
    H = hi(z)
    s, ds = C.sinsq01_d(t)
    return L + (H - L) * s, (H - L) * ds


def G2_d(z: D2, theta: D2, mode: str) -> D2:
    one_minus_z2 = ONE - z * z
    root = one_minus_z2.sqrt_pos()
    ct = theta.cos()
    st = theta.sin()
    y = root * st
    m = SQRT_6_OVER_5 * z
    r2 = (THREE / FIVE) * z * z + one_minus_z2 * ct * ct
    r = r2.sqrt_pos()

    if mode == "middle":
        a3 = C.A3_middle_d(m / (THREE * y))
    elif mode == "full":
        a3 = D2(TWO * PI, ZERO, ZERO)
    else:
        raise ValueError(mode)

    delta = FOUR * (
        TWO - SQRT2 * (THREE * r2 + m * m) / (FIVE * r * r * r)
    )
    return (
        (TEN / SIXTY_FOUR)
        * y
        * TWO
        * a3
        * (TWO - SQRT2 * r)
        * delta
    )


def G3_d(z: D2, theta: D2) -> D2:
    one_minus_z2 = ONE - z * z
    root = one_minus_z2.sqrt_pos()
    ct = theta.cos()
    x = root * ct
    m = SQRT_6_OVER_5 * z
    r2 = (TWO / FIVE) * z * z + one_minus_z2 * ct * ct
    r = r2.sqrt_pos()

    a3 = C.A3_middle_d(m / (THREE * x))
    delta = TWO * (TWO - SQRT2 / r) * (
        TWO - SQRT2 * (TWO * r2 + m * m) / (FIVE * r * r * r)
    )
    return (
        (TEN / SIXTY_FOUR)
        * x
        * a3
        * TWO
        * (TWO - SQRT2 * r)
        * delta
    )


def transformed_i2_d(
    w: D2,
    t: D2,
    za: arb,
    zb: arb,
    lo: Callable[[D2], D2],
    hi: Callable[[D2], D2],
    mode: str,
) -> D2:
    z, jz = C.map_interval_d(w, za, zb)
    theta, jt = theta_map_d(t, z, lo, hi)
    return G2_d(z, theta, mode) * jz * jt


def transformed_i3_d(
    w: D2,
    t: D2,
    za: arb,
    zb: arb,
    lo: Callable[[D2], D2],
    hi: Callable[[D2], D2],
) -> D2:
    z, jz = C.map_interval_d(w, za, zb)
    theta, jt = theta_map_d(t, z, lo, hi)
    return G3_d(z, theta) * jz * jt


@dataclass(frozen=True)
class Piece2D:
    name: str
    component: str
    d_eval: Callable[[D2, D2], D2]
    natural_eval: Callable[[arb, arb], arb]


PIECES_2D = [
    Piece2D(
        "I2_01_mid_zA_zB_ell2_r2", "I2",
        lambda w, t: transformed_i2_d(w, t, K.zA, K.zB, ell2_d, r2_d, "middle"),
        lambda w, t: K.I2_transformed(w, t, K.zA, K.zB, K.ell2, K.r2_bound, "middle"),
    ),
    Piece2D(
        "I2_02_full_zB_zD_ell2_nu", "I2",
        lambda w, t: transformed_i2_d(w, t, K.zB, K.zD, ell2_d, nu2_d, "full"),
        lambda w, t: K.I2_transformed(w, t, K.zB, K.zD, K.ell2, K.nu2, "full"),
    ),
    Piece2D(
        "I2_03_full_zD_zC_0_nu", "I2",
        lambda w, t: transformed_i2_d(w, t, K.zD, K.zC, zero_d, nu2_d, "full"),
        lambda w, t: K.I2_transformed(w, t, K.zD, K.zC, K.zero_angle, K.nu2, "full"),
    ),
    Piece2D(
        "I2_04_mid_zB_zC_nu_r2", "I2",
        lambda w, t: transformed_i2_d(w, t, K.zB, K.zC, nu2_d, r2_d, "middle"),
        lambda w, t: K.I2_transformed(w, t, K.zB, K.zC, K.nu2, K.r2_bound, "middle"),
    ),
    Piece2D(
        "I2_05_full_zC_1_0_pi2", "I2",
        lambda w, t: transformed_i2_d(w, t, K.zC, ONE, zero_d, halfpi_d, "full"),
        lambda w, t: K.I2_transformed(w, t, K.zC, ONE, K.zero_angle, K.halfpi_angle, "full"),
    ),
    Piece2D(
        "I3_01_mid_zA_zB_ell3_r3", "I3",
        lambda w, t: transformed_i3_d(w, t, K.zA, K.zB, ell3_d, r3_d),
        lambda w, t: K.I3_transformed(w, t, K.zA, K.zB, K.ell3, K.r3_bound),
    ),
    Piece2D(
        "I3_02_mid_zB_zC_0_r3", "I3",
        lambda w, t: transformed_i3_d(w, t, K.zB, K.zC, zero_d, r3_d),
        lambda w, t: K.I3_transformed(w, t, K.zB, K.zC, K.zero_angle, K.r3_bound),
    ),
]

PIECES_1D = [p for p in K.PIECES if p.dimension == 1]


def interval_radius(x: arb) -> arb:
    return (x.upper() - x.lower()) / TWO


# ---------------------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------------------


def fresh_state(meta: dict) -> dict:
    return {
        "format_version": 1,
        "meta": meta,
        "phase": "1d",
        "piece_index": 0,
        "next_index": 0,
        "pieces": {},
        "started_unix": time.time(),
        "updated_unix": time.time(),
    }


def make_meta(args, script_path: Path, kernel_path: Path, centered_path: Path) -> dict:
    return {
        "n1": args.n1,
        "n2": args.n2,
        "dps": args.dps,
        "boundary_layers": args.boundary_layers,
        "target": args.target,
        "script_sha256": sha256_file(script_path),
        "kernel_sha256": sha256_file(kernel_path),
        "centered_kernel_sha256": sha256_file(centered_path),
        "python_flint": getattr(flint, "__version__", "unknown"),
    }


def verify_resume_meta(saved: dict, current: dict) -> None:
    keys = [
        "n1", "n2", "dps", "boundary_layers", "target",
        "script_sha256", "kernel_sha256", "centered_kernel_sha256",
    ]
    bad = [(k, saved.get(k), current.get(k)) for k in keys if saved.get(k) != current.get(k)]
    if bad:
        msg = "\n".join(f"  {k}: checkpoint={a!r}, current={b!r}" for k, a, b in bad)
        raise SystemExit(
            "Refusing to resume because parameters/source hashes changed:\n" + msg +
            "\nUse a new --outdir or delete the checkpoint to start fresh."
        )


def load_acc(record: dict | None) -> tuple[arb, arb, int, int, int]:
    if not record:
        return ZERO, ZERO, 0, 0, 0
    return (
        exact_arb_unpack(record["lower_exact"]),
        exact_arb_unpack(record["upper_exact"]),
        int(record.get("centered_boxes", 0)),
        int(record.get("natural_boxes", 0)),
        int(record.get("ad_fallbacks", 0)),
    )


def store_acc(state: dict, piece_name: str, lo: arb, hi: arb,
              centered: int = 0, natural: int = 0, fallbacks: int = 0,
              complete: bool = False) -> None:
    state["pieces"][piece_name] = {
        "lower_exact": exact_arb_pack(lo),
        "upper_exact": exact_arb_pack(hi),
        "centered_boxes": centered,
        "natural_boxes": natural,
        "ad_fallbacks": fallbacks,
        "complete": complete,
    }
    state["updated_unix"] = time.time()


# ---------------------------------------------------------------------------
# Rigorous integration routines with exact directed summation
# ---------------------------------------------------------------------------


def run_1d_piece(piece, n: int, start_i: int, lo0: arb, hi0: arb,
                 checkpoint_every: int, report_every: int,
                 on_checkpoint: Callable[[int, arb, arb], None]):
    cell_len = ONE / arb(n)
    lo_total, hi_total = lo0, hi0
    t0 = time.time()

    for i in range(start_i, n):
        xb = K.dyadic_cell(i, n)
        val = piece.evaluator(xb) * cell_len
        lo_total = directed_add_lower(lo_total, val.lower())
        hi_total = directed_add_upper(hi_total, val.upper())

        done = i + 1
        if checkpoint_every and (done % checkpoint_every == 0 or done == n):
            on_checkpoint(done, lo_total, hi_total)
        if report_every and (done % report_every == 0 or done == n):
            print(
                f"    cells {done:9d}/{n}; "
                f"lo={endpoint_str(lo_total, 18)} hi={endpoint_str(hi_total, 18)}; "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
    return lo_total, hi_total


def run_2d_piece(piece: Piece2D, n: int, start_row: int,
                 lo0: arb, hi0: arb, centered0: int, natural0: int, fallbacks0: int,
                 boundary_layers: int, checkpoint_every_rows: int, report_every_rows: int,
                 on_checkpoint: Callable[[int, arb, arb, int, int, int], None]):
    delta = ONE / arb(n)
    half = delta / TWO
    area = delta * delta

    total_lo, total_hi = lo0, hi0
    centered_boxes, natural_boxes, fallbacks = centered0, natural0, fallbacks0
    t0 = time.time()

    for i in range(start_row, n):
        wb = K.dyadic_cell(i, n)
        wc = arb(f"{2*i+1}/{2*n}")
        row_lo, row_hi = ZERO, ZERO
        row_centered = row_natural = row_fallbacks = 0

        for j in range(n):
            tb = K.dyadic_cell(j, n)
            tc = arb(f"{2*j+1}/{2*n}")

            force_natural = (
                i < boundary_layers or i >= n - boundary_layers
                or j < boundary_layers or j >= n - boundary_layers
            )

            if not force_natural:
                try:
                    dval = piece.d_eval(D2.w(wb), D2.t(tb))
                    rw = interval_radius(dval.dw)
                    rt = interval_radius(dval.dt)

                    fc = piece.natural_eval(wc, tc)
                    center_contrib = area * fc
                    # error = Delta^3/4 * (rw+rt)
                    err = area * half * (rw + rt) / TWO
                    e = err.upper()
                    lo = center_contrib.lower() - e
                    hi = center_contrib.upper() + e
                    row_centered += 1
                except (DerivativeDomainError, ZeroDivisionError, ValueError):
                    force_natural = True
                    row_fallbacks += 1

            if force_natural:
                nv = piece.natural_eval(wb, tb) * area
                lo = nv.lower()
                hi = nv.upper()
                row_natural += 1

            row_lo = directed_add_lower(row_lo, lo)
            row_hi = directed_add_upper(row_hi, hi)

        total_lo = directed_add_lower(total_lo, row_lo)
        total_hi = directed_add_upper(total_hi, row_hi)
        centered_boxes += row_centered
        natural_boxes += row_natural
        fallbacks += row_fallbacks

        done = i + 1
        if checkpoint_every_rows and (done % checkpoint_every_rows == 0 or done == n):
            on_checkpoint(done, total_lo, total_hi, centered_boxes, natural_boxes, fallbacks)
        if report_every_rows and (done % report_every_rows == 0 or done == n):
            print(
                f"    rows {done:5d}/{n}; centered={centered_boxes}; "
                f"natural={natural_boxes}; fallbacks={fallbacks}; "
                f"lo={endpoint_str(total_lo, 16)} hi={endpoint_str(total_hi, 16)}; "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    return total_lo, total_hi, centered_boxes, natural_boxes, fallbacks


# ---------------------------------------------------------------------------
# Final aggregation
# ---------------------------------------------------------------------------


def piece_endpoints(state: dict, name: str) -> tuple[arb, arb]:
    rec = state["pieces"][name]
    if not rec.get("complete"):
        raise ValueError(f"piece not complete: {name}")
    return exact_arb_unpack(rec["lower_exact"]), exact_arb_unpack(rec["upper_exact"])


def aggregate(state: dict) -> tuple[dict, arb, arb]:
    comp = {"I1": [ZERO, ZERO], "I2": [ZERO, ZERO], "I3": [ZERO, ZERO]}
    component_of = {p.name: p.component for p in K.PIECES}
    for p in K.PIECES:
        lo, hi = piece_endpoints(state, p.name)
        c = component_of[p.name]
        comp[c][0] = directed_add_lower(comp[c][0], lo)
        comp[c][1] = directed_add_upper(comp[c][1], hi)

    total_lo, total_hi = ZERO, ZERO
    for c in ("I1", "I2", "I3"):
        total_lo = directed_add_lower(total_lo, comp[c][0])
        total_hi = directed_add_upper(total_hi, comp[c][1])
    return comp, total_lo, total_hi


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Human-checkable uniform centered-form Arb certificate with checkpoint/resume."
    )
    ap.add_argument("--n1", type=int, default=1048576,
                    help="uniform cells for each of the three 1D pieces")
    ap.add_argument("--n2", type=int, default=4096,
                    help="uniform n2 x n2 grid for each of the seven 2D pieces")
    ap.add_argument("--dps", type=int, default=60)
    ap.add_argument("--boundary-layers", type=int, default=1)
    ap.add_argument("--target", default="0.2219")
    ap.add_argument("--checkpoint-every-cells", type=int, default=16384)
    ap.add_argument("--checkpoint-every-rows", type=int, default=16)
    ap.add_argument("--report-every-cells", type=int, default=65536)
    ap.add_argument("--report-every-rows", type=int, default=64)
    ap.add_argument("--digits", type=int, default=50)
    ap.add_argument("--outdir", default="u4_human_checkable_output")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the atomic checkpoint in --outdir")
    args = ap.parse_args()

    if args.n1 <= 0 or args.n2 <= 0:
        raise SystemExit("--n1 and --n2 must be positive")
    if args.dps < 30:
        raise SystemExit("use --dps >= 30")
    if args.boundary_layers < 0 or 2 * args.boundary_layers >= args.n2:
        raise SystemExit("bad --boundary-layers")

    ctx.dps = args.dps
    target = arb(args.target)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "u4_human_checkable_checkpoint.json"
    manifest_path = outdir / "u4_human_checkable_manifest.json"

    script_path = Path(__file__).resolve()
    kernel_path = Path(K.__file__).resolve()
    centered_path = Path(C.__file__).resolve()
    meta = make_meta(args, script_path, kernel_path, centered_path)

    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"--resume requested but checkpoint not found: {checkpoint_path}")
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        verify_resume_meta(state["meta"], meta)
        print(f"RESUME checkpoint: {checkpoint_path}")
        print(f"phase={state['phase']} piece_index={state['piece_index']} next_index={state['next_index']}")
    else:
        if checkpoint_path.exists():
            raise SystemExit(
                f"Checkpoint already exists: {checkpoint_path}\n"
                "Use --resume to continue it, or use a new --outdir to start fresh."
            )
        state = fresh_state(meta)
        atomic_write_json(checkpoint_path, state)

    print("=== U4 human-checkable rigorous Python/Arb verifier ===")
    print(f"python        : {sys.version.split()[0]}")
    print(f"python-flint  : {getattr(flint, '__version__', 'unknown')}")
    print(f"platform      : {platform.platform()}")
    print(f"Arb dps       : {args.dps}")
    print(f"1D grid       : n1={args.n1} for each of 3 pieces")
    print(f"2D grid       : n2={args.n2}; {args.n2*args.n2} boxes for each of 7 pieces")
    print(f"target        : {args.target}")
    print(f"checkpoint    : {checkpoint_path}")
    print(f"script SHA256 : {meta['script_sha256']}")
    print(f"kernel SHA256 : {meta['kernel_sha256']}")
    print(f"centered SHA256: {meta['centered_kernel_sha256']}")
    print(flush=True)

    global_t0 = time.time()

    # ---- 1D pieces --------------------------------------------------------
    if state["phase"] == "1d":
        for idx in range(state["piece_index"], len(PIECES_1D)):
            p = PIECES_1D[idx]
            start = state["next_index"] if idx == state["piece_index"] else 0
            rec = state["pieces"].get(p.name)
            lo0, hi0, _, _, _ = load_acc(rec)
            print(f"[1D {idx+1}/3] {p.name}; resume cell={start}", flush=True)

            def save_1d(next_i, lo, hi, piece=p, piece_idx=idx):
                store_acc(state, piece.name, lo, hi, natural=next_i, complete=(next_i == args.n1))
                state["phase"] = "1d"
                state["piece_index"] = piece_idx
                state["next_index"] = next_i
                if next_i == args.n1:
                    state["piece_index"] = piece_idx + 1
                    state["next_index"] = 0
                atomic_write_json(checkpoint_path, state)

            lo, hi = run_1d_piece(
                p, args.n1, start, lo0, hi0,
                args.checkpoint_every_cells, args.report_every_cells, save_1d,
            )
            store_acc(state, p.name, lo, hi, natural=args.n1, complete=True)
            state["piece_index"] = idx + 1
            state["next_index"] = 0
            atomic_write_json(checkpoint_path, state)
            print(f"    DONE lower={endpoint_str(lo,args.digits)}")
            print(f"         upper={endpoint_str(hi,args.digits)}\n", flush=True)

        state["phase"] = "2d"
        state["piece_index"] = 0
        state["next_index"] = 0
        atomic_write_json(checkpoint_path, state)

    # ---- 2D pieces --------------------------------------------------------
    if state["phase"] == "2d":
        for idx in range(state["piece_index"], len(PIECES_2D)):
            p = PIECES_2D[idx]
            start_row = state["next_index"] if idx == state["piece_index"] else 0
            rec = state["pieces"].get(p.name)
            lo0, hi0, centered0, natural0, fallbacks0 = load_acc(rec)
            print(f"[2D {idx+1}/7] {p.name}; resume row={start_row}", flush=True)

            def save_2d(next_row, lo, hi, centered, natural, fallbacks,
                        piece=p, piece_idx=idx):
                store_acc(
                    state, piece.name, lo, hi, centered, natural, fallbacks,
                    complete=(next_row == args.n2),
                )
                state["phase"] = "2d"
                state["piece_index"] = piece_idx
                state["next_index"] = next_row
                if next_row == args.n2:
                    state["piece_index"] = piece_idx + 1
                    state["next_index"] = 0
                atomic_write_json(checkpoint_path, state)

            lo, hi, centered, natural, fallbacks = run_2d_piece(
                p, args.n2, start_row,
                lo0, hi0, centered0, natural0, fallbacks0,
                args.boundary_layers,
                args.checkpoint_every_rows, args.report_every_rows,
                save_2d,
            )
            store_acc(state, p.name, lo, hi, centered, natural, fallbacks, complete=True)
            state["piece_index"] = idx + 1
            state["next_index"] = 0
            atomic_write_json(checkpoint_path, state)
            print(f"    DONE lower={endpoint_str(lo,args.digits)}")
            print(f"         upper={endpoint_str(hi,args.digits)}")
            print(f"         centered={centered} natural={natural} fallbacks={fallbacks}\n", flush=True)

        state["phase"] = "done"
        state["piece_index"] = len(PIECES_2D)
        state["next_index"] = 0
        atomic_write_json(checkpoint_path, state)

    # ---- Final directed aggregation -------------------------------------
    if state["phase"] != "done":
        raise SystemExit("Internal error: run ended before phase=done")

    comp, total_lo, total_hi = aggregate(state)
    proved = total_hi < target.lower()

    print("=== component enclosures (directed endpoints) ===")
    for c in ("I1", "I2", "I3"):
        print(f"{c}: [{endpoint_str(comp[c][0],args.digits)}, {endpoint_str(comp[c][1],args.digits)}]")
    print("\n=== final enclosure ===")
    print(f"lower  = {endpoint_str(total_lo,args.digits)}")
    print(f"upper  = {endpoint_str(total_hi,args.digits)}")
    print(f"target = {args.target}")
    print(f"STRICT UPPER TARGET PROVED: {'YES' if proved else 'NO'}")

    manifest_obj = {
        "certificate": "U4 human-checkable uniform-grid Python/Arb certificate",
        "proof_kernel": {
            "1d": "natural interval range enclosure on each uniform cell",
            "2d": "same centered first-derivative lemma on every 2D piece; natural interval fallback",
            "summation": "directed lower/upper endpoint summation",
            "checkpoint": "exact binary endpoint serialization via man_exp; atomic replace",
        },
        "parameters": meta,
        "pieces": state["pieces"],
        "components": {
            c: {
                "lower_exact": exact_arb_pack(comp[c][0]),
                "upper_exact": exact_arb_pack(comp[c][1]),
                "lower": endpoint_str(comp[c][0], args.digits),
                "upper": endpoint_str(comp[c][1], args.digits),
            }
            for c in ("I1", "I2", "I3")
        },
        "total_lower_exact": exact_arb_pack(total_lo),
        "total_upper_exact": exact_arb_pack(total_hi),
        "total_lower": endpoint_str(total_lo, args.digits),
        "total_upper": endpoint_str(total_hi, args.digits),
        "target": args.target,
        "strict_upper_target_proved": proved,
        "elapsed_this_invocation_seconds": time.time() - global_t0,
        "python": sys.version,
        "python_flint": getattr(flint, "__version__", "unknown"),
        "platform": platform.platform(),
    }
    atomic_write_json(manifest_path, manifest_obj)
    print(f"manifest       : {manifest_path}")
    print(f"manifest SHA256: {sha256_file(manifest_path)}")
    print(f"checkpoint     : {checkpoint_path}")
    return 0 if proved else 2


if __name__ == "__main__":
    raise SystemExit(main())
