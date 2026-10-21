#!/usr/bin/env python3
"""Supervisor-facing AMO production driver (V1 -> V2).

The heavy computation remains in the existing AMO generators.  This driver
adds a reproducible unattended workflow: package preflight, Windows anti-sleep,
streamed logs, timing, checkpoint/resume through the generators, independent
verification after every completed stage, and early stop at the exact
separator.

Evidence boundary: this file is orchestration only.  A stage is accepted only
after ``verify_certificate.py`` independently reconstructs/rechecks the final
certificate.  Generator floating-point diagnostics, worker order, cache hits,
and wall-clock timings are not proof data.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def keep_awake_on_windows():
    if os.name != "nt":
        return lambda: None
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    rc = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if not rc:
        raise OSError("SetThreadExecutionState failed")
    return lambda: ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_logged(cmd, log_path, *, dry_run=False):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"{stamp} COMMAND {quote_cmd(cmd)}"
    print(line, flush=True)
    if dry_run:
        return 0
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(line + "\n")
        log.flush()
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for out in proc.stdout:
            print(out, end="", flush=True)
            log.write(out)
            log.flush()
        rc = proc.wait()
        elapsed = time.perf_counter() - started
        tail = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} EXIT {rc} elapsed={elapsed/3600:.3f}h"
        print(tail, flush=True)
        log.write(tail + "\n")
    return rc


def verified_threshold_status(cert, *, dry_run=False):
    cert = Path(cert)
    if not dry_run and not cert.exists():
        raise FileNotFoundError(cert)
    verify_log = ROOT / "logs" / f"{cert.stem}_verify.log"
    rc = run_logged([sys.executable, "verify_certificate.py", cert], verify_log, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"independent verifier failed for {cert}; see {verify_log}")
    if dry_run:
        return False
    # Only read the threshold flag after the independent verifier has accepted
    # the certificate schema, endpoint witnesses, fan, and exact volume sum.
    import json
    obj = json.loads(cert.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("strict_threshold_crossed"), bool):
        raise AssertionError("verified certificate missing boolean strict_threshold_crossed")
    return obj["strict_threshold_crossed"]


def parse_levels(text):
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v < 0:
            raise argparse.ArgumentTypeError("levels must be nonnegative")
        vals.append(v)
    if not vals:
        raise argparse.ArgumentTypeError("at least one V1 level is required")
    if vals != sorted(set(vals)):
        raise argparse.ArgumentTypeError("V1 levels must be strictly increasing and unique")
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--radius-exp", type=int, default=24)
    ap.add_argument("--shared-cache", type=Path, default=Path("cache/AMO_shared_ray_cache_r24.json"))
    ap.add_argument("--v1-levels", type=parse_levels, default=parse_levels("4,5,6"),
                    help="comma-separated substantive V1 levels; default 4,5,6")
    ap.add_argument("--adaptive-base", type=int, default=5)
    ap.add_argument("--adaptive-fraction", type=float, default=0.20)
    ap.add_argument("--adaptive-rounds", type=int, default=6)
    ap.add_argument("--nearest", type=int, default=18)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.radius_exp < 1:
        raise SystemExit("--radius-exp must be positive")
    if not (0 < args.adaptive_fraction <= 1):
        raise SystemExit("--adaptive-fraction must lie in (0,1]")
    if args.adaptive_rounds < 0 or args.adaptive_base < 0:
        raise SystemExit("adaptive level/round controls must be nonnegative")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    for d in ("certificates", "logs", "cache"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    release = keep_awake_on_windows()
    overall = time.perf_counter()
    try:
        if not args.skip_preflight:
            rc = run_logged([sys.executable, "preflight_handoff.py"], ROOT/"logs"/"preflight_handoff.log",
                            dry_run=args.dry_run)
            if rc != 0:
                raise SystemExit(rc)
        if args.preflight_only:
            print("AMO SERVER PREFLIGHT COMPLETE")
            return 0

        # If a requested V1 certificate already exists, never overwrite it
        # before independently verifying it.  A valid crossing ends the run.
        for level in args.v1_levels:
            cert = ROOT / "certificates" / f"AMO_k{level}.json"
            checkpoint = ROOT / "certificates" / f"AMO_k{level}.checkpoint.json"
            if cert.exists() and not args.dry_run:
                print(f"EXISTING certificate found: {cert.name}; verifying before deciding whether to resume/recompute")
                if verified_threshold_status(cert):
                    print(f"COMPLETE: V1 level {level} already gives the strict separator")
                    return 0
                print(f"Existing V1 level {level} is valid but below threshold; generator may resume/rewrite it from checkpoint/cache")

            gen = [
                sys.executable, "-u", "generate_amo_certificate.py",
                "--level", str(level),
                "--radius-exp", str(args.radius_exp),
                "--workers", str(args.workers),
                "--nearest", str(args.nearest),
                "--shared-cache", args.shared_cache,
                "--out", cert,
                "--checkpoint", checkpoint,
            ]
            rc = run_logged(gen, ROOT/"logs"/f"AMO_k{level}_generate.log", dry_run=args.dry_run)
            if rc != 0:
                raise SystemExit(rc)
            if verified_threshold_status(cert, dry_run=args.dry_run):
                print(f"COMPLETE: V1 level {level} gives the strict separator")
                return 0

        cert = ROOT / "certificates" / "AMO_adaptive.json"
        checkpoint = ROOT / "certificates" / "AMO_adaptive.checkpoint.json"
        if cert.exists() and not args.dry_run:
            print(f"EXISTING certificate found: {cert.name}; verifying before adaptive resume")
            if verified_threshold_status(cert):
                print("COMPLETE: existing V2 adaptive certificate gives the strict separator")
                return 0

        gen = [
            sys.executable, "-u", "generate_amo_adaptive_certificate.py",
            "--base-level", str(args.adaptive_base),
            "--fraction", str(args.adaptive_fraction),
            "--max-rounds", str(args.adaptive_rounds),
            "--radius-exp", str(args.radius_exp),
            "--workers", str(args.workers),
            "--nearest", str(args.nearest),
            "--shared-cache", args.shared_cache,
            "--out", cert,
            "--checkpoint", checkpoint,
        ]
        rc = run_logged(gen, ROOT/"logs"/"AMO_adaptive_generate.log", dry_run=args.dry_run)
        if rc != 0:
            raise SystemExit(rc)
        if verified_threshold_status(cert, dry_run=args.dry_run):
            print("COMPLETE: V2 adaptive certificate gives the strict separator")
            return 0

        print("VALID AMO V1/V2 certificates remain below the separator; retain all checkpoints/cache and review refinement diagnostics.")
        return 2
    finally:
        release()
        print(f"TOTAL DRIVER ELAPSED {(time.perf_counter()-overall)/3600:.3f}h", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
