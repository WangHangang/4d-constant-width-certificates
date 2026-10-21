#!/usr/bin/env python3
"""Cheap supervisor-facing level-4 probe for the five comparison bodies.

Purpose: obtain one common-resolution data point for each body before committing
Andriy's workstation to a multi-day production level.  Every generated file is
independently verified.  The output is diagnostic only; extrapolation from the
probe is never a proof.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREDICATE_BODIES = ("M4", "RaisedMeissnerFace", "Body5", "RaisedU3")


def qcmd(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run(cmd, log, dry_run=False):
    log = ROOT / log
    log.parent.mkdir(parents=True, exist_ok=True)
    print("COMMAND", qcmd(cmd), flush=True)
    if dry_run:
        return 0
    started = time.perf_counter()
    with log.open("a", encoding="utf-8", errors="replace") as f:
        f.write(f"COMMAND {qcmd(cmd)}\n")
        f.flush()
        proc = subprocess.Popen([str(x) for x in cmd], cwd=ROOT,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line); f.flush()
        rc = proc.wait()
        f.write(f"EXIT {rc} elapsed={time.perf_counter()-started:.3f}s\n")
        return rc


def independently_verify(cert: Path, dry_run=False):
    rc = run([sys.executable, "verify_certificate.py", cert],
             Path("logs") / f"{cert.stem}_probe_verify.log", dry_run)
    if rc:
        raise SystemExit(rc)
    if dry_run:
        return None
    obj = json.loads(cert.read_text(encoding="utf-8"))
    required = {"body", "decimal_display_only", "strict_threshold_crossed", "fan"}
    if not isinstance(obj, dict) or not required.issubset(obj):
        raise AssertionError(f"verified certificate has unexpected summary schema: {cert}")
    return {
        "body": obj["body"],
        "lower_bound": obj["decimal_display_only"],
        "above_threshold": obj["strict_threshold_crossed"],
        "rays": obj["fan"]["ray_count"],
        "cells": obj["fan"]["cell_count"],
        "certificate": str(cert.relative_to(ROOT)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--predicate-radius-exp", type=int, default=22)
    ap.add_argument("--amo-radius-exp", type=int, default=24)
    ap.add_argument("--nearest", type=int, default=18)
    ap.add_argument("--shared-cache", type=Path,
                    default=Path("cache/AMO_shared_ray_cache_r24.json"))
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    for d in ("certificates", "logs", "cache"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight:
        rc = run([sys.executable, "preflight_handoff.py"],
                 Path("logs/preflight_level4_probe.log"), args.dry_run)
        if rc:
            return rc

    summary = []
    for body in PREDICATE_BODIES:
        cert = ROOT / "certificates" / f"{body}_k4.json"
        if cert.exists() and not args.dry_run:
            print(f"EXISTING {cert.name}: verify, do not overwrite", flush=True)
        else:
            rc = run([
                sys.executable, "-u", "generate_certificate.py",
                "--body", body, "--level", "4",
                "--radius-exp", str(args.predicate_radius_exp),
                "--workers", str(args.workers),
                "--out", cert,
            ], Path("logs") / f"{body}_k4_probe_generate.log", args.dry_run)
            if rc:
                return rc
        item = independently_verify(cert, args.dry_run)
        if item is not None:
            summary.append(item)

    amo = ROOT / "certificates" / "AMO_k4.json"
    checkpoint = ROOT / "certificates" / "AMO_k4.checkpoint.json"
    if amo.exists() and not args.dry_run:
        print(f"EXISTING {amo.name}: verify, do not overwrite", flush=True)
    else:
        rc = run([
            sys.executable, "-u", "generate_amo_certificate.py",
            "--level", "4", "--radius-exp", str(args.amo_radius_exp),
            "--workers", str(args.workers), "--nearest", str(args.nearest),
            "--shared-cache", args.shared_cache,
            "--out", amo, "--checkpoint", checkpoint,
        ], Path("logs/AMO_k4_probe_generate.log"), args.dry_run)
        if rc:
            return rc
    item = independently_verify(amo, args.dry_run)
    if item is not None:
        summary.append(item)

    if args.dry_run:
        print("DRY RUN COMPLETE: no computation started")
        return 0

    out = ROOT / "logs" / "level4_probe_summary.json"
    payload = {
        "schema": "bl4_level4_probe_summary_v1",
        "warning": "diagnostic only; extrapolation is not proof",
        "fan_level": 4,
        "results": summary,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"WROTE {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
