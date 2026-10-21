#!/usr/bin/env python3
"""Generate a fixed-fan rational inner-polytope certificate."""
import argparse, json, time
from pathlib import Path
from bl4 import TARGET_NUM, TARGET_DEN
from bl4.certificate import SCHEMA, write_certificate
from bl4.fan import build_fan, audit_fan
from bl4.oracles import REGISTRY, PENDING
from bl4.radial_parallel import certified_radii_parallel
from bl4.volume import radial_volume, apply_chart_factor, above_target, decimal_value


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--radius-exp", type=int, default=22)
    ap.add_argument("--max-radius", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="independent exact ray workers; default 1")
    ap.add_argument("--out", type=Path, required=True)
    args=ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.body in PENDING:
        raise SystemExit(f"PENDING_ORACLE {args.body}: {PENDING[args.body]}")
    if args.body not in REGISTRY:
        raise SystemExit("unknown body; ready bodies: " + ", ".join(sorted(REGISTRY)))
    oracle=REGISTRY[args.body]
    if oracle.verify_witness is not None:
        raise SystemExit(f"WITNESS_GENERATOR_REQUIRED {args.body}: use generate_amo_certificate.py")
    started=time.perf_counter()
    vertices,cells,scale=build_fan(args.level)
    fan=audit_fan(args.level,vertices,cells,scale)
    t1=time.perf_counter()
    radius_den=1<<args.radius_exp
    radii=certified_radii_parallel(vertices,scale,radius_den,oracle.name,args.max_radius,args.workers)
    t2=time.perf_counter()
    raw_num,raw_den=radial_volume(vertices,cells,radii,radius_den*scale)
    num,den,radicand=apply_chart_factor(raw_num,raw_den,oracle.chart_factor)
    crossed=above_target(num,den,radicand,TARGET_NUM,TARGET_DEN)
    t3=time.perf_counter()
    cert={
        "schema_version":SCHEMA,"body":oracle.name,"oracle":oracle.name,
        "oracle_description":oracle.description,"fan_type":"uniform","fan_level":args.level,"fan":fan,
        "radius_den":radius_den,"radius_num":radii,
        "raw_coordinate_volume":{"num":str(raw_num),"den":str(raw_den)},
        "chart_factor":oracle.chart_factor,
        "euclidean_volume":{"num":str(num),"den":str(den),"sqrt_radicand":radicand},
        "decimal_display_only":f"{decimal_value(num,den,radicand):.15f}",
        "target":{"num":TARGET_NUM,"den":TARGET_DEN},
        "strict_threshold_crossed":crossed,
        "generation_seconds":{"fan_audit":t1-started,"radii":t2-t1,"volume":t3-t2},
    }
    write_certificate(args.out,cert)
    print(json.dumps({"body":oracle.name,"level":args.level,"rays":len(vertices),
                      "cells":len(cells),"lower_bound":cert["decimal_display_only"],
                      "above_threshold":crossed,"output":str(args.out)},indent=2))


if __name__=="__main__": main()
