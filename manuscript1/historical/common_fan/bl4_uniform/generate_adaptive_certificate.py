#!/usr/bin/env python3
"""Generate a rigorous adaptively refined radial certificate.

The refinement score is only a speed heuristic.  Correctness depends solely on
the final exact point checks and the independently audited leaf fan.
"""
import argparse, json, math, time
from pathlib import Path
from bl4 import TARGET_NUM,TARGET_DEN
from bl4.certificate import SCHEMA,write_certificate
from bl4.fan import build_fan,refine_selected,audit_adaptive_fan,determinant4
from bl4.oracles import REGISTRY,PENDING
from bl4.radial_parallel import certified_radii_parallel
from bl4.volume import radial_volume,apply_chart_factor,above_target,decimal_value


def refinement_score(cell,vertices,radii,scale):
    us=[vertices[i] for i in cell]; rs=[radii[i] for i in cell]
    det=abs(determinant4(us)); high=max(rs)**4; low=min(rs)**4
    spread=0
    for i in range(4):
        for j in range(i):
            spread+=sum((us[i][d]-us[j][d])**2 for d in range(4))
    return det*(spread*high+scale*scale*(high-low))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--body",required=True)
    ap.add_argument("--base-level",type=int,default=5)
    ap.add_argument("--max-rounds",type=int,default=3)
    ap.add_argument("--fraction",type=float,default=.15)
    ap.add_argument("--radius-exp",type=int,default=22)
    ap.add_argument("--max-radius",type=float,default=4.0)
    ap.add_argument("--workers",type=int,default=1,
                    help="independent exact ray workers; default 1")
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.workers < 1: raise SystemExit("--workers must be >= 1")
    if args.body in PENDING: raise SystemExit(f"PENDING_ORACLE {args.body}: {PENDING[args.body]}")
    if args.body not in REGISTRY: raise SystemExit("unknown body")
    if not (0<args.fraction<=1) or args.max_rounds<0: raise SystemExit("invalid refinement controls")
    oracle=REGISTRY[args.body]; radius_den=1<<args.radius_exp
    vertices,cells,scale=build_fan(args.base_level)
    schedule=[]; cache={}; started=time.perf_counter(); history=[]
    radii=None
    for round_index in range(args.max_rounds+1):
        radii=certified_radii_parallel(vertices,scale,radius_den,oracle.name,args.max_radius,args.workers,cache)
        raw_num,raw_den=radial_volume(vertices,cells,radii,radius_den*scale)
        num,den,radicand=apply_chart_factor(raw_num,raw_den,oracle.chart_factor)
        crossed=above_target(num,den,radicand,TARGET_NUM,TARGET_DEN)
        history.append({"round":round_index,"rays":len(vertices),"cells":len(cells),
                        "lower_bound":f"{decimal_value(num,den,radicand):.15f}",
                        "above_threshold":crossed})
        print(json.dumps(history[-1]))
        if crossed or round_index==args.max_rounds: break
        count=max(1,int(math.ceil(args.fraction*len(cells))))
        ranked=sorted(range(len(cells)),key=lambda i:refinement_score(cells[i],vertices,radii,scale),reverse=True)
        selected=sorted(ranked[:count]); schedule.append(selected)
        vertices,cells,scale=refine_selected(vertices,cells,scale,selected)
    fan=audit_adaptive_fan(args.base_level,schedule,vertices,cells,scale)
    cert={"schema_version":SCHEMA,"body":oracle.name,"oracle":oracle.name,
          "oracle_description":oracle.description,"fan_type":"adaptive",
          "base_level":args.base_level,"adaptive_schedule":schedule,"fan":fan,
          "radius_den":radius_den,"radius_num":radii,
          "raw_coordinate_volume":{"num":str(raw_num),"den":str(raw_den)},
          "chart_factor":oracle.chart_factor,
          "euclidean_volume":{"num":str(num),"den":str(den),"sqrt_radicand":radicand},
          "decimal_display_only":history[-1]["lower_bound"],
          "target":{"num":TARGET_NUM,"den":TARGET_DEN},
          "strict_threshold_crossed":crossed,"adaptive_history":history,
          "generation_seconds":time.perf_counter()-started,
          "refinement_rule":"top fraction by determinant-weighted angular/radial spread; heuristic only"}
    write_certificate(args.out,cert)
    print(json.dumps({"output":str(args.out),"history":history},indent=2))


if __name__=="__main__": main()

