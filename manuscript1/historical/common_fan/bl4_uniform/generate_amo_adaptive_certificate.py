#!/usr/bin/env python3
"""V2: adaptive fixed-fan AMO certificate generator.

The refinement score and SciPy searches are heuristics only.  The final proof
is the exact audited adaptive fan, exact AMO endpoint witnesses, and exact
rational determinant sum.  Checkpoint data are schema-checked before reuse.
"""
import argparse, ctypes, hashlib, json, math, os, time
from fractions import Fraction
from pathlib import Path

from bl4 import TARGET_NUM,TARGET_DEN
from bl4.amo_parallel import iter_certifications
from bl4.amo_ray_cache import ray_key, load_cache, atomic_write_cache, get_record, put_record, parse_ray_key
from bl4.certificate import SCHEMA,write_certificate
from bl4.fan import build_fan,refine_selected,audit_adaptive_fan,determinant4
from bl4.oracles import REGISTRY
from bl4.volume import radial_volume,apply_chart_factor,above_target,decimal_value

CK_SCHEMA="amo_adaptive_generator_checkpoint_v1"



def atomic_json(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2),encoding="utf-8"); os.replace(tmp,path)


def keep_awake_on_windows():
    if os.name!="nt": return lambda:None
    ES_CONTINUOUS=0x80000000; ES_SYSTEM_REQUIRED=0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)
    return lambda: ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def refinement_score(cell,vertices,radii,scale):
    us=[vertices[i] for i in cell]; rs=[radii[i] for i in cell]
    det=abs(determinant4(us)); high=max(rs)**4; low=min(rs)**4
    spread=sum(sum((us[i][d]-us[j][d])**2 for d in range(4))
               for i in range(4) for j in range(i))
    return det*(spread*high+scale*scale*(high-low))


def preflight_state(path,base_level,radius_den,fraction,max_rounds):
    o=json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(o,dict) and o.get("schema")==CK_SCHEMA
    assert o.get("body")=="AMO" and o.get("indexing")=="ray cache keyed by primitive integer direction plus scale denominator"
    assert int(o["base_level"])==base_level and int(o["radius_den"])==radius_den
    assert abs(float(o["fraction"])-fraction)<1e-15 and int(o["max_rounds"])==max_rounds
    assert isinstance(o.get("schedule"),list) and isinstance(o.get("cache"),dict)
    for sel in o["schedule"]:
        assert isinstance(sel,list) and all(isinstance(i,int) and i>=0 for i in sel)
    for k,v in o["cache"].items():
        parse_ray_key(k)
        assert isinstance(v,dict) and isinstance(v.get("radius_num"),int) and v["radius_num"]>=0
        assert isinstance(v.get("witness"),dict)
    return o


def rebuild(base_level,schedule):
    vertices,cells,scale=build_fan(base_level)
    for selected in schedule:
        if selected and max(selected)>=len(cells):
            raise AssertionError("checkpoint adaptive selection index out of range")
        vertices,cells,scale=refine_selected(vertices,cells,scale,selected)
    return vertices,cells,scale


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-level",type=int,default=3)
    ap.add_argument("--max-rounds",type=int,default=4)
    ap.add_argument("--fraction",type=float,default=.20)
    ap.add_argument("--radius-exp",type=int,default=24)
    ap.add_argument("--nearest",type=int,default=18)
    ap.add_argument("--workers",type=int,default=1,
                    help="parallel proposal workers; proof verifier is unchanged")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path)
    ap.add_argument("--checkpoint-every",type=int,default=25)
    ap.add_argument("--shared-cache",type=Path,
                    help="optional AMO ray cache shared with V1 and other V2 runs")
    args=ap.parse_args()
    if not (0<args.fraction<=1) or args.max_rounds<0:
        raise SystemExit("invalid adaptive controls")
    if args.workers < 1:
        raise SystemExit("--workers must be >=1")
    radius_den=1<<args.radius_exp; oracle=REGISTRY["AMO"]
    ck=args.checkpoint or args.out.with_suffix(".checkpoint.json")
    state={"schema":CK_SCHEMA,"body":"AMO",
           "indexing":"ray cache keyed by primitive integer direction plus scale denominator",
           "base_level":args.base_level,"radius_den":radius_den,"fraction":args.fraction,
           "max_rounds":args.max_rounds,"schedule":[],"cache":{},"history":[]}
    if ck.exists():
        state=preflight_state(ck,args.base_level,radius_den,args.fraction,args.max_rounds)
        print(f"RESUME preflight PASS: schedule rounds={len(state['schedule'])}, cached rays={len(state['cache'])}")
    vertices,cells,scale=rebuild(args.base_level,state["schedule"])
    shared = load_cache(args.shared_cache, radius_den) if args.shared_cache else None
    shared_reused_total = 0
    hist_saved=list(state.get("history",[]))
    sched_len=len(state["schedule"])
    if len(hist_saved) not in (sched_len,sched_len+1):
        raise AssertionError("checkpoint history/schedule mismatch")
    saved_current_row=None
    # A fully completed checkpoint has one row for the current final fan.
    # Drop it locally and recompute the mathematics from cached exact witnesses.
    # If the recomputation agrees, preserve its historical new-ray count so
    # a no-op resume does not rewrite the execution history.
    if len(hist_saved)==sched_len+1:
        if int(hist_saved[-1].get("round",-1)) != sched_len:
            raise AssertionError("checkpoint final history row has wrong round index")
        saved_current_row=hist_saved[-1]
        hist_saved=hist_saved[:-1]
    print(f"AMO proposal workers={args.workers}; cached rays={len(state['cache'])}",flush=True)
    release=keep_awake_on_windows(); t0=time.perf_counter()
    try:
        # Recompute the current round from cached exact witnesses; this makes resume simple.
        history=hist_saved
        for round_index in range(len(state["schedule"]),args.max_rounds+1):
            round_start=time.perf_counter()
            missing_map={}
            point_den=radius_den*scale
            reused_this_round=0
            for u in vertices:
                key=ray_key(u,scale)
                if key in state["cache"]:
                    continue
                rec=get_record(shared,u,scale) if shared is not None else None
                if rec is not None:
                    r=int(rec["radius_num"]); w=rec["witness"]
                    if r < 0 or not oracle.verify_witness(tuple(r*x for x in u),point_den,w):
                        raise AssertionError("shared AMO cache record failed exact endpoint re-verification")
                    state["cache"][key]={"radius_num":r,"witness":w,
                                         "diagnostic":dict(rec.get("diagnostic",{}),shared_cache_reuse=True)}
                    reused_this_round += 1
                    shared_reused_total += 1
                else:
                    missing_map.setdefault(key,tuple(u))
            tasks=[(key,u,scale,radius_den,args.nearest)
                   for key,u in missing_map.items()]
            new_count=0
            print(f"round={round_index} rays={len(vertices)} new={len(tasks)} reused_shared={reused_this_round} "
                  f"cells={len(cells)} scale={scale}",flush=True)
            for key,r,w,diag,ray_seconds in iter_certifications(tasks,args.workers):
                state["cache"][key]={"radius_num":r,"witness":w,"diagnostic":diag}
                if shared is not None:
                    ku,kden=parse_ray_key(key)
                    put_record(shared,ku,kden,r,w,diag)
                new_count+=1
                if new_count%max(1,args.checkpoint_every)==0:
                    atomic_json(ck,state)
                    if shared is not None:
                        atomic_write_cache(args.shared_cache,shared)
                    elapsed=time.perf_counter()-round_start
                    rate=max(1e-12,new_count/elapsed)
                    eta=(len(tasks)-new_count)/rate
                    print(f"round={round_index} certified new={new_count}/{len(tasks)} "
                          f"last_worker={ray_seconds:.2f}s ETA={eta/60:.1f}min",flush=True)
            if tasks or reused_this_round:
                atomic_json(ck,state)
            if shared is not None:
                atomic_write_cache(args.shared_cache,shared)
            radii=[]; witnesses=[]
            for u in vertices:
                rec=state["cache"].get(ray_key(u,scale))
                if rec is None:
                    raise AssertionError("missing cached AMO ray after proposal stage")
                radii.append(int(rec["radius_num"])); witnesses.append(rec["witness"])
            raw_num,raw_den=radial_volume(vertices,cells,radii,radius_den*scale)
            num,den,rad=apply_chart_factor(raw_num,raw_den,oracle.chart_factor)
            crossed=above_target(num,den,rad,TARGET_NUM,TARGET_DEN)
            row={"round":round_index,"rays":len(vertices),"cells":len(cells),"scale":scale,
                 "new_rays":new_count,"shared_cache_reused_rays":reused_this_round,"lower_bound":f"{decimal_value(num,den,rad):.15f}",
                 "above_threshold":crossed}
            if saved_current_row is not None and round_index==sched_len and not tasks:
                for field in ("round","rays","cells","scale","lower_bound","above_threshold"):
                    if row[field] != saved_current_row.get(field):
                        raise AssertionError("completed checkpoint current-round recomputation mismatch")
                row["new_rays"]=int(saved_current_row.get("new_rays",0))
            history.append(row); print(json.dumps(row),flush=True)
            if crossed or round_index==args.max_rounds: break
            count=max(1,int(math.ceil(args.fraction*len(cells))))
            ranked=sorted(range(len(cells)),key=lambda i:refinement_score(cells[i],vertices,radii,scale),reverse=True)
            selected=sorted(ranked[:count]); state["schedule"].append(selected)
            state["history"]=history; atomic_json(ck,state)
            vertices,cells,scale=refine_selected(vertices,cells,scale,selected)
        fan=audit_adaptive_fan(args.base_level,state["schedule"],vertices,cells,scale)
        cert={"schema_version":SCHEMA,"body":"AMO","oracle":"AMO","oracle_description":oracle.description,
              "fan_type":"adaptive","base_level":args.base_level,"adaptive_schedule":state["schedule"],"fan":fan,
              "radius_den":radius_den,"radius_num":radii,"endpoint_witnesses":witnesses,
              "raw_coordinate_volume":{"num":str(raw_num),"den":str(raw_den)},"chart_factor":oracle.chart_factor,
              "euclidean_volume":{"num":str(num),"den":str(den),"sqrt_radicand":rad},
              "decimal_display_only":history[-1]["lower_bound"],"target":{"num":TARGET_NUM,"den":TARGET_DEN},
              "strict_threshold_crossed":crossed,"adaptive_history":history,
              "generation_seconds":time.perf_counter()-t0,
              "proposal_workers":args.workers,
              "shared_cache_reused_rays":shared_reused_total,
              "shared_cache_sha256":hashlib.sha256(Path(args.shared_cache).read_bytes()).hexdigest() if args.shared_cache else None,
              "refinement_rule":"top fraction by determinant-weighted angular/radial spread; heuristic only",
              "generator_evidence_boundary":"SciPy search/refinement is proposal-only; exact witness verification and fan audit are proof",
              "checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest() if ck.exists() else None}
        write_certificate(args.out,cert)
        state["history"]=history; atomic_json(ck,state)
        print(json.dumps({"output":str(args.out),"history":history},indent=2))
    finally:
        release()

if __name__=="__main__": main()
