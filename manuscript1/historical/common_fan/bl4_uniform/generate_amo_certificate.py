#!/usr/bin/env python3
"""V1: uniform fixed-fan certificate generator for the direct AMO peabody.

Floating SciPy searches are proposal-only.  Every saved ray endpoint is
rechecked by the standard-library witness verifier.  The final certificate can
be checked independently with ``verify_certificate.py`` without NumPy/SciPy.
"""
import argparse, ctypes, hashlib, json, os, time
from pathlib import Path

from bl4 import TARGET_NUM,TARGET_DEN
from bl4.amo_parallel import iter_certifications
from bl4.amo_ray_cache import load_cache, atomic_write_cache, get_record, put_record
from bl4.certificate import SCHEMA, write_certificate
from bl4.fan import build_fan, audit_fan
from bl4.oracles import REGISTRY
from bl4.volume import radial_volume, apply_chart_factor, above_target, decimal_value

CK_SCHEMA="amo_uniform_generator_checkpoint_v1"


def sha256_jsonable(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def atomic_json(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2),encoding="utf-8"); os.replace(tmp,path)


def keep_awake_on_windows():
    if os.name!="nt": return lambda:None
    ES_CONTINUOUS=0x80000000; ES_SYSTEM_REQUIRED=0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)
    return lambda: ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def preflight_checkpoint(path,*,level,scale,radius_den,fan_hash,ray_count):
    """Verify schema/indexing before reading any saved ray data."""
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(obj,dict) and obj.get("schema")==CK_SCHEMA
    assert obj.get("body")=="AMO" and obj.get("indexing")=="0-based fan vertex index"
    assert int(obj.get("fan_level"))==level and int(obj.get("direction_scale"))==scale
    assert int(obj.get("radius_den"))==radius_den and obj.get("fan_sha256")==fan_hash
    assert int(obj.get("ray_count"))==ray_count
    rays=obj.get("rays"); assert isinstance(rays,dict)
    for k,v in rays.items():
        i=int(k); assert 0<=i<ray_count and isinstance(v,dict)
        assert isinstance(v.get("radius_num"),int) and v["radius_num"]>=0
        assert isinstance(v.get("witness"),dict)
    return obj


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--level",type=int,required=True)
    ap.add_argument("--radius-exp",type=int,default=24)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path)
    ap.add_argument("--checkpoint-every",type=int,default=25)
    ap.add_argument("--shared-cache",type=Path,
                    help="optional AMO ray cache shared across V1 levels and V2")
    ap.add_argument("--nearest",type=int,default=18)
    ap.add_argument("--workers",type=int,default=1,
                    help="parallel proposal workers; proof verifier is unchanged")
    args=ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >=1")
    oracle=REGISTRY["AMO"]; radius_den=1<<args.radius_exp
    vertices,cells,scale=build_fan(args.level); fan=audit_fan(args.level,vertices,cells,scale)
    ck_path=args.checkpoint or args.out.with_suffix(".checkpoint.json")
    state={"schema":CK_SCHEMA,"body":"AMO","indexing":"0-based fan vertex index",
           "fan_level":args.level,"direction_scale":scale,"radius_den":radius_den,
           "fan_sha256":fan["fan_sha256"],"ray_count":len(vertices),"rays":{}}
    if ck_path.exists():
        state=preflight_checkpoint(ck_path,level=args.level,scale=scale,radius_den=radius_den,
                                   fan_hash=fan["fan_sha256"],ray_count=len(vertices))
        print(f"RESUME preflight PASS: {len(state['rays'])}/{len(vertices)} rays")
    shared = load_cache(args.shared_cache, radius_den) if args.shared_cache else None
    reused = 0
    if shared is not None:
        point_den = radius_den*scale
        for i,u in enumerate(vertices):
            if str(i) in state["rays"]:
                continue
            rec = get_record(shared,u,scale)
            if rec is None:
                continue
            r=int(rec["radius_num"]); w=rec["witness"]
            if r < 0 or not oracle.verify_witness(tuple(r*x for x in u),point_den,w):
                raise AssertionError("shared AMO cache record failed exact endpoint re-verification")
            state["rays"][str(i)]={"radius_num":r,"witness":w,
                                   "diagnostic":dict(rec.get("diagnostic",{}),shared_cache_reuse=True)}
            reused += 1
        print(f"SHARED-CACHE preflight/reverify PASS: reused {reused} rays; records={len(shared['records'])}")
        if reused:
            atomic_json(ck_path,state)
    release=keep_awake_on_windows(); t0=time.perf_counter()
    try:
        done0=len(state["rays"])
        missing=[(i,vertices[i],scale,radius_den,args.nearest)
                 for i in range(len(vertices)) if str(i) not in state["rays"]]
        print(f"AMO proposal workers={args.workers}; missing rays={len(missing)}",flush=True)
        for i,r,w,diag,ray_seconds in iter_certifications(missing,args.workers):
            state["rays"][str(i)]={"radius_num":r,"witness":w,"diagnostic":diag}
            if shared is not None:
                put_record(shared,vertices[i],scale,r,w,diag)
            done=len(state["rays"]); elapsed=time.perf_counter()-t0
            rate=max(1e-12,(done-done0)/elapsed); eta=(len(vertices)-done)/rate
            print(f"ray {done}/{len(vertices)} idx={i} r={r} patch={diag['patch']} "
                  f"res={diag['direction_residual']:.2e} worker={ray_seconds:.2f}s "
                  f"ETA={eta/60:.1f}min",flush=True)
            if done%max(1,args.checkpoint_every)==0:
                atomic_json(ck_path,state)
                if shared is not None:
                    atomic_write_cache(args.shared_cache,shared)
        atomic_json(ck_path,state)
        if shared is not None:
            atomic_write_cache(args.shared_cache,shared)
        radii=[state["rays"][str(i)]["radius_num"] for i in range(len(vertices))]
        witnesses=[state["rays"][str(i)]["witness"] for i in range(len(vertices))]
        raw_num,raw_den=radial_volume(vertices,cells,radii,radius_den*scale)
        num,den,radicand=apply_chart_factor(raw_num,raw_den,oracle.chart_factor)
        crossed=above_target(num,den,radicand,TARGET_NUM,TARGET_DEN)
        cert={"schema_version":SCHEMA,"body":"AMO","oracle":"AMO",
              "oracle_description":oracle.description,"fan_type":"uniform","fan_level":args.level,"fan":fan,
              "radius_den":radius_den,"radius_num":radii,"endpoint_witnesses":witnesses,
              "raw_coordinate_volume":{"num":str(raw_num),"den":str(raw_den)},
              "chart_factor":oracle.chart_factor,
              "euclidean_volume":{"num":str(num),"den":str(den),"sqrt_radicand":radicand},
              "decimal_display_only":f"{decimal_value(num,den,radicand):.15f}",
              "target":{"num":TARGET_NUM,"den":TARGET_DEN},"strict_threshold_crossed":crossed,
              "generation_seconds":time.perf_counter()-t0,
              "proposal_workers":args.workers,
              "shared_cache_reused_rays":reused,
              "shared_cache_sha256":hashlib.sha256(Path(args.shared_cache).read_bytes()).hexdigest() if args.shared_cache else None,
              "generator_evidence_boundary":"SciPy search proposes rational boundary parameters; proof is endpoint_witnesses + exact verifier",
              "checkpoint_sha256":hashlib.sha256(ck_path.read_bytes()).hexdigest()}
        write_certificate(args.out,cert)
        print(json.dumps({"body":"AMO","level":args.level,"rays":len(vertices),"cells":len(cells),
                          "lower_bound":cert["decimal_display_only"],"above_threshold":crossed,
                          "output":str(args.out),"checkpoint":str(ck_path)},indent=2))
    finally:
        release()

if __name__=="__main__": main()
