#!/usr/bin/env python3
"""Reproduce selected checks; quick is an artifact/arithmetic audit, not a proof replay."""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, localcontext
import argparse, datetime, hashlib, importlib.metadata, json, platform
import subprocess, sys, uuid, shutil
from src.audit_u4_return import audit as audit_u4_return

ROOT = Path(__file__).resolve().parent

def load(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8-sig"))

def man_exp(pair):
    n,e=int(pair[0]),int(pair[1])
    return Fraction(n*2**e) if e>=0 else Fraction(n,2**(-e))

def rat(obj):
    return Fraction(int(obj["numerator"]),int(obj["denominator"]))

def require(test,message):
    if not test:
        raise RuntimeError(message)

def quick():
    manifest=load("manifest.json")
    for f in manifest["files"]:
        p=ROOT/f["path"]
        require(p.is_file(),"Missing file: "+f["path"])
        require(p.stat().st_size==f["bytes"],"Size mismatch: "+f["path"])
        require(hashlib.sha256(p.read_bytes()).hexdigest()==f["sha256"],
                "SHA-256 mismatch: "+f["path"])
    grid=load("certificates/u4_grid/u4_human_checkable_manifest.json")
    for key,name in [("script_sha256","u4_human_checkable_checkpoint_arb.py"),
                     ("kernel_sha256","u4_validated_interval_arb.py"),
                     ("centered_kernel_sha256","u4_i2_04_centered_arb.py")]:
        require(hashlib.sha256((ROOT/"src/u4"/name).read_bytes()).hexdigest()==grid["parameters"][key],
                "U4-G original manifest source identity: "+name)
    compact_original=load("src/raised_comparison/CERTIFICATE_MANIFEST.json")
    for name,record in compact_original["files"].items():
        require(hashlib.sha256((ROOT/"src/raised_comparison"/name).read_bytes()).hexdigest()==record["sha256"],
                "Recovered compact manifest identity: "+name)
    lo=man_exp(grid["total_lower_exact"]);hi=man_exp(grid["total_upper_exact"])
    pieces=grid["pieces"]
    require(len(pieces)==10 and all(v["complete"] for v in pieces.values()),"Grid piece schema")
    pl=sum((man_exp(v["lower_exact"]) for v in pieces.values()),Fraction())
    ph=sum((man_exp(v["upper_exact"]) for v in pieces.values()),Fraction())
    require(lo<=pl<=ph<=hi,"Directed grid total does not enclose the piece sum")
    require(Fraction("0.2218338853720453")<lo<hi<Fraction("0.2218382994641744"),"U4-G displayed enclosure")
    T=Fraction(4437,20000)
    require(hi<T,"U4-G separator")
    S=36415211368949752100591187;D=1048576
    require(2*S*S*T.denominator**2>(192*D**4*T.numerator)**2,"RM-L exact separator")
    frozen=load("src/raised_comparison/CERTIFICATE_MANIFEST.json")
    bounds=frozen["successful_fresh_replay"]["exact_raised_bounds"]
    rmup=rat(bounds["rm_upper"]);u3lo=rat(bounds["u3_lower"]);u3up=rat(bounds["u3_upper"])
    require(rat(bounds["rm_positive_upper"])+rat(bounds["rm_negative_upper"])==rmup,"RM upper aggregation")
    require(rmup<u3lo<u3up,"Raised-body exact separation")
    b5=load("certificates/body5_record.json")
    require(hashlib.sha256((ROOT/"src/body5/body5_independent_validated_certificate.py").read_bytes()).hexdigest()==b5["script_sha256"],
            "Body5 original record source identity")
    # Original strings round-trip binary64 endpoints; use their exact dyadic values.
    b5lo=Fraction.from_float(float(b5["volume"]["lower"]))
    b5hi=Fraction.from_float(float(b5["volume"]["upper"]))
    require(u3up<Fraction("0.222868116845350")<b5lo,"Current U3/Body5 separation")
    require(Fraction("0.222877646477730")<b5lo<b5hi<Fraction("0.222991176426990")<Fraction(".248"),"Body5 outward display and M4 separator")
    with localcontext() as ctx:
        ctx.prec=70
        rmlow=Decimal(S)*Decimal(2).sqrt()/(Decimal(192)*Decimal(D)**4)
        decimal_rm=str(rmlow)
    require(decimal_rm.startswith("0.221869166132387373"),"RM decimal cross-check")
    return {
        "status":"PASS",
        "scope":"Frozen-file identities and stored exact arithmetic; no geometric or integration replay",
        "frozen_files_checked":len(manifest["files"]),
        "u4_grid_exact_total_encloses_piece_sum":True,
        "rm_lower_decimal_diagnostic":decimal_rm,
        "raised_comparisons_exact":True,
        "body5_current_endpoint_separation":True,
        "pending_artifacts":manifest["missing"],
        "u4_s_return":audit_u4_return()
    }

def fresh_directory(mode):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p=ROOT/"runs"/(stamp+"_"+mode+"_"+uuid.uuid4().hex[:8])
    p.mkdir(parents=True,exist_ok=False)
    return p

def replay(mode):
    out=fresh_directory(mode)
    if mode=="u4-resume":
        audit_u4_return()
        shutil.copyfile(ROOT/"certificates/u4_s_20261021/u4_i2_04_adaptive_checkpoint.json",
                        out/"u4_i2_04_adaptive_checkpoint.json")
        command=[sys.executable,"-u",str(ROOT/"src/u4/u4_certify_6dp_adaptive.py"),
                 "--base-n","4096","--refine","4","--dps","50","--outdir",str(out),"--resume"]
    elif mode=="body5":
        command=[sys.executable,"-u",str(ROOT/"src/body5/body5_independent_validated_certificate.py")]
    else:
        command=[sys.executable,"-u",str(ROOT/"src/raised_comparison/certify_comparison.py"),
                 "--out",str(out/"verification_result.json")]
    versions={}
    for pkg in ("numpy","numba","llvmlite","mpmath","python-flint"):
        try:versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:pass
    metadata={"mode":mode,"python":sys.version,"platform":platform.platform(),"packages":versions,
              "command":command,"fresh_output_directory":str(out)}
    (out/"environment.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print("Fresh replay:",out,flush=True)
    with (out/"replay.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding="utf-8",errors="replace",cwd=ROOT)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end="",flush=True)
        code=process.wait()
    require(code==0,f"{mode} checker exited with code {code}; see {out/'replay.log'}")
    logtext=(out/"replay.log").read_text(encoding="utf-8")
    if mode=="u4-resume":
        require("RIGOROUS 6-DECIMAL CERTIFICATE: SUCCESS" in logtext,"Missing U4-S success token")
        original=load("certificates/u4_s_20261021/u4_6dp_adaptive_certificate.json")
        current=json.loads((out/"u4_6dp_adaptive_certificate.json").read_text(encoding="utf-8"))
        def mathematical(v):
            if isinstance(v,dict):
                return {k:mathematical(x) for k,x in v.items() if "elapsed" not in k and "seconds" not in k}
            if isinstance(v,list):return list(map(mathematical,v))
            return v
        for key in ("parameters","fresh_1d","i2_04","imported_other_middle_2d","total",
                    "rounding_6dp","rounding_7dp","script_sha256","source_sha256"):
            require(mathematical(current[key])==mathematical(original[key]),"U4-S replay mismatch: "+key)
        scope="Fresh six 1D integrals and aggregation; supplied completed adaptive checkpoint and three U4-G intervals reused; no fresh 2D integration"
    elif mode=="body5":
        require("BODY5_VALIDATED_LOWER_CERTIFICATE_SUCCESS" in logtext,"Missing Body5 success token")
        scope="Complete Body5 production integration; original external raised-U3 endpoint is not recomputed"
    else:
        r=json.loads((out/"verification_result.json").read_text(encoding="utf-8"))
        require(r["status"]=="PASS" and all(r["comparisons"].values()),"Raised-body comparison failed")
        original=load("src/raised_comparison/CERTIFICATE_MANIFEST.json")["successful_fresh_replay"]
        require(r["exact_raised_bounds"]==original["exact_raised_bounds"],"Exact endpoints differ from historical record; investigate")
        scope="Complete compact raised-body replay; U4 and Body5 endpoints remain external inputs"
    result={"status":"PASS","mode":mode,"scope":scope,"outputs":str(out)}
    (out/"replay_status.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["quick","body5","raised","u4-resume"])
    args=parser.parse_args()
    result=quick() if args.mode=="quick" else replay(args.mode)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    main()
