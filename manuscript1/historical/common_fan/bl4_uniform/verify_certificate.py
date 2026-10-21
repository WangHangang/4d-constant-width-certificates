#!/usr/bin/env python3
"""Independent standard-library verifier for a radial certificate.

The verifier (1) rebuilds and audits the fan, (2) checks the radius count,
(3) re-certifies every radial vertex with the named exact oracle, (4)
recomputes the determinant sum, and (5) performs the exact algebraic threshold
comparison.  It imports only Python standard-library modules and ``bl4``.
"""
import argparse, json
from bl4 import TARGET_NUM, TARGET_DEN
from bl4.certificate import read_certificate
from bl4.fan import build_fan, audit_fan, build_adaptive_fan, audit_adaptive_fan
from bl4.oracles import REGISTRY
from bl4.volume import radial_volume, apply_chart_factor, above_target, decimal_value


def verify(path):
    cert=read_certificate(path)
    name=cert["oracle"]
    if name not in REGISTRY or cert["body"] != name:
        raise AssertionError("body/oracle identity mismatch")
    oracle=REGISTRY[name]
    if cert.get("oracle_description") != oracle.description:
        raise AssertionError("oracle description mismatch")
    if cert.get("target") != {"num":TARGET_NUM,"den":TARGET_DEN}:
        raise AssertionError("threshold mismatch")
    if cert.get("fan_type","uniform")=="adaptive":
        base_level=int(cert["base_level"])
        schedule=[[int(i) for i in row] for row in cert["adaptive_schedule"]]
        vertices,cells,scale=build_adaptive_fan(base_level,schedule)
        fan=audit_adaptive_fan(base_level,schedule,vertices,cells,scale)
    else:
        level=int(cert["fan_level"])
        vertices,cells,scale=build_fan(level)
        fan=audit_fan(level,vertices,cells,scale)
    if fan != cert["fan"]:
        raise AssertionError("fan metadata/hash mismatch")
    radii=[int(x) for x in cert["radius_num"]]
    if len(radii) != len(vertices):
        raise AssertionError("radius count mismatch")
    radius_den=int(cert["radius_den"])
    point_den=radius_den*scale
    witnesses=cert.get("endpoint_witnesses")
    if oracle.verify_witness is not None:
        if not isinstance(witnesses,list) or len(witnesses)!=len(vertices):
            raise AssertionError("missing or wrong-length endpoint witness list")
        for u,r,witness in zip(vertices,radii,witnesses):
            point=tuple(r*x for x in u)
            if r < 0:
                raise AssertionError("negative radial vertex")
            if r==0 and witness is None:
                if not oracle.contains(point,point_den):
                    raise AssertionError("uncertified origin endpoint")
            elif not oracle.verify_witness(point,point_den,witness):
                raise AssertionError("uncertified witnessed radial vertex")
    else:
        if witnesses is not None:
            raise AssertionError("unexpected endpoint witnesses for predicate oracle")
        for u,r in zip(vertices,radii):
            if r < 0 or not oracle.contains(tuple(r*x for x in u),point_den):
                raise AssertionError("uncertified radial vertex")
    raw_num,raw_den=radial_volume(vertices,cells,radii,point_den)
    if cert["raw_coordinate_volume"] != {"num":str(raw_num),"den":str(raw_den)}:
        raise AssertionError("raw determinant sum mismatch")
    if cert["chart_factor"] != oracle.chart_factor:
        raise AssertionError("chart factor mismatch")
    num,den,radicand=apply_chart_factor(raw_num,raw_den,oracle.chart_factor)
    expected={"num":str(num),"den":str(den),"sqrt_radicand":radicand}
    if cert["euclidean_volume"] != expected:
        raise AssertionError("Euclidean volume representation mismatch")
    crossed=above_target(num,den,radicand,TARGET_NUM,TARGET_DEN)
    if crossed != bool(cert["strict_threshold_crossed"]):
        raise AssertionError("threshold flag mismatch")
    return {"status":"PASS","body":name,"oracle":oracle.description,
            "rays":len(vertices),"cells":len(cells),
            "lower_bound":f"{decimal_value(num,den,radicand):.15f}",
            "above_threshold":crossed}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("certificate")
    result=verify(ap.parse_args().certificate)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
