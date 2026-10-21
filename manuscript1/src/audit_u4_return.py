"""Audit returned U4-S records with exact arithmetic; no integration replay."""
from pathlib import Path
from fractions import Fraction as F
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'certificates/u4_s_20261021'

def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def dyadic(pair):
    m, e = int(pair[0]), int(pair[1])
    return F(m * 2**e) if e >= 0 else F(m, 2**(-e))

def audit():
    manifest = load(RETURN/'RETURN_MANIFEST.json')
    for name, expected in manifest['files'].items():
        require(Path(name).name == name, 'Unexpected manifest path')
        require(hashlib.sha256((RETURN/name).read_bytes()).hexdigest() == expected,
                'Returned file identity: '+name)
    cert = load(RETURN/'u4_6dp_adaptive_certificate.json')
    checkpoint = load(RETURN/'u4_i2_04_adaptive_checkpoint.json')
    hashes = {cert['script']:cert['script_sha256'], **cert['source_sha256']}
    for name, expected in hashes.items():
        require(hashlib.sha256((ROOT/'src/u4'/name).read_bytes()).hexdigest() == expected,
                'Frozen source identity: '+name)
    require(checkpoint['schema_version'] == cert['schema_version'] == 1, 'Schema version')
    p=cert['parameters']; meta=checkpoint['meta']; counts=checkpoint['counts']
    for key in ['base_n','refine','dps','boundary_layers_refined']:
        require(p[key] == meta[key], 'Parameter mismatch: '+key)
    require(meta['script_sha256'] == cert['script_sha256'] and meta['source_sha256'] == cert['source_sha256'], 'Checkpoint source identity')
    require(p['base_n']==4096 and p['refine']==4 and p['fine_n']==16384 and p['dps']==50, 'Unexpected production parameters')
    require(checkpoint['complete'] is True and checkpoint['next_row']==p['base_n'], 'Incomplete checkpoint')
    require(all(type(x) is int and x>=0 for x in counts.values()), 'Invalid counters')
    require(counts==cert['i2_04']['counts'], 'Certificate/checkpoint counters differ')
    require(counts['base_centered']+counts['base_refined']==p['base_n']**2, 'Base-cell coverage')
    require(counts['refined_centered']+counts['refined_natural']==counts['base_refined']*p['refine']**2, 'Subcell coverage')
    require(counts['base_derivative_fallbacks']<=counts['base_refined'] and counts['refined_derivative_fallbacks']<=counts['refined_natural'], 'Fallback counters')
    clo,chi=dyadic(checkpoint['lower_exact']),dyadic(checkpoint['upper_exact'])
    require(clo<chi, 'Checkpoint endpoints')
    # Decimal displays are rounded strings, not exact directed endpoints.
    display_slack=F(1,10**95)
    require(abs(F(cert['i2_04']['lower'])-clo)<display_slack and abs(F(cert['i2_04']['upper'])-chi)<display_slack, 'Exact checkpoint/display agreement')
    grid=load(ROOT/'certificates/u4_grid/u4_human_checkable_manifest.json')
    for key, bounds in cert['imported_other_middle_2d'].items():
        matches=[v for k,v in grid['pieces'].items() if k.startswith(key+'_')]
        require(len(matches)==1 and matches[0]['complete'], 'Missing original imported piece: '+key)
        require(F(bounds[0])<=dyadic(matches[0]['lower_exact'])<=dyadic(matches[0]['upper_exact'])<=F(bounds[1]), 'Imported interval is not outward from exact U4-G record: '+key)
    require(set(cert['fresh_1d'])=={'I1_01','I1_02','I1_03','I2_02','I2_03','I2_05'}, 'Six 1D pieces')
    lo=clo;hi=chi
    for piece in cert['fresh_1d'].values():
        a,b=F(piece['lower']),F(piece['upper'])
        require(a<b, '1D endpoints')
        lo+=a-F(1,10**85);hi+=b+F(1,10**85)
    for a,b in cert['imported_other_middle_2d'].values():lo+=F(a);hi+=F(b)
    total_lo,total_hi=F(cert['total']['lower']),F(cert['total']['upper'])
    require(total_lo<lo<hi<total_hi, 'Reported total fails to enclose independently summed pieces')
    outer_lo,outer_hi=F('0.2218357659033942'),F('0.2218363538061581')
    require(outer_lo<total_lo<total_hi<outer_hi, 'Paper outward endpoints')
    require(F('0.2218355')<outer_lo<outer_hi<F('0.2218365'), 'Six-decimal rounding cell')
    require(outer_hi<F(4437,20000), 'Strict comparison separator')
    require(cert['rounding_6dp']['certified'] is True and cert['rounding_6dp']['candidate']=='0.221836', 'Reported rounding status')
    require(cert['rounding_7dp']['certified'] is False, 'Seven decimals must not be claimed')
    log=next(RETURN.glob('run_*.log')).read_text(encoding='utf-8-sig')
    require('resume row     : 0' in log and 'rows  4096/4096;' in log and 'RIGOROUS 6-DECIMAL CERTIFICATE: SUCCESS' in log, 'Incomplete production log')
    return {'status':'PASS','scope':'Returned file/source identities, exact completed-checkpoint coverage counters, imported U4-G endpoint provenance, rational ten-piece aggregation and six-decimal rounding. No fresh 2D integration in this audit.',
            'certificate_sha256':hashlib.sha256((RETURN/'u4_6dp_adaptive_certificate.json').read_bytes()).hexdigest(),
            'returned_files_checked':len(manifest['files']),'source_files_checked':len(hashes),
            'base_cells':p['base_n']**2,'refined_base_cells':counts['base_refined'],
            'refined_subcells':counts['refined_centered']+counts['refined_natural'],
            'paper_enclosure':['0.2218357659033942','0.2218363538061581'],
            'correctly_rounded_6dp':'0.221836','seven_decimal_rounding_certified':False,
            'production_elapsed_seconds':cert['elapsed_this_invocation_seconds'],
            'historical_record_status':'New production run; original b787db8a5fad... JSON remains unrecovered.'}

if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
