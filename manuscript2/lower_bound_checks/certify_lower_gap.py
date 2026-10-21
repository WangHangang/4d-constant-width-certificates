"""Scalar checks for the spectral and projection lower bounds (16 Oct 2026).

Run: python certify_lower_gap.py
Requires python-flint==0.9.0. No input data, internet, or U4 certificate is used.
The mathematical proofs and the cited three-dimensional theorem are separate.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse, hashlib, json, platform
import flint
from flint import arb, ctx

def require(condition, message):
    if not condition:
        raise ArithmeticError(message)

def check(precision):
    ctx.prec=precision
    a,b,L,r=F(9,50),F(27,100),F(36,5),F(1,40)
    d=F(1,4)-(a+b)**2
    C=d/b**2
    k=F(5,224)/C
    require(d==F(19,400),'Pointwise deficit')
    require(C==F(475,729) and k==F(729,21280),'Deficit coefficients')
    require(k<F(1,28),'Comparison of the two deficits')
    require(F(360,7)<L**2 and L*r==a,'Gradient bound and arc radius')
    require(F(35-15,28*(35-3))==F(5,224),'Degree-five equality')
    # For lambda>=35 the coefficient difference is
    # 3*(lambda-35)/(224*(lambda-3)), which is nonnegative.
    require(F(1,28)-F(5,224)==F(3,224),'Coefficient of lambda')
    require(-F(15,28)+F(15,224)==-F(105,224),'Constant coefficient')
    A=lambda q:arb(q.numerator)/q.denominator
    pi=arb.pi()
    require(A(r)<pi/2,'Disjoint antipodal arcs')
    rJ=1-(arb(2)/5).sqrt()
    basic=11*pi*pi*rJ/224
    eps=5*A(b)**2*(4*pi*A(r))/224
    eps_simple=729*pi/4480000
    require(eps.overlaps(eps_simple),'Simplified surface-area constant')
    spectral=basic+rJ*eps_simple/4
    projection=4953*pi*rJ/32000
    # Exact conversion of Jiang's width-one three-dimensional constant.
    require(F(3,8)*F(1651,4000)==F(4953,32000),'Projection normalization')
    spec_endpoints=['0.17818404223635132','0.17818404223635133']
    proj_endpoints=['0.17872203991039462','0.17872203991039463']
    for value,endpoints in [(spectral,spec_endpoints),(projection,proj_endpoints)]:
        require(value>arb(endpoints[0]) and value<arb(endpoints[1]),'Decimal enclosure')
    require(spectral>arb('0.17818404'),'Displayed spectral lower endpoint')
    require(projection>arb('0.17872203'),'Displayed projection lower endpoint')
    require(projection<arb('0.17872204'),'Upward-rounded number is not a lower endpoint')
    previous=basic+rJ*(arb(5)/714)*(arb(153)/500)**2*pi*(arb(7)/288)
    require(spectral>previous and projection>spectral,'Comparison with previous bounds')
    return {
        'precision_bits':precision,'threshold_a':str(a),'threshold_b':str(b),
        'arc_radius':str(r),'rational_gradient_bound':str(L),
        'degree_three_gradient_squared_bound':'360/7',
        'tail_trace_coefficient':'5/224','pointwise_deficit':str(d),
        'deficit_coefficient':str(C),'spectral_multiplier':str(k),
        'surface_area_gap':str(eps_simple),'basic_volume_bound':str(basic),
        'morning_spectral_bound':str(previous),
        'spectral_volume_bound':str(spectral),
        'spectral_constant_strict_enclosure':spec_endpoints,
        'projection_volume_bound':str(projection),
        'projection_constant_strict_enclosure':proj_endpoints,
        'displayed_strict_lower_endpoints':{'spectral':'0.17818404','projection':'0.17872203'},
        'literature_input':{'author':'Yunjiang Jiang','theorem':'Theorem 1.1',
            'url':'https://arxiv.org/html/2610.13964v1','width_one_3d_constant':'1651/4000',
            'external_analytic_theorem_not_verified_by_this_script':True},
        'all_scalar_checks_passed':True,
    }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision',type=int,default=192)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    args=parser.parse_args()
    if args.precision<128:parser.error('--precision must be at least 128')
    out=check(args.precision)
    out.update({'python_version':platform.python_version(),'python_flint_version':flint.__version__,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Exact rational scalar comparisons and Arb enclosures only. Intervals enclose proved bound constants, not the unknown minimum. No global U4 feasibility or external theorem proof is verified here.'})
    args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
