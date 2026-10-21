# Lower-bound scalar checks

This folder accompanies the second manuscript's 16 October 2026 follow-up revision.
Use Python 3.12 (the tested version) and install the listed dependency:

    python -m pip install -r requirements.txt
    python certify_lower_gap.py

On Windows, `py` may be used instead of `python`. The script needs no input
data or internet access. It writes certify_lower_gap.json beside itself and
prints all_scalar_checks_passed: true when every check passes.

The script checks the rational steps in Theorem 5.2 and the conversion of
Jiang's three-dimensional constant in Corollary 5.4. With rJ=1-sqrt(2/5),

    L_spec = 11*pi^2*rJ/224 + rJ*(729*pi/4480000)/4
    L_proj = 4953*pi*rJ/32000.

At 192-bit precision it certifies:

    0.17818404223635132 < L_spec < 0.17818404223635133
    0.17872203991039462 < L_proj < 0.17872203991039463.

The displayed strict lower endpoints are 0.17818404 and 0.17872203.
Rounding L_proj to 0.17872204 does not justify using that rounded number as
a lower bound. These intervals enclose constants, not the unknown optimum.

The spectral proof is in the manuscript. The projection corollary additionally
depends on Yunjiang Jiang, Theorem 1.1, arXiv:2610.13964v1, which states the
width-one three-dimensional bound 1651/4000. Neither that external proof nor
the manuscript's analytic arguments are verified by this scalar script.

The script uses the tail trace coefficient 5/224, proved degree by degree in
the manuscript, and the simple thresholds a=9/50 and b=27/100. No parameter
optimization or U4 volume computation is performed.

Optional:

    python certify_lower_gap.py --precision 256 --output check_256.json
