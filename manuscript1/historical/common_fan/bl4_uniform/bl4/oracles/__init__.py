"""Proof-ready oracle registry.

Modules for bodies whose fixed-fan predicate is still under derivation are
deliberately not registered.  This prevents a placeholder from producing a
misleading certificate.
"""
from .m4 import ORACLE as M4
from .raised_meissner import ORACLE as RAISED_MEISSNER
from .body5 import ORACLE as BODY5
from .raised_u3 import ORACLE as RAISED_U3
from .amo import ORACLE as AMO

REGISTRY = {o.name: o for o in (M4, RAISED_MEISSNER, BODY5, RAISED_U3, AMO)}

PENDING = {}
