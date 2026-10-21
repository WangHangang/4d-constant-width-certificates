"""Parallel proposal utility for AMO certificate generators.

This module belongs to the *generator* side only.  It may use multiprocessing
and SciPy through :mod:`bl4.amo_generate`.  Every returned endpoint has already
passed the exact standard-library AMO witness verifier inside ``certify_ray``;
the final independent certificate verifier repeats that check.

Tasks have the form
    (token, direction, direction_scale, radius_den, nearest)
where ``token`` is any pickleable caller identifier (uniform ray index or
adaptive primitive-ray cache key).

The process pool is deliberately bounded: at most ``2*workers`` tasks are
in flight.  This keeps memory predictable on the intended 32-GB workstation.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from .amo_generate import ProposalLibrary, certify_ray

_WORKER_LIBRARY = None


def _limit_numeric_threads():
    """Avoid BLAS/OpenMP oversubscription when many Python workers are used."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, "1")


def _worker_init():
    global _WORKER_LIBRARY
    _limit_numeric_threads()
    _WORKER_LIBRARY = ProposalLibrary()


def _worker_certify(task):
    global _WORKER_LIBRARY
    if _WORKER_LIBRARY is None:
        _worker_init()
    token, u, scale, radius_den, nearest = task
    t0 = time.perf_counter()
    r, witness, diag = certify_ray(
        tuple(int(x) for x in u), int(scale), int(radius_den),
        _WORKER_LIBRARY, nearest=int(nearest)
    )
    return token, int(r), witness, diag, time.perf_counter() - t0


def iter_certifications(tasks, workers=1):
    """Yield rigorously checked ray proposals as they complete.

    Parallel completion order is irrelevant to the proof: callers serialize
    results by ray index/cache key, and the independent verifier later rebuilds
    the fan and rechecks every witness.  ``workers=1`` is a deterministic
    single-process fallback useful for debugging.
    """
    tasks = list(tasks)
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >=1")
    if not tasks:
        return
    if workers == 1:
        _worker_init()
        for task in tasks:
            yield _worker_certify(task)
        return

    _limit_numeric_threads()
    max_inflight = max(workers, 2 * workers)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        it = iter(tasks)
        pending = {}

        def submit_one():
            try:
                task = next(it)
            except StopIteration:
                return False
            fut = ex.submit(_worker_certify, task)
            pending[fut] = task[0]
            return True

        for _ in range(min(max_inflight, len(tasks))):
            submit_one()

        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                pending.pop(fut, None)
                yield fut.result()
                submit_one()
