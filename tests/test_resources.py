"""Resource controls: native thread-pool limits are real, not env-declared."""

from __future__ import annotations


def test_worker_init_limits_native_thread_pools():
    # Review round 3, point 4: setting OMP/OPENBLAS/MKL_NUM_THREADS after
    # NumPy import caps nothing (measured: 12 OpenBLAS threads before and
    # after late env changes; 1 thread when set before import). _worker_init
    # must limit the loaded pools at runtime, and the test introspects the
    # actual pool settings.
    import numpy as np  # noqa: F401 — loads OpenBLAS into this process
    from threadpoolctl import threadpool_info, threadpool_limits

    from seqconsistency.pipeline import _worker_init

    _worker_init(1)
    try:
        pools = [
            i
            for i in threadpool_info()
            if i.get("internal_api") in ("openblas", "mkl", "blis", "blas", "omp")
        ]
        assert pools, "no native math pools loaded in the test process"
        assert all(i["num_threads"] <= 1 for i in pools), pools
    finally:
        threadpool_limits(limits=None).restore_original_limits()


def test_worker_init_limit_survives_garbage_collection():
    # The controller must stay referenced: if _worker_init's limit were bound
    # to a local, GC would undo it. Import-order trick: force a GC after init
    # and re-check.
    import gc

    import numpy as np  # noqa: F401
    from threadpoolctl import threadpool_info, threadpool_limits

    from seqconsistency import pipeline
    from seqconsistency.pipeline import _worker_init

    _worker_init(2)
    gc.collect()
    try:
        pools = [i for i in threadpool_info() if i.get("num_threads") is not None]
        assert all(i["num_threads"] <= 2 for i in pools), pools
        assert pipeline._thread_limit is not None
    finally:
        threadpool_limits(limits=None).restore_original_limits()
