"""Articulait cinematic tour worker: renders tour plans to MP4 (tour_render) and segments tour
photos with OneFormer (tour_segment). Entry points: tour_worker.jobs.router.run_job, handler.py."""

# The maths libraries' thread pools sized to the CPUs the container may use, before numpy / OpenCV /
# torch load (PROGRESS.md: planning on the pod ran ~8x slower than on a dev machine - nproc says 128,
# the container's quota is ~14, and numpy's BLAS threads fought over it). A value set in the
# environment wins.
def _cap_threads() -> None:
    import os
    from tour_worker.cpus import usable_cpus
    n = str(usable_cpus())
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, n)


_cap_threads()
