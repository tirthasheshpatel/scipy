# cython: language_level=3
cimport cython
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
import numpy as np
cimport numpy as np
from numpy.random cimport bitgen_t
from numpy.random import PCG64

__all__ = ["randn"]

cdef extern from "unuran.h":
    struct unur_distr
    struct unur_par
    struct unur_gen
    struct unur_urng
    unur_urng * unur_urng_new(double (*sampler)(void *params), void *params)
    # TODO: unur_urng * unur_set_default_urng(unur_urng *urng_new)
    # TODO: unur_urng * unur_set_default_urng_aux(unur_urng *urng_new)
    unur_distr * unur_distr_normal(double *params, int n_params)
    unur_par * unur_tdr_new(unur_distr *distribution)
    unur_gen * unur_init(unur_par *par)
    int unur_set_urng(unur_par *par, unur_urng *urng)
    double unur_sample_cont(unur_gen *rng)
    # TODO: int unur_sample_vec(unur_gen *rng, double *out)
    void unur_distr_free(unur_distr *distribution)
    void unur_urng_free(unur_urng *urng)
    void unur_free(unur_gen *rng)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def randn(Py_ssize_t size):
    """Generate random numbers from a standard normal distribution."""
    cdef bitgen_t *_numpy_urng
    cdef unur_distr *distr
    cdef unur_par *par
    cdef unur_gen *rng
    cdef unur_urng *urng

    # Initialize NumPy's BitGenerator
    _numpy_rng = PCG64()
    _capsule = _numpy_rng.capsule
    cdef const char *_capsule_name = "BitGenerator"
    if not PyCapsule_IsValid(_capsule, _capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    _numpy_urng = <bitgen_t *> PyCapsule_GetPointer(_capsule, _capsule_name)

    # Create a new URNG that UNU.RAN understands
    urng = unur_urng_new(_numpy_urng.next_double, <void *>(_numpy_urng.state))

    if ( urng == NULL ): # error
        raise RuntimeError("Failed to create the URNG!")

    # Create the standard normal distribution and
    # the parameter object.
    distr = unur_distr_normal(NULL, 0)
    par = unur_tdr_new(distr)

    # Set the NumPy's BitGenerator to sample from.
    unur_set_urng(par, urng)

    # Initialize a UNU.RAN random number generator.
    rng = unur_init(par)

    if (rng == NULL): # error
        raise RuntimeError("Failed to initialize the RNG!")

    # Free the distribution as we don't need it anymore.
    unur_distr_free(distr)

    # Start sampling.
    out = np.empty(size, dtype=np.float64)
    cdef Py_ssize_t i
    for i in range(size):
        out[i] = unur_sample_cont(rng)

    # Free the remaining objects.
    unur_free(rng)
    unur_urng_free(urng)

    # Return the samples.
    return out
