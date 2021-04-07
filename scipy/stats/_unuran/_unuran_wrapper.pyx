# cython: language_level=3

cimport cython
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
import ctypes

import numpy as np
cimport numpy as np
from numpy.random cimport bitgen_t
from numpy.random import PCG64


__all__ = ["randn", "tdr", "dau"]


# Signature of PDF, etc that UNU.RAN expects.
ctypedef double (*cont_func_t)(double, const unur_distr *)


cdef extern from "unuran.h":
    # =======================================================================
    # UNU.RAN Structures
    # =======================================================================

    struct unur_distr  # Distribution Object
    struct unur_par    # Parameter Object
    struct unur_gen    # Generator Object
    struct unur_urng   # URNG Object

    # =======================================================================
    # UNU.RAN Core Functionality
    # =======================================================================

    # URNG API
    # TODO: set NumPy's BitGenerator as default
    # unur_urng * unur_set_default_urng(unur_urng *urng_new)
    # unur_urng * unur_set_default_urng_aux(unur_urng *urng_new)
    unur_urng * unur_urng_new(double (*sampler)(void *state), void *state)
    int unur_set_urng(unur_par *par, unur_urng *urng)

    # Continuous Distributions
    unur_distr * unur_distr_cont_new()
    int unur_distr_cont_set_pdf(unur_distr *distribution,
                                cont_func_t pdf)
    int unur_distr_cont_set_dpdf(unur_distr *distribution,
                                 cont_func_t dpdf)
    # int unur_distr_cont_set_pdfparams(unur_distr *distribution,
    #                                   const double *params, int n_params)
    # int unur_distr_cont_get_pdfparams(const unur_distr *distribution,
    #                                   const double **params)
    int unur_distr_cont_set_domain(unur_distr *distribution,
                                   double left, double right)
    unur_par * unur_tdr_new(unur_distr *distribution)

    # Discrete Distributions
    unur_distr * unur_distr_discr_new()
    int unur_distr_discr_set_pmf(unur_distr *distribution,
                                 double (*pmf)(int k,
                                               const unur_distr *distr))
    # int unur_distr_discr_set_pmfparams(unur_distr *distribution,
    #                                    const double *params, int n_params)
    # int unur_distr_discr_get_pmfparams(const unur_distr *distribution,
    #                                    const double **params)
    int unur_distr_discr_set_domain(unur_distr *distribution,
                                    int left, int right)
    int unur_distr_discr_set_pv(unur_distr *distribution,
                                const double *pv, int n_pv)
    unur_par *unur_dau_new(const unur_distr *distribution)

    # Generator and Sampling
    unur_gen * unur_init(unur_par *par)
    double unur_sample_cont(unur_gen *rng)
    int unur_sample_discr(unur_gen *generator)

    # XXX: Only for testing purposes. Need to remove it later
    unur_distr * unur_distr_normal(double *params, int n_params)

    # Routines to free the allocated distributions and generators
    void unur_distr_free(unur_distr *distribution)
    void unur_urng_free(unur_urng *urng)
    void unur_free(unur_gen *rng)


# ===========================================================================
# Testing if UNU.RAN builds properly.
# ===========================================================================


# XXX: This function is present here only for testing purposes.
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
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
    urng = unur_urng_new(_numpy_urng.next_double,
                         <void *>(_numpy_urng.state))

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
    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(size,
                                                         dtype=np.float64)
    cdef Py_ssize_t i
    for i in range(size):
        out[i] = unur_sample_cont(rng)

    # Free the remaining objects.
    unur_free(rng)
    unur_urng_free(urng)

    # Return the samples.
    return out


# ===========================================================================
# Global Attributes.
# ===========================================================================


# XXX: This is bad. I am trying to use ``ctypes`` to convert the
#      Python function to Cython function. While I figure it out,
#      I have used global attributes. Is it OK to do so? I think
#      it won't be thread safe.
_global_params = None

# For continuous distributions
_global_pdf = None
_global_dpdf = None

# For discrete distributions
_global_pmf = None


# ===========================================================================
# Wrappers for python functions.
# ===========================================================================


# Wrap PDF.
cdef double _pdf_wrapper(double x, const unur_distr *distr):
    return _global_pdf(x, *_global_params)

# Wrap DPDF.
cdef double _dpdf_wrapper(double x, const unur_distr *distr):
    return _global_dpdf(x, *_global_params)

# Wrap PMF.
cdef double _pmf_wrapper(int k, const unur_distr *distr):
    return _global_pmf(k, *_global_params)


# ===========================================================================
# TDR method for sampling from continuous distributions.
# ===========================================================================


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
def tdr(pdf, dpdf, params=(), domain=None, Py_ssize_t size=1):
    """
    TDR Method for sampling from continuous distributions.

    Parameters
    ----------
    pdf : callable
        PDF of the distribution. The expected signature is:
        ``def pdf(x: float, *params) -> float`` where
        ``params`` are the parameters of the distribution.
    dpdf : callable
        Derivative of the PDF of the distribution. Same signature
        as the PDF.
    params : {tuple, list}, optional
        Parameters of the distribution. Default is an empty tuple.
    domain : {tuple, list}, optional
        A tuple or a list of two ``float``s. The first represents
        the lower bound and the second represents the upper bound.
        Default is negative infinity to infinity.
    size : int, optional
        The number of samples to draw from the distribution.
        Default is 1.

    Returns
    -------
    rvs : array_like
        Samples from the distribution.

    Examples
    --------
    >>> from scipy.stats._unuran import tdr
    
    Suppose we want to sample from the following distribution:

         /  1 - x*x , |x| ≤ 1
    x = <
         \     0    , otherwise

    As this is a continuous distribution and the derivative is easy to
    obtain, we can use the TDR method:

    >>> rvs = tdr(lambda x: 1 - x*x, lambda x: -2*x, params=(),
    ...           domain=(-1, 1), size=10000)

    We can verify that the samples are from the expected distribution:

    >>> import matploblib.pyplot as plt
    >>> pdf = lambda x: 1 - x*x
    >>> x = np.linspace(-1, 1, 1000)
    >>> px = pdf(x)
    >>> plt.plot(x, px)
    >>> plt.hist(rvs, bins=50, density=True)
    >>> plt.show()
    """
    cdef bitgen_t *_numpy_urng
    cdef unur_distr *distr
    cdef unur_par *par
    cdef unur_gen *rng
    cdef unur_urng *urng
    global _global_pdf, _global_dpdf, _global_params

    # Initialize NumPy's BitGenerator
    _numpy_rng = PCG64()
    _capsule = _numpy_rng.capsule
    cdef const char *_capsule_name = "BitGenerator"
    if not PyCapsule_IsValid(_capsule, _capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    _numpy_urng = <bitgen_t *> PyCapsule_GetPointer(_capsule, _capsule_name)

    # Create a new URNG that UNU.RAN understands
    urng = unur_urng_new(_numpy_urng.next_double,
                         <void *>(_numpy_urng.state))

    if ( urng == NULL ): # error
        raise RuntimeError("Failed to create the URNG!")

    # ==================================================================
    # TODO: Use ``ctypes`` to convert the Python function to Cython.
    #       The workaround currently is using global attributes.
    # ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p)
    # cdef object _pdf_wrapper = ftype(_pdf_wrapper_py)
    # cdef object _dpdf_wrapper = ftype(_dpdf_wrapper_py)
    # ==================================================================

    # Set the global attributes
    _global_params = params
    _global_pdf = pdf
    _global_dpdf = dpdf

    # Create a new distribution and set it's attributes.
    distr = unur_distr_cont_new()
    unur_distr_cont_set_pdf(distr, _pdf_wrapper)
    unur_distr_cont_set_dpdf(distr, _dpdf_wrapper)
    if domain:
        unur_distr_cont_set_domain(distr, domain[0], domain[1])
    # ==================================================================
    # TODO: use Cython functions instead of global wrappers.
    # unur_distr_cont_set_pdf(distr,
    #                         (<cont_func_t *><size_t>ctypes.addressof(
    #                                                     _pdf_wrapper
    #                                                 ))[0])
    # unur_distr_cont_set_dpdf(distr,
    #                          (<cont_func_t *><size_t>ctypes.addressof(
    #                                                      _dpdf_wrapper
    #                                                  ))[0])
    # ==================================================================

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
    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(size,
                                                         dtype=np.float64)
    cdef Py_ssize_t i
    for i in range(size):
        out[i] = unur_sample_cont(rng)

    # Free the remaining objects.
    unur_free(rng)
    unur_urng_free(urng)

    # Return the samples.
    return out


# ===========================================================================
# DAU method for sampling from discrete distributions.
# ===========================================================================


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
def dau(pv=None, pmf=None, params=(), domain=None, Py_ssize_t size=1):
    """
    DAU Method for sampling from continuous distributions.

    Parameters
    ----------
    pv : array_like, optional
        Probability vector. If probability vector is not available,
        PMF is expected. Default is None.
    pmf : callable, optional
        PMF of the distribution. The expected signature is:
        ``def pmf(k: int, *params) -> float`` where
        ``params`` are the parameters of the distribution. Only required
        if the probability vector is not available. Default is None.
    params : {tuple, list}, optional
        Parameters of the distribution. Default is an empty tuple.
    domain : {tuple, list}, optional
        A tuple or a list of two ``float``s. The first represents
        the lower bound and the second represents the upper bound.
        Default is negative infinity to infinity.
    size : int, optional
        The number of samples to draw from the distribution.
        Default is 1.

    Returns
    -------
    rvs : array_like
        Samples from the distribution.

    Examples
    --------
    >>> from scipy.stats._unuran import dau
    
    DAU accepts either a Probability Vector (PV) or the Probability
    Mass Function (PMF) of the discrete distribution. Say, we have a
    probability vector ``[0.18, 0.02, 0.8]``. We can use the DAU method
    to sample from this distribution:

    >>> rvs = dau([0.18, 0.02, 0.8], domain=(0, 2), size=10000)
    >>> rvs.mean()  # mean of the random variates.
    1.6257  # may vary
    >>> (0 * 0.18 + 1 * 0.02 + 2 * 0.8)  # mean of the actual distribution.
    1.62

    On the other hand, if a probability vector is not available, we can use
    a PMF:

    >>> from math import factorial as f
    >>> from math import pow
    >>> pmf = lambda x, n, p : f(n)/(f(x)*f(n-x)) * pow(p, x)*pow(1-p, n-x)
    >>> n, p = 10, 0.2
    >>> rvs = dau(pmf=pmf, params=(n, p), domain=(0, n), size=10000)
    >>> rvs.mean(), rvs.var()  # mean and variance of the random variates.
    (2.0088, 1.6161225600000002)  # may vary
    >>> n*p, n*p*(1-p)  # actual mean and variance
    (2.0, 1.6)
    """
    if pv and pmf:
        raise ValueError("expected either PV or PMF, not both.")
    if not (pv or pmf):
        raise ValueError("expected at least PV or PMF.")

    cdef bitgen_t *_numpy_urng
    cdef unur_distr *distr
    cdef unur_par *par
    cdef unur_gen *rng
    cdef unur_urng *urng
    cdef np.ndarray[np.float64_t, ndim=1, mode = 'c'] pv_view
    global _global_pmf, _global_params

    if pv:
        pv_view = np.asarray(pv, dtype=np.float64)
        pv_view.setflags(write=False)

    # Initialize NumPy's BitGenerator
    _numpy_rng = PCG64()
    _capsule = _numpy_rng.capsule
    cdef const char *_capsule_name = "BitGenerator"
    if not PyCapsule_IsValid(_capsule, _capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    _numpy_urng = <bitgen_t *> PyCapsule_GetPointer(_capsule, _capsule_name)

    # Create a new URNG that UNU.RAN understands
    urng = unur_urng_new(_numpy_urng.next_double,
                         <void *>(_numpy_urng.state))

    if ( urng == NULL ): # error
        raise RuntimeError("Failed to create the URNG!")

    # ==================================================================
    # TODO: Use ``ctypes`` to convert the Python function to Cython.
    #       The workaround currently is using global attributes.
    # ==================================================================

    # Set the global attributes
    _global_params = params
    if pmf:
        _global_pmf = pmf

    # Create a new distribution and set it's attributes.
    distr = unur_distr_discr_new()
    if pv:
        n_pv = len(pv_view)
        unur_distr_discr_set_pv(distr, <const double *>(pv_view.data), n_pv)
    else:
        unur_distr_discr_set_pmf(distr, _pmf_wrapper)
        # ===================================================================
        # TODO: use Cython functions instead of global wrappers.
        # unur_distr_discr_set_pmf(distr,
        #                          (<discr_func_t *><size_t>ctypes.addressof(
        #                                                       _pmf_wrapper
        #                                                   ))[0])
        # ===================================================================
    if domain:
        unur_distr_discr_set_domain(distr, domain[0], domain[1])

    par = unur_dau_new(distr)

    # Set the NumPy's BitGenerator to sample from.
    unur_set_urng(par, urng)

    # Initialize a UNU.RAN random number generator.
    rng = unur_init(par)

    if (rng == NULL): # error
        raise RuntimeError("Failed to initialize the RNG!")

    # Free the distribution as we don't need it anymore.
    unur_distr_free(distr)

    # Start sampling.
    cdef np.ndarray[np.int32_t, ndim=1] out = np.empty(size,
                                                       dtype=np.int32)
    cdef Py_ssize_t i
    for i in range(size):
        out[i] = unur_sample_discr(rng)

    # Free the remaining objects.
    unur_free(rng)
    unur_urng_free(urng)

    # Return the samples.
    return out
