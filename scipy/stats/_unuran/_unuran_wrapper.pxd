import numpy as np
cimport numpy as np

# Signature of PDF, etc that UNU.RAN expects.
ctypedef double (*cont_func_t)(double, const unur_distr *)
ctypedef double (*discr_func_t)(int, const unur_distr *)

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
    unur_urng * unur_set_default_urng(unur_urng *urng_new)
    unur_urng * unur_set_default_urng_aux(unur_urng *urng_new)
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
    int unur_tdr_set_c(unur_par *parameters, double c)
    int unur_tdr_set_variant_ia(unur_par *parameters)
    int unur_tdr_set_variant_ps(unur_par *parameters)
    int unur_tdr_set_variant_gw(unur_par *parameters)
    int unur_tdr_set_cpoints(unur_par *parameters, int n_stp,
                             const double *stp)

    # Discrete Distributions
    unur_distr * unur_distr_discr_new()
    int unur_distr_discr_set_pmf(unur_distr *distribution,
                                 discr_func_t pmf)
    # int unur_distr_discr_set_pmfparams(unur_distr *distribution,
    #                                    const double *params, int n_params)
    # int unur_distr_discr_get_pmfparams(const unur_distr *distribution,
    #                                    const double **params)
    int unur_distr_discr_set_domain(unur_distr *distribution,
                                    int left, int right)
    int unur_distr_discr_set_pv(unur_distr *distribution,
                                const double *pv, int n_pv)
    unur_par *unur_dau_new(const unur_distr *distribution)
    int unur_dau_set_urnfactor(unur_par* parameters, double factor)

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


cdef class Method:
    cdef unur_gen *_rng
    cdef unur_urng *_urng
    cdef public object _numpy_rng
    cdef void _set_rng(self, object seed, unur_par *par,
                       unur_distr *distr) except *
    cdef inline np.ndarray[np.float64_t, ndim=1] _sample_cont(
        self,
        Py_ssize_t size
    )
    cdef inline np.ndarray[np.int32_t, ndim=1] _sample_discr(
        self,
        Py_ssize_t size
    )
    # cdef object params
    # cdef object pdf
    # cdef object dpdf
    # cdef object _pdf_wrapper
    # cdef object _dpdf_wrapper
    # cdef object _pdf_wrapper_py
    # cdef object _dpdf_wrapper_py

cdef class TDR(Method):
    pass


cdef class DAU(Method):
    pass
