cdef extern from "unuran.h":
    struct unur_gen
    struct unur_urng


cdef class TDR:
    cdef unur_gen *_rng
    cdef unur_urng *_urng
    # cdef object params
    # cdef object pdf
    # cdef object dpdf
    # cdef object _pdf_wrapper
    # cdef object _dpdf_wrapper
    # cdef object _pdf_wrapper_py
    # cdef object _dpdf_wrapper_py
    cdef object _numpy_rng


cdef class DAU:
    cdef unur_gen *_rng
    cdef unur_urng *_urng
    cdef object _numpy_rng
