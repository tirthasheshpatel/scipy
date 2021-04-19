import pathlib


def isNPY_OLD():
    '''
    A new random C API was added in 1.18 and became stable in 1.19.
    Prefer the new random C API when building with recent numpy.
    '''
    import numpy as np
    ver = tuple(int(num) for num in np.__version__.split('.')[:2])
    return ver < (1, 19)


def make_unuran():
    '''Substitute True/False values for NPY_OLD Cython build variable.'''
    unuran_base = (pathlib.Path(__file__).parent / '_unuran_wrapper').absolute()
    error_info = {
        "err_reason": "",
        "err_objid": "",
        "err_errortype": ""
    }
    with open(unuran_base.with_suffix('.pyx.templ'), 'r') as src:
        contents = src.read()
    with open(unuran_base.with_suffix('.pyx'), 'w') as dest:
        dest.write(contents.format(NPY_OLD=str(bool(isNPY_OLD()))))


if __name__ == '__main__':
    make_unuran()
