import pytest
from numpy.testing import assert_allclose
import scipy.stats._unuran._unuran_wrapper as _unuran_wrapper


def test_randn():
    # this is a dummy test to see if UNU.RAN
    # builds properly.
    rvs = _unuran_wrapper.randn(100000)
    assert_allclose(rvs.mean(), 0, atol=1e-2)
    assert_allclose(rvs.std(), 1, atol=1e-2)
