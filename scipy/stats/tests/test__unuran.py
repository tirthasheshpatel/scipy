import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats._unuran import randn, TDR, DAU
import pytest


def test_randn():
    # XXX: remove this later.
    # this is a dummy test to see if UNU.RAN
    # builds properly.
    rvs = randn(100_000, seed=123)
    assert_allclose(rvs.mean(), 0, atol=1e-2)
    assert_allclose(rvs.std(), 1, atol=1e-2)


def test_tdr():
    # A small test to ensure that this is working.
    # See the test case in gh-13051
    from math import exp, sqrt, pow, pi
    pdf = lambda x: exp(-pow((x/.1), 2) / 2) / sqrt(2*pi*.1)
    dpdf = lambda x: -x/pow(.1, 2) * pdf(x)
    rng = TDR(pdf, dpdf, c=0., cpoints=20, variant='gw', seed=123)
    rvs = rng.sample(size=100_000)
    assert_allclose(rvs.mean(), 0, rtol=1e-2, atol=1e-2)
    assert_allclose(rvs.std(), 0.1, rtol=1e-2, atol=1e-2)


def test_dau():
    from math import factorial as f
    from math import pow
    # check with PV.
    rng = DAU([0.18, 0.02, 0.8], domain=(0, 2))
    rvs = rng.sample(size=100_000)
    expected = (0 * 0.18 + 1 * 0.02 + 2 * 0.8)
    assert_allclose(rvs.mean(), expected, rtol=1e-2, atol=1e-2)
    # check with PMF.
    pmf = lambda x, n, p : f(n)/(f(x)*f(n-x)) * pow(p, x)*pow(1-p, n-x)
    n, p = 10, 0.2
    with pytest.raises(UserWarning, match=r"PV. Try to compute it."):
        rng = DAU(pmf, params=(n, p), domain=(0, n), urnfactor=2, seed=123)
    rvs = rng.sample(size=100_000)
    assert_allclose(rvs.mean(), n*p, rtol=1e-2, atol=1e-2)
    assert_allclose(rvs.var(), n*p*(1-p), rtol=1e-2, atol=1e-2)
