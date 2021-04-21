import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats._unuran import TDR, DAU
from scipy import stats


def test_dau():
    from math import factorial as f
    from math import pow
    # check with PV.
    rng = DAU([0.18, 0.02, 0.8], domain=(0, 2), seed=123)
    rvs = rng.sample(size=100_000)
    expected = (0 * 0.18 + 1 * 0.02 + 2 * 0.8)
    assert_allclose(rvs.mean(), expected, rtol=1e-2, atol=1e-2)
    # check with PMF.
    pmf = lambda x, n, p : f(n)/(f(x)*f(n-x)) * pow(p, x)*pow(1-p, n-x)
    n, p = 10, 0.2
    with pytest.warns(UserWarning, match=r"PV. Try to compute it."):
        rng = DAU(pmf, params=(n, p), domain=(0, n), urnfactor=2, seed=123)
    rvs = rng.sample(size=100_000)
    assert_allclose(rvs.mean(), n*p, rtol=1e-2, atol=1e-2)
    assert_allclose(rvs.var(), n*p*(1-p), rtol=1e-2, atol=1e-2)


class TestTDR:
    @pytest.mark.parametrize("dist, dpdf, params",
        [
            (  # test case in gh-13051
                stats.norm,
                lambda x, loc, scale: (-(x-loc)/(scale**2) *
                                      stats.norm._pdf((x-loc)/scale)),
                (0., 0.1)
            ),
            (
                stats.expon,
                lambda x: -stats.expon._pdf(x),
                ()
            ),
            (
                stats.laplace,
                lambda x: 0 if x == 0 else (
                    -stats.laplace._pdf(x) if x > 0 else stats.laplace._pdf(x)
                ),
                ()
            )
        ]
    )
    def test_sampling(self, dist, dpdf, params):
        domain = dist.support(*params)
        # call the private method to avoid validations and expensive
        # numpy operations.
        pdf = lambda x, loc=0, scale=1: dist._pdf((x-loc)/scale)
        rng = TDR(pdf, dpdf, params=params, domain=domain, seed=123)
        rvs = rng.sample(100_000)
        mv_expected = dist.stats(*params, moments='mv')
        mv = rvs.mean(), rvs.var()
        assert_allclose(mv, mv_expected, atol=1e-2)

    @pytest.mark.parametrize("pdf, dpdf, msg",
        [
            (lambda x: -x, lambda x: -1, r"PDF\(x\) < 0."),
            (lambda x: None, lambda x: None, r"must be real number"),
            (lambda x: undef_name, lambda x: 1,
             r"name 'undef_name' is not defined")
        ]
    )
    def test_bad_pdf(self, pdf, dpdf, msg):
        with pytest.raises(Exception, match=msg):
            rng = TDR(pdf, dpdf)

    @pytest.mark.parametrize("domain", [(0, 0), (1, 0), (np.inf, np.inf),
                                        (-np.inf, -np.inf)])
    def test_bad_domain(self, domain):
        with pytest.raises(ValueError, match=r"domain, left >= right"):
            TDR(lambda x: x, lambda x: 1, domain=domain)

    @pytest.mark.parametrize("domain", [(np.nan, np.nan), (np.inf, np.nan),
                                        (np.nan, -np.inf), (np.nan, 0),
                                        (-1, np.nan), (0, float("nan"))])
    def test_nan_domain(self, domain):
        with pytest.raises(ValueError, match=r"only non-nan values"):
            TDR(lambda x: x, lambda x: 1, domain=domain)

    def test_bad_cpoints(self):
        # test bad cpoints
        with pytest.warns(UserWarning, match=r"number of starting points < 0"):
            TDR(lambda x: x, lambda x: 1, domain=(0, 10), cpoints=-10)

        with pytest.warns(UserWarning, match=r"hat/squeeze ratio too small"):
            TDR(lambda x: 1-x*x, lambda x: -2*x, domain=(-1,1), cpoints=1)

    def test_bad_c(self):
        # c < -0.5
        with pytest.raises(ValueError, match=r"c < -0.5 not implemented yet"):
            TDR(lambda x: x, lambda x: 1, domain=(0, 10), c=-1.)
        with pytest.raises(ValueError, match=r"c < -0.5 not implemented yet"):
            TDR(lambda x: x, lambda x: 1, domain=(0, 10), c=-np.inf)

        #  c > 0
        with pytest.warns(UserWarning, match=r"c > 0"):
            TDR(lambda x: x, lambda x: 1, domain=(0, 10), c=10.)

    def test_bad_variant(self):
        with pytest.raises(ValueError, match=r"Invalid option for the variant"):
            TDR(lambda x: x, lambda x: 1, domain=(0, 10), variant="foo")
