import pytest
import warnings
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats._unuran import TDR, DAU
from scipy import stats


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
        with pytest.raises(ValueError, match=r"left >= right"):
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


class TestDAU:
    @pytest.mark.parametrize("dist, params",
        [  # discrete distributions with finite support.
            (stats.hypergeom, (20, 7, 12)),
            (stats.nhypergeom, (20, 7, 12)),
            (stats.binom, (20, 0.3))
        ]
    )
    def test_sampling_with_pmf(self, dist, params):
        domain = dist.support(*params)
        pmf = dist._pmf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rng = DAU(pmf, domain=domain, params=params, seed=123)
        rvs = rng.sample(100_000)
        mv = rvs.mean(), rvs.var()
        mv_expected = dist.stats(*params, moments='mv')
        assert_allclose(mv, mv_expected, atol=1e-2)

    @pytest.mark.parametrize("pv",
        [
            [0.18, 0.02, 0.8],
            [1.],
            [0., 0., 0., 0., 1.],
            [1., 2., 3., 4., 5., 6.]
        ]
    )
    def test_sampling_with_pv(self, pv):
        pv = np.asarray(pv, dtype=np.float64)
        rng = DAU(pv, seed=123)
        rvs = rng.sample(100_000)
        pv /= pv.sum()
        variates = np.arange(0, len(pv))
        m_expected = np.average(variates, weights=pv)
        v_expected = np.average((variates-m_expected)**2, weights=pv)
        mv_expected = m_expected, v_expected
        mv = rvs.mean(), rvs.var()
        assert_allclose(mv, mv_expected, atol=1e-2)

    @pytest.mark.parametrize("pmf, msg",
        [
            (None, r"must not be None"),
            (lambda x: None, r"must be real number"),
            (lambda x: undef_name, r"name 'undef_name' is not defined"),
            (lambda x: -x, r"probability < 0")
        ]
    )
    def test_bad_pmf(self, pmf, msg):
        with pytest.raises(Exception, match=msg):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                DAU(pmf, domain=(0, 10))

    @pytest.mark.parametrize("domain", [(0, 0), (1, 0)])
    def test_bad_domain(self, domain):
        with pytest.raises(ValueError, match=r"left >= right"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                DAU(lambda x: x, domain=domain)

    @pytest.mark.parametrize("domain", [(np.nan, np.nan), (np.inf, np.nan),
                                        (np.nan, -np.inf), (np.nan, 0),
                                        (-1, np.nan), (0, float("nan"))])
    def test_nan_domain(self, domain):
        with pytest.raises(ValueError, match=r"must contain only non-nan values"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                DAU(lambda x: x, domain=domain)

    @pytest.mark.parametrize("domain", [(-np.inf, np.inf), (np.inf, np.inf),
                                        (-np.inf, -np.inf), (0, np.inf),
                                        (-np.inf, 0)])
    def test_inf_domain(self, domain):
        with pytest.raises(ValueError, match=r"must be finite"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                DAU(lambda x: x, domain=domain)

    @pytest.mark.parametrize("pv",
        [
            [0.], [], [0., 0., 0.],
            [0., np.inf], [-np.inf, 0.],
            [np.inf, np.inf], [-np.inf, -np.inf],
            [np.nan], [np.nan, np.inf, -np.inf],
            [[1., 0.], [0.5, 0.5]]
        ]
    )
    def test_bad_pv(self, pv):
        with pytest.raises(ValueError):
            DAU(pv)

    def test_bad_urnfactor(self):
        with pytest.warns(UserWarning, match=r"relative urn size < 1."):
            DAU([0.5, 0.5], urnfactor=-1)
