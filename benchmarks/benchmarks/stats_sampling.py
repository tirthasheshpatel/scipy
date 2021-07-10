import numpy as np
from .common import Benchmark, safe_import

with safe_import():
    from scipy import stats, integrate
with safe_import():
    from scipy.stats._distr_params import distdiscrete


# Beta distribution with a = 2, b = 3
class beta23:
    scipy_dist = stats.beta(2, 3)
    def __init__(self):
        self.mode = 1/3
        self.pdfarea = integrate.quad(self.pdf, 0, 1)[0]
        self.domain = (0, 1)
    def pdf(self, x):
        return x * (1-x)**2
    def logpdf(self, x):
        if x == 0 or x == 1:
            return -np.inf
        return np.log(x) + 2*np.log(1-x)
    def dpdf(self, x):
        return (1-x)**2 - 2*x*(1-x)
    def dlogpdf(self, x):
        return 1/x - 2/(1-x)
    def cdf(self, x):
        return stats.beta._cdf(x, 2, 3)
    def __repr__(self):
        # asv prints this.
        return 'beta(2, 3)'


# Standard Normal Distribution
class stdnorm:
    scipy_dist = stats.norm()
    def __init__(self):
        self.mode = 0
        self.pdfarea = integrate.quad(self.pdf, -np.inf, np.inf)[0]
        self.domain = (-np.inf, np.inf)
    def pdf(self, x):
        return np.exp(-0.5 * x*x)
    def logpdf(self, x):
        return -0.5 * x*x
    def dpdf(self, x):
        return -x * np.exp(-0.5 * x*x)
    def dlogpdf(self, x):
        return -x
    def cdf(self, x):
        return stats.norm._cdf(x)
    def __repr__(self):
        return 'norm(0, 1)'


allcontdists = [beta23(), stdnorm()]


class AllContinuous(Benchmark):

    param_names = ['method', 'dist', 'n_samples']

    params = [
        [('TransformedDensityRejection', ),
         ('AutomaticRatioOfUniforms', ),
         ('AdaptiveRejectionSampling', ),
         ('NumericalInverseHermiteUNURAN', ),
         ('NumericalInverse', ),
         ('NaiveRatioOfUniforms', 'mode'),
         ('NumericalInversePolynomial', ),
         ('SimpleRatioOfUniforms', 'mode', 'area'),
         ('SimpleSetupRejection', 'mode', 'area'),
         ('PiecewiseConstantHatsTable', 'mode', 'area')],
        allcontdists,
        [10, 100, 1000, 10000, 100000]
    ]

    def setup(self, method, dist, n_samples):
        Method = getattr(stats, method[0])
        args = method[1:]
        kwargs = {}
        if 'mode' in args:
            kwargs['mode'] = dist.mode
        if 'area' in args:
            kwargs['area'] = dist.pdfarea
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            try:
                self.rng = Method(dist, domain=dist.domain, seed=123, **kwargs)
            except stats.UNURANError as e:
                raise NotImplementedError(f'{method}, {dist}, {n_samples} : RNG creation failed with error -- {e.args[0]}')

    def time_rvs(self, method, dist, n_samples):
        self.rng.rvs(n_samples)


class AllContinuousSciPy(Benchmark):

    param_names = ['dist', 'n_samples']

    params = [allcontdists, [10, 100, 1000, 10000, 100000]]

    def time_scipy_rvs(self, dist, n_samples):
        dist.scipy_dist.rvs(size=n_samples, random_state=123)

class AllContinuousNumPy(Benchmark):

    param_names = ['dist', 'n_samples']

    params = [[('standard_normal', ), ('beta', 2, 3)],
              [10, 100, 1000, 10000, 100000]]

    def setup(self, dist, n_samples):
        rng = np.random.default_rng(123)
        self.rng = getattr(rng, dist[0])
        self.args = dist[1:]

    def time_numpy_rvs(self, dist, n_samples):
        self.rng(*self.args, size=n_samples)


class TransformedDensityRejection(Benchmark):

    param_names = ['dist', 'c', 'cpoints']

    params = [allcontdists, [0., -0.5], [10, 20, 30, 50]]

    def setup(self, dist, c, cpoints):
        self.urng = np.random.default_rng(0xfaad7df1c89e050200dbe258636b3265)
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            try:
                self.rng = stats.TransformedDensityRejection(dist, c=c,
                                                             cpoints=cpoints,
                                                             seed=self.urng)
            except RuntimeError:
                # contdist3 is not T-concave for c=0. So, skip such test-cases
                raise NotImplementedError(f"{dist} not T-concave for c={c}")

    def time_tdr_setup(self, dist, c, cpoints):
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            rng = stats.TransformedDensityRejection(dist, c=c,
                                                    cpoints=cpoints,
                                                    seed=self.urng)

    def time_tdr_rvs(self, dist, c, cpoints):
        rvs = self.rng.rvs(100000)


class DiscreteAliasUrn(Benchmark):

    param_names = ['distribution']

    params = [
        # a subset of discrete distributions with finite domain.
        [['nhypergeom', (20, 7, 1)],
         ['hypergeom', (30, 12, 6)],
         ['nchypergeom_wallenius', (140, 80, 60, 0.5)],
         ['binom', (5, 0.4)]]
    ]

    def setup(self, distribution):
        distname, params = distribution
        dist = getattr(stats, distname)
        domain = dist.support(*params)
        self.urng = np.random.default_rng(0x2fc9eb71cd5120352fa31b7a048aa867)
        x = np.arange(domain[0], domain[1] + 1)
        self.pv = dist.pmf(x, *params)
        self.rng = stats.DiscreteAliasUrn(self.pv, seed=self.urng)

    def time_dau_setup(self, distribution):
        rng = stats.DiscreteAliasUrn(self.pv, seed=self.urng)

    def time_dau_rvs(self, distribution):
        rvs = self.rng.rvs(100000)
