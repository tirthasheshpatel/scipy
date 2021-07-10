import re
from functools import partial
import numpy as np
from scipy import stats, integrate
import perfplot


# Beta distribution with a = 2, b = 3
class beta23:
    scipy_dist = stats.beta(2, 3)
    numpy_dist = ["beta", (2, 3)]
    def __init__(self):
        self.mode = 1/3
        self.area = integrate.quad(self.pdf, 0, 1)[0]
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
        return 'beta(2, 3)'


# Standard Normal Distribution
class stdnorm:
    scipy_dist = stats.norm()
    numpy_dist = ["standard_normal", ()]
    def __init__(self):
        self.mode = 0
        self.area = integrate.quad(self.pdf, -np.inf, np.inf)[0]
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


methods = [
    'TransformedDensityRejection',
    'AutomaticRatioOfUniforms',
    'AdaptiveRejectionSampling',
    'NumericalInverseHermiteUNURAN',
    'NumericalInverse',
    'NaiveRatioOfUniforms[mode]',
    'NumericalInversePolynomial',
    'SimpleRatioOfUniforms[mode, area]',
    'SimpleSetupRejection[mode, area]',
    'PiecewiseConstantHatsTable[mode, area]'
]


def get_rng(methodname, dist):
    # parse the method string
    match = re.match(r"([A-Za-z][A-Za-z0-9]*) *(\[(.*)\])?", methodname)
    methodname = match.group(1)
    methodargs = match.group(3)
    if methodargs is not None:
        methodargs = [x.strip() for x in methodargs.split(',')]
    else:
        methodargs = []
    methodkwargs = {}
    for arg in methodargs:
        if arg.partition("=")[1]:
            arg, _, val = arg.partition("=")
            arg, val = arg.strip(), val.strip()
            match = re.match(r"\((.*)\) *(.*)", val)
            dtype, val = match.group(1), match.group(2)
            if dtype == 'int': val = int(val)
            elif dtype == 'float': val = float(val)
            elif dtype == 'str': val = val
            else: raise NotImplementedError("unknown dtype.")
        else:
            val = getattr(dist, arg)
        methodkwargs[arg] = val
    method = getattr(stats, methodname)
    rng = method(dist, domain=dist.domain, seed=123, **methodkwargs)
    return rng, methodname


for dist in allcontdists:
    print(f"Generating plots for distribution {dist}")
    rngs = []
    for methodname in methods:
        try:
            rngs.append(get_rng(methodname, dist))
        except stats.UNURANError as e:
            print(f"[Skipping] Creation of {methodname} failed with error -- {e.args[0]}")
    np_rng = np.random.default_rng(123)
    class numpy_rng:
        pass
    np_dist = partial(getattr(np_rng, dist.numpy_dist[0]), *dist.numpy_dist[1])
    numpy_rng.rvs = lambda size: np_dist(size=size)
    rngs.append([numpy_rng, "NumPy RNG"])
    class scipy_rng:
        pass
    scipy_rng.rvs = lambda size: dist.scipy_dist.rvs(size=size, random_state=np.random.default_rng(123))
    rngs.append([scipy_rng, "SciPy RVS Method"])

    def funcwrap(rng):
        def func(size):
            return rng.rvs(size=size)
        return func

    kernels = [funcwrap(rng) for rng, _ in rngs]

    perfplot.show(
        # filename=str(dist) + "_sampling.png",
        setup=lambda n: n,
        kernels=kernels,
        labels=[rng[1] for rng in rngs],
        n_range=[2**k for k in range(0, 26)],
        xlabel="n",
        equality_check=None,
        max_time=5,
        logx=True,
        logy=False,
        target_time_per_measurement=3
    )
