import re
import os
import multiprocessing
import time
from functools import partial
import numpy as np
from scipy import stats, integrate, optimize
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
        if x == 0 or x == 1:
            return np.nan
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


# Gamma Distribution
class gamma2:
    scipy_dist = stats.gamma(2)
    numpy_dist = ["gamma", (2,)]
    def __init__(self):
        self.mode = 1
        self.area = integrate.quad(self.pdf, 0, np.inf)[0]
        self.domain = (0, np.inf)
    def pdf(self, x):
        return x * np.exp(-x)
    def logpdf(self, x):
        if x == 0:
            return -np.inf
        return np.log(x) - x
    def dpdf(self, x):
        return np.exp(-x) - x * np.exp(-x)
    def dlogpdf(self, x):
        if x == 0:
            return np.inf
        return 1/x - 1
    def cdf(self, x):
        return stats.gamma._cdf(x, 2)
    def __repr__(self):
        return 'gamma(2)'


# Generalized Normal Distribution
class gennorm3:
    scipy_dist = stats.gennorm(3)
    numpy_dist = None
    def __init__(self):
        self.mode = 0
        self.area = integrate.quad(self.pdf, -np.inf, np.inf)[0]
        self.domain = (-np.inf, np.inf)
    def pdf(self, x):
        return np.exp(-np.abs(x)**3)
    def logpdf(self, x):
        return -np.abs(x)**3
    def dpdf(self, x):
        return -3 * np.sign(x) * x*x * np.exp(-np.abs(x)**3)
    def dlogpdf(self, x):
        return -3 * np.sign(x) * x*x
    def cdf(self, x):
        return stats.gennorm._cdf(x, 3)
    def __repr__(self):
        return 'gennorm(3)'


# Gauss hypergeometric Distribution
class gausshyper:
    a, b, c, z = 13.7637716041307, 3.118963664868143, 2.514598035018302, 5.1811649903971615
    scipy_dist = stats.gausshyper(a,b,c,z)
    numpy_dist = None
    def __init__(self):
        self.mode = optimize.minimize_scalar(lambda x: -self.pdf(x), method='bounded', bounds=(0, 1)).x
        self.area = integrate.quad(self.pdf, 0, 1)[0]
        self.domain = (0, 1)
    def pdf(self, x):
        return x**(self.a-1) * (1-x)**(self.b-1) * (1+self.z*x)**(-self.c)
    def logpdf(self, x):
        if x == 0 or x == 1:
            return -np.inf
        return (self.a-1)*np.log(x) + (self.b-1)*np.log(1-x) - self.c*np.log(1+self.z*x)
    def dpdf(self, x):
        a  =  (self.a-1) * x**(self.a-2) * (1-x)**(self.b-1) * (1+self.z*x)**(-self.c)
        b  = -(self.b-1) * (1-x)**(self.b-2) * x**(self.a-1) * (1+self.z*x)**(-self.c)
        cz = -self.c*self.z * (1+self.z*x)**(-self.c-1) * x**(self.a-1) * (1-x)**(self.b-1)
        return a+b+cz
    def dlogpdf(self, x):
        if x == 0 or x == 1:
            return np.nan
        return (self.a-1)/x - (self.b-1)/(1-x) - self.c*self.z/(1+self.z*x)
    def cdf(self, x):
        return stats.gausshyper._cdf(x, self.a, self.b, self.c, self.z)
    def __repr__(self):
        return 'gausshyper'


# Generalized Exponential Distribution
class genexpon:
    a, b, c = 9.1325976465418908, 16.231956600590632, 3.2819552690843983
    scipy_dist = stats.genexpon(a,b,c)
    numpy_dist = None
    def __init__(self):
        self.mode = optimize.minimize_scalar(lambda x: -self.pdf(x), method='bounded', bounds=(0, np.inf)).x
        self.area = 1
        self.domain = (0, np.inf)
    def pdf(self, x):
        return ( self.a + self.b * (1-np.exp(-self.c*x)) ) * np.exp( -self.a*x - self.b*x + self.b/self.c * (1-np.exp(-self.c*x)) )
    def logpdf(self, x):
        return np.log(self.a + self.b * (1-np.exp(-self.c*x))) - self.a*x - self.b*x + self.b/self.c * (1-np.exp(-self.c*x))
    def dpdf(self, x):
        first = self.b*self.c*np.exp(-self.c*x) * np.exp( -self.a*x - self.b*x + self.b/self.c * (1-np.exp(-self.c*x)) )
        second = (-self.a - self.b + self.b * np.exp(-self.c*x)) * self.pdf(x)
        return first + second
    def dlogpdf(self, x):
        first = (self.b*self.c*np.exp(-self.c*x)) / (self.a + self.b * (1-np.exp(-self.c*x)))
        second = -self.a - self.b + self.b * np.exp(-self.c*x)
        return first + second
    def cdf(self, x):
        return stats.genexpon._cdf(x, self.a, self.b, self.c)
    def __repr__(self):
        return 'genexpon'


# Inverse Gamma Distribution
class invgamma:
    a = 4.0668996136993067
    scipy_dist = stats.invgamma(a)
    numpy_dist = None
    def __init__(self):
        self.mode = 1 / (self.a + 1)
        self.area = integrate.quad(self.pdf, 0, np.inf)[0]
        self.domain = (0, np.inf)
    def pdf(self, x):
        if x == 0:
            return 0.0
        return x**(-self.a-1) * np.exp(-1./x)
    def logpdf(self, x):
        if x == 0:
            return -np.inf
        return -(self.a+1)*np.log(x) - 1/x
    def dpdf(self, x):
        if x == 0:
            return 0
        first = (-self.a-1) * x**(-self.a-2) * np.exp(-1/x)
        second = 1/x**2 * self.pdf(x)
        return first + second
    def dlogpdf(self, x):
        if x == 0:
            return np.nan
        return -(self.a+1)/x + 1/x**2
    def cdf(self, x):
        if x == 0:
            return 0
        return stats.invgamma._cdf(x, self.a)
    def __repr__(self):
        return f'invgamma({self.a})'


# Nakagam Distribution
class nakagami:
    m = 4.9673794866666237
    scipy_dist = stats.nakagami(m)
    numpy_dist = None
    def __init__(self):
        self.mode = np.sqrt(2)/2 * np.sqrt((2*self.m-1)/self.m)
        self.area = integrate.quad(self.pdf, 0, np.inf)[0]
        self.domain = (0, np.inf)
    def pdf(self, x):
        if x == 0:
            return 0.0
        x = np.asarray(x)
        return x**(2*self.m-1) * np.exp(-self.m * x**2)
    def logpdf(self, x):
        if x == 0:
            return -np.inf
        x = np.asarray(x)
        return (2*self.m-1)*np.log(x) - self.m*x**2
    def dpdf(self, x):
        if x == 0:
            return 0
        x = np.asarray(x)
        first = (2*self.m-1)*x**(2*self.m-2) * np.exp(-self.m * x**2)
        second = -2 * self.m * x * self.pdf(x)
        return first + second
    def dlogpdf(self, x):
        if x == 0:
            return np.nan
        x = np.asarray(x)
        return (2*self.m-1)/x - 2*self.m*x
    def cdf(self, x):
        if x == 0:
            return 0
        x = np.asarray(x)
        return stats.nakagami._cdf(x, self.m)
    def __repr__(self):
        return f'nakagami({self.m})'


# Studentized Range Distribution
class studentized_range310:
    scipy_dist = stats.studentized_range(3.0, 10.0)
    numpy_dist = None
    def __init__(self):
        self.mode = optimize.minimize_scalar(lambda x: -self.pdf(x)).x
        self.area = 1
        self.domain = (-np.inf, np.inf)
    def pdf(self, x):
        return stats.studentized_range._pdf(x, 3.0, 10.0)
    def logpdf(self, x):
        return stats.studentized_range._logpdf(x, 3.0, 10.0)
    def cdf(self, x):
        return stats.studentized_range._cdf(x, 3.0, 10.0)
    def __repr__(self):
        return 'studentized_range(3, 10)'


# Ksone distribution
class ksone:
    scipy_dist = stats.ksone(1000)
    numpy_dist = None
    def __init__(self):
        self.mode = optimize.minimize_scalar(lambda x: -self.pdf(x), method='bounded', bounds=(0, 1)).x
        self.area = 1
        self.domain = (0, 1)
    def pdf(self, x):
        return stats.ksone._pdf(x, 1000)
    def logpdf(self, x):
        return stats.ksone._logpdf(x, 1000)
    def cdf(self, x):
        return stats.ksone._cdf(x, 1000)
    def __repr__(self):
        return 'ksone(1000)'


# KStwo distribution
class kstwo:
    scipy_dist = stats.kstwo(10)
    numpy_dist = None
    def __init__(self):
        self.mode = optimize.minimize_scalar(lambda x: -self.pdf(x), method='bounded', bounds=self.scipy_dist.support()).x
        self.area = 1
        self.domain = self.scipy_dist.support()
    def pdf(self, x):
        return stats.kstwo._pdf(x, 10)
    def logpdf(self, x):
        return stats.kstwo._logpdf(x, 10)
    def cdf(self, x):
        return stats.kstwo._cdf(x, 10)
    def __repr__(self):
        return 'kstwo(10)'


allcontdists = [
    kstwo(),
    ksone(),
    studentized_range310(),
    nakagami(),
    invgamma(),
    genexpon(),
    gausshyper(),
    gennorm3(),
    gamma2(),
    stdnorm(),
    beta23()
]


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


if __name__ == "__main__":
    processes = []

    for dist_obj in allcontdists:
        def plot_func(dist):
            print(f"Generating plots for distribution {dist}")
            rngs = []
            with np.testing.suppress_warnings() as sup:
                sup.filter(RuntimeWarning)
                sup.filter(integrate.IntegrationWarning)
                for methodname in methods:
                    try:
                        def func():
                            try:
                                get_rng(methodname, dist)
                            except:
                                pass
                        tracker = multiprocessing.Process(target=func)
                        start = time.time()
                        tracker.start()
                        while tracker.is_alive():
                            time.sleep(0.1)
                            if time.time() - start > 30:
                                tracker.terminate()
                                raise TimeoutError(f"{methodname} is taking too long, aborting.")
                        tracker.join()
                        rngs.append(get_rng(methodname, dist))
                    except (stats.UNURANError, ValueError, TimeoutError) as e:
                        print(f"[{dist}] [Skipping] Creation of {methodname} failed with error -- {e.args[0]}")
                np_rng = np.random.default_rng(123)

                if dist.numpy_dist:
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

                if not os.path.exists("plots"):
                    os.mkdir(f"plots")

                perfplot.save(
                    filename=os.path.join("plots", str(dist) + ".svg"),
                    setup=lambda n: n,
                    kernels=kernels,
                    labels=[rng[1] for rng in rngs],
                    n_range=[2**k for k in range(0, 26)],
                    xlabel="n (number of samples)",
                    equality_check=None,
                    max_time=5,
                    logx=True,
                    logy=True,
                    show_progress=False
                )
                print(f"Completed generating plots for {dist}")
        processes.append(multiprocessing.Process(target=plot_func, args=(dist_obj,)))
    for process in processes:
        process.start()
    for process in processes:
        process.join()
