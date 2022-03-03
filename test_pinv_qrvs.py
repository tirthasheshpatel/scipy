import sys

print(sys.version)

import numpy as np
from scipy import stats, special
from scipy.stats import sampling, qmc

class StandardNormal:
    def pdf(self, x):
        # normalization constant needed for NumericalInverseHermite
        return 1./np.sqrt(2.*np.pi) * np.exp(-0.5 * x*x)

    def dpdf(self, x):
        return 1./np.sqrt(2.*np.pi) * -x * np.exp(-0.5 * x*x)

    def cdf(self, x):
        return special.ndtr(x)

print("running pinv")
gen = sampling.NumericalInversePolynomial(stats.geninvgauss(0.5, 1.))
shape_expected = (4, 3)
qrvs = gen.qrvs(size=(4,), d=3, qmc_engine=None)
assert(qrvs.shape == shape_expected)
