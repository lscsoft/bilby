#!/usr/bin/env python
"""

As part of the :code:`bilby.result.Result` object, we provide a method to
calculate the Occam factor (c.f., Chapter 28, `Mackay "Information Theory,
Inference, and Learning Algorithms"
<http://www.inference.org.uk/itprnn/book.html>`). This is an approximate
estimate based on the posterior samples, and assumes the posteriors are well
approximate by a Gaussian.

The Occam factor penalizes models with larger numbers of parameters (or
equivalently a larger "prior volume"). This example won't try to go through
explaining the meaning of this, or how it is calculated (those details are
sufficiently well done in Mackay's book linked above). Instead, it demonstrates
how to calculate the Occam factor in :code:`bilby` and shows an example of it
working in practise.

If you have a :code:`result` object, the Occam factor can be calculated simply
from :code:`result.occam_factor(priors)` where :code:`priors` is the dictionary
of priors used during the model fitting. These priors should be uniform
priors only. Other priors may cause unexpected behaviour.

In the example, we generate a data set which contains Gaussian noise added to a
quadratic function. We then fit polynomials of differing degree. The final plot
shows that the largest evidence favours the quadratic polynomial (as expected)
and as the degree of polynomial increases, the evidence falls of in line with
the increasing (negative) Occam factor.

Note - the code uses a course 100-point estimation for speed, results can be
improved by increasing this to say 500 or 1000.

"""
import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils.random import rng, seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# A few simple setup steps
label = "occam_factor"
outdir = "outdir"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

sigma = 1

N = 100
time = np.linspace(0, 1, N)
coeffs = [1, 2, 3]
data = np.polyval(coeffs, time) + rng.normal(0, sigma, N)

fig, ax = plt.subplots()
ax.plot(time, data, "o", label="data", color="C0")
ax.plot(time, np.polyval(coeffs, time), label="true signal", color="C1")
ax.set_xlabel("time")
ax.set_ylabel("y")
ax.legend()
fig.savefig("{}/{}_data.png".format(outdir, label))


class Polynomial(bilby.Likelihood):
    def __init__(self, x, y, sigma, n):
        """
        A Gaussian likelihood for polynomial of degree `n`.

        Parameters
        ----------
        x, y: array_like
            The data to analyse.
        sigma: float
            The standard deviation of the noise.
        n: int
            The degree of the polynomial to fit.
        """
        self.keys = ["c{}".format(k) for k in range(n)]
        super().__init__(parameters={k: None for k in self.keys})
        self.x = x
        self.y = y
        self.sigma = sigma
        self.N = len(x)
        self.n = n

    def polynomial(self, x, parameters):
        coeffs = [parameters[k] for k in self.keys]
        return np.polyval(coeffs, x)

    def log_likelihood(self):
        res = self.y - self.polynomial(self.x, self.parameters)
        return -0.5 * (
            np.sum((res / self.sigma) ** 2)
            + self.N * np.log(2 * np.pi * self.sigma**2)
        )


def fit(n):
    likelihood = Polynomial(time, data, sigma, n)
    priors = {}
    for i in range(n):
        k = "c{}".format(i)
        priors[k] = bilby.core.prior.Uniform(0, 10, k)

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        nlive=100,
        outdir=outdir,
        label=label,
        sampler="nestle",
    )
    return (
        result.log_evidence,
        result.log_evidence_err,
        np.log(result.occam_factor(priors)),
    )


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

log_evidences = []
log_evidences_err = []
log_occam_factors = []
ns = range(1, 11)
for ll in ns:
    e, e_err, o = fit(ll)
    log_evidences.append(e)
    log_evidences_err.append(e_err)
    log_occam_factors.append(o)

ax1.errorbar(ns, log_evidences, yerr=log_evidences_err, fmt="-o", color="C0")
ax1.set_ylabel("Unnormalized log evidence", color="C0")
ax1.tick_params("y", colors="C0")

ax2.plot(ns, log_occam_factors, "-o", color="C1", alpha=0.5)
ax2.tick_params("y", colors="C1")
ax2.set_ylabel("Occam factor", color="C1")
ax1.set_xlabel("Degree of polynomial")

fig.savefig("{}/{}_test".format(outdir, label))
