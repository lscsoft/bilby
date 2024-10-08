#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise. This example uses a custom
likelihood function to show how it should be defined, although this
would give equivalent results as using the pre-defined 'Gaussian Likelihood'

"""

import bilby
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from bilby.core.sampler.pymc import Pymc
from bilby.core.utils.random import rng, seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# A few simple setup steps
label = "linear_regression_pymc_custom_likelihood"
outdir = "outdir"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict(m=0.5, c=0.2)

# For this example, we'll use standard Gaussian noise
sigma = 1

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_paramsters when calling the model function.
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
data = model(time, **injection_parameters) + rng.normal(0, sigma, N)

# We quickly plot the data to check it looks sensible
fig, ax = plt.subplots()
ax.plot(time, data, "o", label="data")
ax.plot(time, model(time, **injection_parameters), "--r", label="signal")
ax.set_xlabel("time")
ax.set_ylabel("y")
ax.legend()
fig.savefig("{}/{}_data.png".format(outdir, label))


# Parameter estimation: we now define a Gaussian Likelihood class relevant for
# our model.
class GaussianLikelihoodPyMC(bilby.core.likelihood.GaussianLikelihood):
    def __init__(self, x, y, sigma, func):
        """
        A general Gaussian likelihood - the parameters are inferred from the
        arguments of function

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        sigma: float
            The standard deviation of the noise
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments are
            will require a prior and will be sampled over (unless a fixed
            value is given).
        """
        super(GaussianLikelihoodPyMC, self).__init__(x=x, y=y, func=func, sigma=sigma)

    def log_likelihood(self, sampler=None):
        """
        Parameters
        ----------
        sampler: :class:`bilby.core.sampler.Pymc`
            A Sampler object must be passed containing the prior distributions
            and PyMC :class:`~pymc.Model` to use as a context manager.
            If this is not passed, the super class is called and the regular
            likelihood is evaluated.
        """

        if not isinstance(sampler, Pymc):
            return super(GaussianLikelihoodPyMC, self).log_likelihood()

        if not hasattr(sampler, "pymc_model"):
            raise AttributeError("Sampler has not PyMC model attribute")

        with sampler.pymc_model:
            mdist = sampler.pymc_priors["m"]
            cdist = sampler.pymc_priors["c"]

            mu = model(time, mdist, cdist)

            # set the likelihood distribution
            pm.Normal("likelihood", mu=mu, sigma=self.sigma, observed=self.y)


# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data and signal model
likelihood = GaussianLikelihoodPyMC(time, data, sigma, model)


# Define a custom prior for one of the parameter for use with PyMC
class PyMCUniformPrior(bilby.core.prior.Uniform):
    def __init__(self, minimum, maximum, name=None, latex_label=None):
        """
        Uniform prior with bounds (should be equivalent to bilby.prior.Uniform)
        """
        bilby.core.prior.Prior.__init__(
            self, name, latex_label, minimum=minimum, maximum=maximum
        )

    def ln_prob(self, sampler=None):
        """
        Change ln_prob method to take in a Sampler and return a PyMC
        distribution.

        If the passed argument is not a `Pymc` sampler, assume that it is a
        float or array to be passed to the superclass.
        """
        if not isinstance(sampler, Pymc):
            return super(PyMCUniformPrior, self).ln_prob(sampler)

        return pm.Uniform(self.name, lower=self.minimum, upper=self.maximum)


# From hereon, the syntax is exactly equivalent to other bilby examples
# We make a prior
priors = dict()
priors["m"] = bilby.core.prior.Uniform(0, 5, "m")
priors["c"] = PyMCUniformPrior(-2, 2, "c")

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="pymc",
    draws=1000,
    tune=1000,
    discard_tuned_samples=True,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)
result.plot_corner()
