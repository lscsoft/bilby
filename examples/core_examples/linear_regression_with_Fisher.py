#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise. We then compare the result to posteriors
estimated using the Fisher Information Matrix approximation.

"""
import copy

import bilby
from bilby.core.utils.random import rng, seed

# sets seed of bilby's generator "rng" to "123"
seed(123)

import numpy as np

# A few simple setup steps
outdir = "outdir"


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict(m=0.5, c=0.2)

# For this example, we'll use standard Gaussian noise

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
sigma = rng.normal(1, 0.01, N)
data = model(time, **injection_parameters) + rng.normal(0, sigma, N)

# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data and signal model
likelihood = bilby.likelihood.GaussianLikelihood(time, data, model, sigma)

# From hereon, the syntax is exactly equivalent to other bilby examples
# We make a prior
priors = dict()
priors["m"] = bilby.core.prior.Uniform(0, 5, "m")
priors["c"] = bilby.core.prior.Uniform(-2, 2, "c")
priors = bilby.core.prior.PriorDict(priors)

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=1000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label="Nested Sampling",
)

# Finally plot a corner plot: all outputs are stored in outdir
result.plot_corner()

fim = bilby.core.fisher.FisherMatrixPosteriorEstimator(likelihood, priors)
result_fim = copy.deepcopy(result)
result_fim.posterior = fim.sample_dataframe("maxL", 10000)
result_fim.label = "Fisher"

bilby.core.result.plot_multiple(
    [result, result_fim], parameters=injection_parameters, truth_color="k"
)
