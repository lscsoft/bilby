#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise

"""
import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.likelihood import GaussianLikelihood
from bilby.core.utils.random import rng, seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# A few simple setup steps
label = "linear_regression_pymc"
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

# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data and signal model
likelihood = GaussianLikelihood(time, data, model, sigma=sigma)

# From hereon, the syntax is exactly equivalent to other bilby examples
# We make a prior
priors = dict()
priors["m"] = bilby.core.prior.Uniform(0, 5, "m")
priors["c"] = bilby.core.prior.Uniform(-2, 2, "c")

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="pymc",
    injection_parameters=injection_parameters,
    outdir=outdir,
    draws=2000,
    label=label,
)
result.plot_corner()
