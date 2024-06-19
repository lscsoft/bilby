from pathlib import Path

import bilby
import george
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.prior import Uniform
from bilby.core.utils.random import rng, seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# In this example we show how we can use the `george` package within
# `bilby`. We begin by synthesizing some data and then use a simple Gaussian
# Process model to fit and interpolate the data. `bilby` implements a
# likelihood interface to the `celerite` and `george` and adds some useful
# utility functions.

# For specific use of `george`, see also the documentation at
# https://george.readthedocs.io/en/stable/, and the paper on the arxiv
# https://arxiv.org/abs/1703.09710.

# For the use of Gaussian Processes in general, Rasmussen and Williams (
# 2006) provides an in-depth introduction. The book is available for free at
# http://www.gaussianprocess.org/.

label = "gaussian_process_george_example"
outdir = "outdir"
Path(outdir).mkdir(exist_ok=True)

# Here, we set up the parameters for the data creation. The data will be a
# Gaussian Bell curve times a sine function. We also add a linear trend and
# some Gaussian white noise.

period = 6
amplitude = 10
width = 20
jitter = 1
offset = 10
slope = 0.1


def linear_function(x, a, b):
    return a * x + b


# For the data creation, we leave a gap in the middle of the time series to
# see how the Gaussian Process model can interpolate the data. We fix the


times = np.linspace(0, 40, 100)
times = np.append(times, np.linspace(60, 100, 100))
dt = times[1] - times[0]
duration = times[-1] - times[0]

ys = (
    amplitude
    * np.sin(2 * np.pi * times / period)
    * np.exp(-((times - 50) ** 2) / 2 / width**2)
    + rng.normal(scale=jitter, size=len(times))
    + linear_function(x=times, a=slope, b=offset)
)

plt.errorbar(times, ys, yerr=jitter, fmt=".k")
plt.xlabel("times")
plt.ylabel("y")
plt.savefig(f"{outdir}/{label}_data.pdf")
plt.show()


# We use a Gaussian Process kernel to model the Bell curve and sine aspects
# of the data. This specific kernel is a Matern 3/2 kernel, which will be
# actually not be a good model! Try using the other available kernels in
# `george` to see if you can achieve a better interpolation!
kernel = 2.0 * george.kernels.Matern32Kernel(metric=5.0)

# `bilby.core.likelihood.function_to_george_mean_model` takes a python
# function and transforms it into the correct class for `celerite` to use as
# a mean function. The first argument of the python function to be passed
# has to be the 'time' or 'x' variable, followed by the parameters. The
# model needs to be initialized with some values, though these do not matter
# for inference.
LinearMeanModel = bilby.core.likelihood.function_to_george_mean_model(linear_function)
mean_model = LinearMeanModel(a=0, b=0)

# Set up the likelihood. We set `yerr=1e-6` so that we can find the amount
# of white noise during the inference process. # Smaller values of `yerr`
# cause the program to break. If you know the `yerr` in your problem,
# you can pass them in as # an array.
likelihood = bilby.core.likelihood.GeorgeLikelihood(
    kernel=kernel, mean_model=mean_model, t=times, y=ys, yerr=1e-6
)

# Print the parameter names. This is useful if we have trouble figuring out
# how `celerite` applies its naming scheme.
print(likelihood.gp.parameter_names)
print(likelihood.gp.parameter_vector)

# Set up the priors. We know the name of the parameters from the print
# statement in the line before.
priors = bilby.core.prior.PriorDict()
priors["kernel:k1:log_constant"] = Uniform(
    minimum=-10, maximum=30, name="log_A", latex_label=r"$\ln A$"
)
priors["kernel:k2:metric:log_M_0_0"] = Uniform(
    minimum=-10, maximum=30, name="log_M_0_0", latex_label=r"$\ln M_{00}$"
)
priors["white_noise:value"] = Uniform(
    minimum=0, maximum=10, name="white noise", latex_label=r"$\sigma$"
)
priors["mean:a"] = Uniform(minimum=-100, maximum=100, name="a", latex_label=r"$a$")
priors["mean:b"] = Uniform(minimum=-100, maximum=100, name="b", latex_label=r"$b$")

# Run the sampling process as usual with `bilby`. The settings are such that
# the run should finish within a few minutes.
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    outdir=outdir,
    label=label,
    sampler="dynesty",
    sample="rslice",
    nlive=300,
    resume=True,
)
result.plot_corner()

# Now let's plot the data with some possible realizations of the mean model,
# and the interpolated prediction from the Gaussian Process using the
# maximum likelihood parameters.
start_time = times[0]
end_time = times[-1]


# First, we extract the inferred Gaussian white noise jitter
jitter = result.posterior.iloc[-1]["white_noise:value"]

# Next, we re-compute the Gaussian process with these y errors.
# `likelihood.gp` is an instance of the `celerite` Gaussian Process class.
# See the `celerite` documentation for a detailed explanation. We can then
# draw predicted means and variances from the `celerite` Gaussian process.
likelihood.gp.compute(times, jitter)
x = np.linspace(start_time, end_time, 5000)
pred_mean, pred_var = likelihood.gp.predict(ys, x, return_var=True)
pred_std = np.sqrt(pred_var)

# Plot the data and prediction.
color = "#ff7f0e"
plt.errorbar(times, ys, yerr=jitter, fmt=".k", capsize=0, label="Data")
plt.plot(x, pred_mean, color=color, label="Prediction")
plt.fill_between(
    x,
    pred_mean + pred_std,
    pred_mean - pred_std,
    color=color,
    alpha=0.3,
    edgecolor="none",
)

# Plot the mean model for the maximum likelihood parameters.
if isinstance(likelihood.mean_model, (float, int)):
    trend = np.ones(len(x)) * likelihood.mean_model
else:
    trend = likelihood.mean_model.get_value(x)
plt.plot(x, trend, color="green", label="Mean")

# Plot the mean model for ten other posterior samples.
samples = [result.posterior.iloc[rng.integer(len(result.posterior))] for _ in range(10)]
for sample in samples:
    likelihood.set_parameters(sample)
    if not isinstance(likelihood.mean_model, (float, int)):
        trend = likelihood.mean_model.get_value(x)
    plt.plot(x, trend, color="green", alpha=0.3)

plt.xlabel("times")
plt.ylabel("y")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"{outdir}/{label}_max_like_fit.pdf")
plt.show()
plt.clf()
