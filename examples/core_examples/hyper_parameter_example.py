#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for hyper params
"""
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.likelihood import GaussianLikelihood
from bilby.core.prior import Uniform
from bilby.core.result import make_pp_plot
from bilby.core.sampler import run_sampler
from bilby.core.utils import check_directory_exists_and_if_not_mkdir
from bilby.core.utils.random import rng, seed
from bilby.hyper.likelihood import HyperparameterLikelihood

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

outdir = "outdir"
check_directory_exists_and_if_not_mkdir(outdir)


# Define a model to fit to each data set
def model(x, c0, c1):
    return c0 + c1 * x


N = 10
x = np.linspace(0, 10, N)
sigma = 1
Nevents = 4
labels = ["a", "b", "c", "d"]

true_mu_c0 = 5
true_sigma_c0 = 1

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

# Make the sample sets
results = list()
for i in range(Nevents):
    c0 = rng.normal(true_mu_c0, true_sigma_c0)
    c1 = rng.uniform(-1, 1)
    injection_parameters = dict(c0=c0, c1=c1)

    data = model(x, **injection_parameters) + rng.normal(0, sigma, N)
    line = ax1.plot(x, data, "-x", label=labels[i])

    likelihood = GaussianLikelihood(x, data, model, sigma)
    priors = dict(c0=Uniform(-10, 10, "c0"), c1=Uniform(-10, 10, "c1"))

    result = run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="nestle",
        nlive=1000,
        outdir=outdir,
        verbose=False,
        label="individual_{}".format(i),
        save=False,
        injection_parameters=injection_parameters,
    )
    ax2.hist(
        result.posterior.c0,
        color=line[0].get_color(),
        density=True,
        alpha=0.5,
        label=labels[i],
    )
    results.append(result)

ax1.set_xlabel("x")
ax1.set_ylabel("y(x)")
ax1.legend()
fig1.savefig("outdir/hyper_parameter_data.png")
ax2.set_xlabel("c0")
ax2.set_ylabel("density")
ax2.legend()
fig2.savefig("outdir/hyper_parameter_combined_posteriors.png")


def hyper_prior(dataset, mu, sigma):
    return (
        np.exp(-((dataset["c0"] - mu) ** 2) / (2 * sigma**2))
        / (2 * np.pi * sigma**2) ** 0.5
    )


samples = [result.posterior for result in results]
for sample in samples:
    sample["prior"] = 1 / 20
evidences = [result.log_evidence for result in results]
hp_likelihood = HyperparameterLikelihood(
    posteriors=samples,
    hyper_prior=hyper_prior,
    log_evidences=evidences,
    max_samples=500,
)

hp_priors = dict(
    mu=Uniform(-10, 10, "mu", r"$\mu_{c0}$"),
    sigma=Uniform(0, 10, "sigma", r"$\sigma_{c0}$"),
)

# And run sampler
result = run_sampler(
    likelihood=hp_likelihood,
    priors=hp_priors,
    sampler="dynesty",
    nlive=1000,
    use_ratio=False,
    outdir=outdir,
    label="hyper_parameter",
    verbose=True,
    clean=True,
)
result.plot_corner(truth=dict(mu=true_mu_c0, sigma=true_sigma_c0))
make_pp_plot(results, filename=outdir + "/hyper_parameter_pp.png")
