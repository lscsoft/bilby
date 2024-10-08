#!/usr/bin/env python
"""
Tutorial to demonstrate how to specify the prior distributions used for
parameter estimation.
"""

import bilby
import numpy as np
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

duration = 4
sampling_frequency = 1024
outdir = "outdir"

injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=4000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

waveform_arguments = dict(
    waveform_approximant="IMRPhenomXPHM",
    reference_frequency=50.0,
    minimum_frequency=20.0,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

# Set up prior
# This loads in a predefined set of priors for BBHs.
priors = bilby.gw.prior.BBHPriorDict()
# These parameters will not be sampled
for key in [
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "phase",
    "theta_jn",
    "ra",
    "dec",
    "geocent_time",
    "psi",
]:
    priors[key] = injection_parameters[key]
# We can make uniform distributions.
del priors["chirp_mass"], priors["mass_ratio"]
# We can make uniform distributions.
priors["mass_1"] = bilby.core.prior.Uniform(
    name="mass_1", minimum=20, maximum=40, unit="$M_{\\odot}$"
)
priors["mass_2"] = bilby.core.prior.Uniform(
    name="mass_2", minimum=20, maximum=40, unit="$M_{\\odot}$"
)
# We can make a power-law distribution, p(x) ~ x^{alpha}
# Note: alpha=0 is a uniform distribution, alpha=-1 is uniform-in-log
priors["a_1"] = bilby.core.prior.PowerLaw(name="a_1", alpha=-1, minimum=1e-2, maximum=1)
# We can define a prior from an array as follows.
# Note: this doesn't have to be properly normalised.
a_2 = np.linspace(0, 1, 1001)
p_a_2 = a_2**4
priors["a_2"] = bilby.core.prior.Interped(
    name="a_2", xx=a_2, yy=p_a_2, minimum=0, maximum=0.5
)
# Additionally, we have Gaussian, TruncatedGaussian, Sine and Cosine.
# It's also possible to load an interpolate a prior from a file.
# Finally, if you don't specify any necessary parameters it will be filled in
# from the default when the sampler starts.
# Enjoy.

# Initialise GravitationalWaveTransient
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

# Run sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    outdir=outdir,
    injection_parameters=injection_parameters,
    label="specify_prior",
)
result.plot_corner()
