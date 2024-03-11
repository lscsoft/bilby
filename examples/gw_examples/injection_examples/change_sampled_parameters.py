#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation sampling in non-standard
parameters for an injected signal.

This example estimates the masses using a uniform prior in chirp mass,
mass ratio and redshift.
The cosmology is according to the Planck 2015 data release.
"""
import bilby
import numpy as np
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

bilby.core.utils.setup_logger(log_level="info")

duration = 4
sampling_frequency = 2048
outdir = "outdir"
label = "different_parameters"


injection_parameters = dict(
    total_mass=66.0,
    mass_ratio=0.9,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=2000,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

waveform_arguments = dict(waveform_approximant="IMRPhenomXP", reference_frequency=50.0)

# Create the waveform_generator using a LAL BinaryBlackHole source function
# We specify a function which transforms a dictionary of parameters into the
# appropriate parameters for the source model.
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=sampling_frequency,
    duration=duration,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.
ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1", "K1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

# Set up prior
# Note it is possible to sample in different parameters to those that were
# injected.
priors = bilby.gw.prior.BBHPriorDict()

del priors["mass_ratio"]
priors["symmetric_mass_ratio"] = bilby.prior.Uniform(
    name="symmetric_mass_ratio", latex_label="q", minimum=0.1, maximum=0.25
)

del priors["luminosity_distance"]
priors["redshift"] = bilby.prior.Uniform(
    name="redshift", latex_label="$z$", minimum=0, maximum=0.5
)
# These parameters will not be sampled
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
]:
    priors[key] = injection_parameters[key]
del priors["theta_jn"]
priors["cos_theta_jn"] = np.cos(injection_parameters["theta_jn"])
print(priors)

# Initialise GravitationalWaveTransient
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

# Run sampler
# Note we've added a post-processing conversion function, this will generate
# many useful additional parameters, e.g., source-frame masses.
result = bilby.core.sampler.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    walks=25,
    nact=5,
    outdir=outdir,
    injection_parameters=injection_parameters,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
)
result.plot_corner()
