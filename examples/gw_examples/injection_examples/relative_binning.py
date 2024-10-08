#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal, using the relative binning likelihood.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.
"""
from copy import deepcopy

import bilby
import numpy as np
from bilby.core.utils.random import rng, seed
from tqdm.auto import trange

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "relative"
bilby.core.utils.setup_logger(outdir=outdir, label=label)


# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=2000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    fiducial=1,
)

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomXP",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    # frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_relative_binning,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

# Set up a PriorDict, which inherits from dict.
# By default we will sample all terms in the signal models.  However, this will
# take a long time for the calculation, so for this example we will set almost
# all of the priors to be equall to their injected values.  This implies the
# prior is a delta function at the true, injected value.  In reality, the
# sampler implementation is smart enough to not sample any parameter that has
# a delta-function prior.
# The above list does *not* include mass_1, mass_2, theta_jn and luminosity
# distance, which means those are the parameters that will be included in the
# sampler.  If we do nothing, then the default priors get used.
priors = bilby.gw.prior.BBHPriorDict()
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

# Perform a check that the prior does not extend to a parameter space longer than the data
priors.validate_prior(duration, minimum_frequency)

# Set up the fiducial parameters for the relative binning likelihood to be the
# injected parameters. Note that because we sample in chirp mass and mass ratio
# but injected with mass_1 and mass_2, we need to convert the mass parameters
fiducial_parameters = injection_parameters.copy()
m1 = fiducial_parameters.pop("mass_1")
m2 = fiducial_parameters.pop("mass_2")
fiducial_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(
    m1, m2
)
fiducial_parameters["mass_ratio"] = m2 / m1

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    distance_marginalization=True,
    fiducial_parameters=fiducial_parameters,
)

# Run sampler.  In this case, we're going to use the `nestle` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="nestle",
    npoints=1000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
)

alt_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    # frequency_domain_source_model=lal_binary_black_hole_relative_binning,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)
alt_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=alt_waveform_generator,
)
likelihood.distance_marginalization = False
weights = list()
for ii in trange(len(result.posterior)):
    parameters = dict(result.posterior.iloc[ii])
    likelihood.parameters.update(parameters)
    alt_likelihood.parameters.update(parameters)
    weights.append(
        alt_likelihood.log_likelihood_ratio() - likelihood.log_likelihood_ratio()
    )
weights = np.exp(weights)
print(
    f"Reweighting efficiency is {np.mean(weights)**2 / np.mean(weights**2) * 100:.2f}%"
)
print(f"Binned vs unbinned log Bayes factor {np.log(np.mean(weights)):.2f}")

# Generate result object with the posterior for the regular likelihood using
# rejection sampling
alt_result = deepcopy(result)
keep = weights > rng.uniform(0, max(weights), len(weights))
alt_result.posterior = result.posterior.iloc[keep]

# Make a comparison corner plot.
bilby.core.result.plot_multiple(
    [result, alt_result],
    labels=["Binned", "Reweighted"],
    filename=f"{outdir}/{label}_corner.png",
)
