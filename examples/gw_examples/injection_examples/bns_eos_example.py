#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a binary neutron star
system taking into account tidal deformabilities with a physically motivated
model for the tidal deformabilities.

WARNING: The code is extremely slow.
"""


import bilby
from bilby.core.utils.random import seed
from bilby.gw.eos import EOSFamily, TabularEOS

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "bns_eos_example"
bilby.core.utils.setup_logger(outdir=outdir, label=label)


# We are going to inject a binary neutron star waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# aligned spins of both black holes (chi_1, chi_2), etc.

# We're also going to 'inject' the MPA1 equation of state.
# This is done by injecting masses for the two neutron-stars,
# assuming a specific equation of state, and calculating
# corresponding tidal deformability parameters from the EoS and
# masses.
mpa1_eos = TabularEOS("MPA1")
mpa1_fam = EOSFamily(mpa1_eos)

mass_1 = 1.5
mass_2 = 1.3
lambda_1 = mpa1_fam.lambda_from_mass(mass_1)
lambda_2 = mpa1_fam.lambda_from_mass(mass_2)


injection_parameters = dict(
    mass_1=mass_1,
    mass_2=mass_2,
    chi_1=0.02,
    chi_2=0.02,
    luminosity_distance=50.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    lambda_1=lambda_1,
    lambda_2=lambda_2,
)

# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into. For the
# TaylorF2 waveform, we cut the signal close to the isco frequency
duration = 32
sampling_frequency = 2048
start_time = injection_parameters["geocent_time"] + 2 - duration

# Fixed arguments passed into the source model. The analysis starts at 40 Hz.
# Note that the EoS sampling is agnostic to waveform model as long as the approximant
# can include tides.
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2_NRTidal",
    reference_frequency=50.0,
    minimum_frequency=40.0,
)

# Create the waveform_generator using a LAL Binary Neutron Star source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).
# These default to their design sensitivity and start at 40 Hz.
interferometers = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for interferometer in interferometers:
    interferometer.minimum_frequency = 40
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
)
interferometers.inject_signal(
    parameters=injection_parameters, waveform_generator=waveform_generator
)

# We're going to sample in chirp_mass, symmetric_mass_ratio, and
# specific EoS model parameters. We're using a 4-parameter
# spectrally-decomposed EoS parameterization from Lindblom (2010).
# BNS have aligned spins by default, if you want to allow precessing spins
# pass aligned_spin=False to the BNSPriorDict
priors = bilby.gw.prior.BNSPriorDict()
for key in [
    "psi",
    "geocent_time",
    "ra",
    "dec",
    "chi_1",
    "chi_2",
    "theta_jn",
    "luminosity_distance",
    "phase",
]:
    priors[key] = injection_parameters[key]
for key in ["mass_1", "mass_2", "lambda_1", "lambda_2", "mass_ratio"]:
    del priors[key]
priors["chirp_mass"] = bilby.core.prior.Gaussian(
    1.215, 0.1, name="chirp_mass", unit="$M_{\\odot}$"
)
priors["symmetric_mass_ratio"] = bilby.core.prior.Uniform(
    0.1, 0.25, name="symmetric_mass_ratio"
)
priors["eos_spectral_gamma_0"] = bilby.core.prior.Uniform(
    0.2, 2.0, name="gamma0", latex_label="$\\gamma_0"
)
priors["eos_spectral_gamma_1"] = bilby.core.prior.Uniform(
    -1.6, 1.7, name="gamma1", latex_label="$\\gamma_1"
)
priors["eos_spectral_gamma_2"] = bilby.core.prior.Uniform(
    -0.6, 0.6, name="gamma2", latex_label="$\\gamma_2"
)
priors["eos_spectral_gamma_3"] = bilby.core.prior.Uniform(
    -0.02, 0.02, name="gamma3", latex_label="$\\gamma_3"
)

# The eos_check prior imposes several hard physical constraints on samples like
# enforcing causality and monotinicity of the EoSs. In almost ever conceivable
# sampling scenario, this should be enabled.
priors["eos_check"] = bilby.gw.prior.EOSCheck()

# Initialise the likelihood by passing in the interferometer data (IFOs)
# and the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator,
)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    npoints=1000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
    resume=True,
)

result.plot_corner()
