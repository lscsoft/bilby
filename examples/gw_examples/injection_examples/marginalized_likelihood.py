#!/usr/bin/env python
"""
Tutorial to demonstrate how to improve the speed and efficiency of parameter
estimation on an injected signal using time, phase and distance marginalisation.

We also demonstrate how the posterior distribution for the marginalised
parameter can be recovered in post-processing.
"""
import bilby
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

duration = 4
sampling_frequency = 1024
outdir = "outdir"
label = "marginalized_likelihood"

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

waveform_arguments = dict(waveform_approximant="IMRPhenomXP", reference_frequency=50)

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
priors = bilby.gw.prior.BBHPriorDict()
priors["geocent_time"] = bilby.core.prior.Uniform(
    minimum=injection_parameters["geocent_time"] - 0.1,
    maximum=injection_parameters["geocent_time"] + 0.1,
    name="geocent_time",
    latex_label="$t_c$",
    unit="$s$",
)
# These parameters will not be sampled
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "ra",
    "dec",
    "mass_1",
    "mass_2",
]:
    priors[key] = injection_parameters[key]
del priors["chirp_mass"], priors["mass_ratio"]

# Initialise GravitationalWaveTransient
# Note that we now need to pass the: priors and flags for each thing that's
# being marginalised. A lookup table is used for distance marginalisation which
# takes a few minutes to build.
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    distance_marginalization=True,
    phase_marginalization=True,
    time_marginalization=True,
)

# Run sampler
# Note that we've added an additional argument `conversion_function`, this is
# a function that is applied to the posterior. Here it generates many additional
# parameters, e.g., source frame masses and effective spin parameters. It also
# reconstructs posterior distributions for the parameters which were
# marginalised over in the likelihood.
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
)
result.plot_corner()
