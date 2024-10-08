#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a full 15 parameter
space for an injected cbc signal. This is the standard injection analysis script
one can modify for the study of injected CBC events.

This will take many hours to run.
"""
import bilby
import numpy as np
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 4.0
sampling_frequency = 1024.0

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "full_15_parameters"
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
)

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomXPHM",
    reference_frequency=50.0,
    minimum_frequency=20.0,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
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

# For this analysis, we implement the standard BBH priors defined, except for
# the definition of the time prior, which is defined as uniform about the
# injected value.
# We change the mass boundaries to be more targeted for the source we
# injected.
# We define priors in the time at the Hanford interferometer and two
# parameters (zenith, azimuth) defining the sky position wrt the two
# interferometers.
priors = bilby.gw.prior.BBHPriorDict()

time_delay = ifos[0].time_delay_from_geocenter(
    injection_parameters["ra"],
    injection_parameters["dec"],
    injection_parameters["geocent_time"],
)
priors["H1_time"] = bilby.core.prior.Uniform(
    minimum=injection_parameters["geocent_time"] + time_delay - 0.1,
    maximum=injection_parameters["geocent_time"] + time_delay + 0.1,
    name="H1_time",
    latex_label="$t_H$",
    unit="$s$",
)
del priors["ra"], priors["dec"]
priors["zenith"] = bilby.core.prior.Sine(latex_label="$\\kappa$")
priors["azimuth"] = bilby.core.prior.Uniform(
    minimum=0, maximum=2 * np.pi, latex_label="$\\epsilon$", boundary="periodic"
)

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveoform generator, as well the priors.
# The explicit distance marginalization is turned on to improve
# convergence, and the posterior is recovered by the conversion function.
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    distance_marginalization=True,
    phase_marginalization=False,
    time_marginalization=False,
    reference_frame="H1L1",
    time_reference="H1",
)

# Run sampler. In this case we're going to use the `dynesty` sampler
# Note that the `walks`, `nact`, and `maxmcmc` parameter are specified
# to ensure sufficient convergence of the analysis.
# We set `npool=4` to parallelize the analysis over four cores.
# The conversion function will determine the distance posterior in post processing
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=1000,
    walks=20,
    nact=50,
    maxmcmc=2000,
    npool=4,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
)

# Plot the inferred waveform superposed on the actual data.
result.plot_waveform_posterior(n_samples=1000)

# Make a corner plot.
result.plot_corner()
