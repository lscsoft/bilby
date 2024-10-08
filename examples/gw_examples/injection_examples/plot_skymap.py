#!/usr/bin/env python
"""
Example script which produces posterior samples of ra and dec and generates a
skymap
"""
import bilby
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

duration = 4
sampling_frequency = 1024
outdir = "outdir"
label = "plot_skymap"
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
    dec=-0.2108,
)

waveform_arguments = dict(waveform_approximant="IMRPhenomXP", reference_frequency=50.0)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameters=injection_parameters,
    waveform_arguments=waveform_arguments,
)

ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

priors = bilby.gw.prior.BBHPriorDict()
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "mass_1",
    "mass_2",
    "phase",
    "geocent_time",
    "theta_jn",
]:
    priors[key] = injection_parameters[key]
del priors["chirp_mass"], priors["mass_ratio"]

likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="nestle",
    npoints=250,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    result_class=bilby.gw.result.CBCResult,
)

# make some plots of the outputs
result.plot_corner()

# will require installation of ligo.skymap (pip install ligo.skymap)
# the skymap generation code is fairly slow when using many points so limit
# ourselves to 500 points in the fit
result.plot_skymap(maxpts=500)
