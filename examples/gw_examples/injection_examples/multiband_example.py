#!/usr/bin/env python
"""
Example of how to use the multi-banding method (see Morisaki, (2021) arXiv:
2104.07813) for a binary neutron star simulated signal in Gaussian noise.
"""

import bilby
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

outdir = "outdir"
label = "multibanding"


minimum_frequency = 20
reference_frequency = 100
duration = 256
sampling_frequency = 4096
approximant = "IMRPhenomD"
injection_parameters = dict(
    chirp_mass=1.2,
    mass_ratio=0.8,
    chi_1=0.0,
    chi_2=0.0,
    ra=3.44616,
    dec=-0.408084,
    luminosity_distance=200.0,
    theta_jn=0.4,
    psi=0.659,
    phase=1.3,
    geocent_time=1187008882,
)

# inject signal
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=dict(
        waveform_approximant=approximant, reference_frequency=reference_frequency
    ),
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
)
ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - duration + 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)
for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency

# make waveform generator for likelihood evaluations
search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
    waveform_arguments=dict(
        waveform_approximant=approximant, reference_frequency=reference_frequency
    ),
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
)

# make prior
priors = bilby.gw.prior.BNSPriorDict()
priors["chi_1"] = 0
priors["chi_2"] = 0
del priors["lambda_1"], priors["lambda_2"]
priors["chirp_mass"] = bilby.core.prior.Uniform(
    name="chirp_mass", minimum=1.15, maximum=1.25
)
priors["geocent_time"] = bilby.core.prior.Uniform(
    injection_parameters["geocent_time"] - 0.1,
    injection_parameters["geocent_time"] + 0.1,
    latex_label="$t_c$",
    unit="s",
)

# make multi-banded likelihood
likelihood = bilby.gw.likelihood.MBGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=search_waveform_generator,
    priors=priors,
    reference_chirp_mass=priors["chirp_mass"].minimum,
    distance_marginalization=True,
    phase_marginalization=True,
)

# sampling
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=500,
    walks=100,
    maxmcmc=5000,
    nact=5,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

# Make a corner plot.
result.plot_corner()
