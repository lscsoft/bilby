#!/usr/bin/env python
"""
This example demonstrates how to simulate some data, add an injected signal
and plot the data.
"""
from bilby.core.utils.random import seed
from bilby.gw.detector import get_empty_interferometer
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)
duration = 4
sampling_frequency = 2048.0

outdir = "outdir"
label = "example"

injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=1000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

waveform_arguments = dict(
    waveform_approximant="IMRPhenomTPHM", reference_frequency=50.0
)

waveform_generator = WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=lal_binary_black_hole,
    parameters=injection_parameters,
    waveform_arguments=waveform_arguments,
)
hf_signal = waveform_generator.frequency_domain_strain(injection_parameters)

ifo = get_empty_interferometer("H1")
ifo.set_strain_data_from_power_spectral_density(
    duration=duration, sampling_frequency=sampling_frequency
)
ifo.inject_signal(injection_polarizations=hf_signal, parameters=injection_parameters)

t0 = injection_parameters["geocent_time"]
ifo.plot_time_domain_data(
    outdir=outdir,
    label=label,
    notches=[50],
    bandpass_frequencies=(50, 200),
    start_end=(-0.5, 0.5),
    t0=t0,
)
