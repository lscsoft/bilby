#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation with calibration
uncertainties included.

We set up the full problem as is required and then just sample over a small
number of calibration parameters.

We demonstrate, two formulations of the calibration model:
- a cubic spline described by gaussian distributions at a set of nodes.
- a set of precomputed curves, in this example we use cubic spline realizations,
  however, it also applies to physically motivated models.
"""

import bilby
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# Set the duration and sampling frequency of the data segment
# that we're going to create and inject the signal into.

duration = 4
sampling_frequency = 1024

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "calibration"
bilby.core.utils.setup_logger(outdir=outdir, label=label)


# We are going to inject a binary black hole waveform. We first establish a
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
waveform_arguments = dict(waveform_approximant="IMRPhenomXP", reference_frequency=50.0)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameters=injection_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers. In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).
# These default to their design sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration
)

ifo = ifos[0]
injection_parameters.update(
    {f"recalib_{ifo.name}_amplitude_{ii}": 0.1 for ii in range(5)}
)
injection_parameters.update({f"recalib_{ifo.name}_phase_{ii}": 0.01 for ii in range(5)})
ifo.calibration_model = bilby.gw.calibration.CubicSpline(
    prefix=f"recalib_{ifo.name}_",
    minimum_frequency=ifo.minimum_frequency,
    maximum_frequency=ifo.maximum_frequency,
    n_points=5,
)
ifo = ifos[1]
injection_parameters["recalib_index_L1"] = 3
ifo.calibration_model = bilby.gw.calibration.Precomputed.constant_uncertainty_spline(
    amplitude_sigma=0.1,
    phase_sigma=0.01,
    label="L1",
    frequency_array=ifo.frequency_array[ifo.frequency_mask],
    n_nodes=5,
    n_curves=100,
)

ifos.inject_signal(
    parameters=injection_parameters, waveform_generator=waveform_generator
)

# Set up prior, which is a dictionary
# Here we fix the injected cbc parameters and most of the calibration parameters
# to the injected values.
# We allow a subset of the calibration parameters to vary.
priors = injection_parameters.copy()
for key in injection_parameters:
    if "recalib" in key:
        priors[key] = injection_parameters[key]
for name in ["recalib_H1_amplitude_0", "recalib_H1_amplitude_1"]:
    priors[name] = bilby.core.prior.Gaussian(
        mu=0, sigma=0.2, name=name, latex_label=f"H1 $A_{name[-1]}$"
    )
priors["recalib_index_L1"] = bilby.core.prior.Categorical(
    ncategories=100, latex_label="recalib index L1"
)

# Initialise the likelihood by passing in the interferometer data (IFOs) and
# the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
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
)

# make some plots of the outputs
result.plot_corner()
