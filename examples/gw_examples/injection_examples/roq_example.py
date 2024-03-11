#!/usr/bin/env python
"""
Example of how to use the Reduced Order Quadrature method (see Smith et al.,
(2016) Phys. Rev. D 94, 044031) for a Binary Black hole simulated signal in
Gaussian noise.

This requires files specifying the appropriate basis weights.
These aren't shipped with Bilby, but are available on LDG clusters and
from the public repository https://git.ligo.org/lscsoft/ROQ_data.

We also reweight the result using the regular waveform model to check how
correct the ROQ result is.
"""

import bilby
import numpy as np
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

outdir = "outdir"
label = "roq"

# The ROQ bases can be given an overall scaling.
# Note how this is applied to durations, frequencies and masses.
scale_factor = 1.6

# Load in the pieces for the linear part of the ROQ. Note you will need to
# adjust the filenames here to the correct paths on your machine
basis_matrix_linear = np.load("B_linear.npy").T
freq_nodes_linear = np.load("fnodes_linear.npy") * scale_factor

# Load in the pieces for the quadratic part of the ROQ
basis_matrix_quadratic = np.load("B_quadratic.npy").T
freq_nodes_quadratic = np.load("fnodes_quadratic.npy") * scale_factor

# Load the parameters describing the valid parameters for the basis.
params = np.genfromtxt("params.dat", names=True)

# Get scaled ROQ quantities
minimum_chirp_mass = params["chirpmassmin"] / scale_factor
maximum_chirp_mass = params["chirpmassmax"] / scale_factor
minimum_component_mass = params["compmin"] / scale_factor


duration = 4 / scale_factor
sampling_frequency = 2048 * scale_factor

injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=1000.0,
    theta_jn=0.4,
    psi=0.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2", reference_frequency=20.0 * scale_factor
)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
)

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2 / scale_factor,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)
for ifo in ifos:
    ifo.minimum_frequency = 20 * scale_factor

# make ROQ waveform generator
search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.binary_black_hole_roq,
    waveform_arguments=dict(
        frequency_nodes_linear=freq_nodes_linear,
        frequency_nodes_quadratic=freq_nodes_quadratic,
        reference_frequency=20.0 * scale_factor,
        waveform_approximant="IMRPhenomPv2",
    ),
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
)

# Here we add constraints on chirp mass and mass ratio to the prior, these are
# determined by the domain of validity of the ROQ basis.
priors = bilby.gw.prior.BBHPriorDict()
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "theta_jn",
    "phase",
    "psi",
    "ra",
    "dec",
    "phi_12",
    "phi_jl",
    "luminosity_distance",
]:
    priors[key] = injection_parameters[key]
for key in ["mass_1", "mass_2"]:
    priors[key].minimum = max(priors[key].minimum, minimum_component_mass)
priors["chirp_mass"] = bilby.core.prior.Uniform(
    name="chirp_mass",
    minimum=float(minimum_chirp_mass),
    maximum=float(maximum_chirp_mass),
)
# The roq parameters typically store the mass ratio bounds as m1/m2 not m2/m1 as in the
# Bilby convention.
priors["mass_ratio"] = bilby.core.prior.Uniform(
    1 / params["qmax"], 1, name="mass_ratio"
)
priors["geocent_time"] = bilby.core.prior.Uniform(
    injection_parameters["geocent_time"] - 0.1,
    injection_parameters["geocent_time"] + 0.1,
    latex_label="$t_c$",
    unit="s",
)

likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=search_waveform_generator,
    linear_matrix=basis_matrix_linear,
    quadratic_matrix=basis_matrix_quadratic,
    priors=priors,
    roq_params=params,
    roq_scale_factor=scale_factor,
)

# write the weights to file so they can be loaded multiple times
likelihood.save_weights("weights.npz")

# remove the basis matrices as these are big for longer bases
del basis_matrix_linear, basis_matrix_quadratic

# load the weights from the file
likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=search_waveform_generator,
    weights="weights.npz",
    priors=priors,
)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    npoints=500,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

# Resample the result using the full waveform model with the FakeSampler.
# This will give us an idea of how good a job the ROQ does.
full_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)
resampled_result = bilby.run_sampler(
    likelihood=full_likelihood,
    priors=priors,
    sampler="fake_sampler",
    label="roq_resampled",
    outdir=outdir,
)

# Make a comparison corner plot with the two likelihoods.
bilby.core.result.plot_multiple([result, resampled_result], labels=["ROQ", "Regular"])
