#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a binary neutron star
with equation of state inference using a spectral decomposition model.

This example estimates the masses and spectral model parameters.

Details on the spectral model can be found in https://arxiv.org/pdf/1805.11217.pdf.
The principal component analysis (PCA) used for the spectral model here can be found
in the appendix of https://arxiv.org/pdf/2001.01747.pdf.
"""


import bilby
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)


# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "bns_spectral_eos_simulation"
bilby.core.utils.setup_logger(outdir=outdir, label=label)


# We are going to inject a binary neutron star waveform with component mass and lambda
# values consistent with the MPA1 equation of state.
# The lambda values are generated from LALSimNeutronStarEOS_MPA1.dat
# lalsim-ns-params -n MPA1 -m 1.523194
# lalsim-ns-params -n MPA1 -m 1.5221469
# Note that we injection with tidal params (lambdas) but recover in eos model params.
injection_parameters = dict(
    mass_1=1.523194,
    mass_2=1.5221469,
    lambda_1=311.418,
    lambda_2=312.717,
    chi_1=0.0001,
    chi_2=0.0001,
    luminosity_distance=57.628867,
    theta_jn=0.66246341,
    psi=0.18407784,
    phase=5.4800181,
    geocent_time=10.0,
    ra=3.6309322,
    dec=-0.30355747,
)

# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into. For the
# TaylorF2 waveform, we cut the signal close to the isco frequency
duration = 64
sampling_frequency = 2 * 2048
start_time = injection_parameters["geocent_time"] + 2 - duration

# Fixed arguments passed into the source model. The analysis starts at 40 Hz.
waveform_arguments = dict(
    waveform_approximant="TaylorF2",
    reference_frequency=30.0,
    minimum_frequency=30.0,
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
# These default to their design sensitivity and start at 30 Hz.
interferometers = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for interferometer in interferometers:
    interferometer.minimum_frequency = 30.0
interferometers.set_strain_data_from_zero_noise(
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
)
interferometers.inject_signal(
    parameters=injection_parameters, waveform_generator=waveform_generator
)

# Load the default prior for binary neutron stars.
# We're going to sample in chirp_mass, mass_ratio, and model parameters
# rather than mass_1, mass_2, or lamba_tilde, delta_lambda_tilde.
# BNS have aligned spins by default, if you want to allow precessing spins
# pass aligned_spin=False to the BNSPriorDict
priors = bilby.gw.prior.BNSPriorDict()
# The following are spectral decomposition model priors
# Note that we sample in 'eos_spectral_pca_gamma_*', which is a
# prior space that enables more efficient. These parameters are
# subsequently converted to the spectral decomposition model space.
# These priors are required for EOS inference
priors["eos_spectral_pca_gamma_0"] = bilby.core.prior.Uniform(
    -4.37722, 4.91227, name="pca_gamma0", latex_label="$\\gamma^{pca}_0$"
)
priors["eos_spectral_pca_gamma_1"] = bilby.core.prior.Uniform(
    -1.82240, 2.06387, name="pca_gamma1", latex_label="$\\gamma^{pca}_1$"
)
priors["eos_spectral_pca_gamma_2"] = bilby.core.prior.Uniform(
    -0.32445, 0.36469, name="pca_gamma2", latex_label="$\\gamma^{pca}_2$"
)
priors["eos_spectral_pca_gamma_3"] = bilby.core.prior.Uniform(
    -0.09529, 0.11046, name="pca_gamma3", latex_label="$\\gamma^{pca}_3$"
)
priors["eos_check"] = bilby.gw.prior.EOSCheck()
# The eos_check sets up a rejection prior for sampling.
# It should be enabled for all eos runs.

# Pinning other parameters to their injected values.
for key in [
    "luminosity_distance",
    "geocent_time",
    "chi_1",
    "chi_2",
    "psi",
    "phase",
    "ra",
    "dec",
    "theta_jn",
]:
    priors[key] = injection_parameters[key]

# Remove tidal parameters from our priors.
priors.pop("lambda_1")
priors.pop("lambda_2")

# Initialise the likelihood by passing in the interferometer data (IFOs)
# and the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator,
    time_marginalization=False,
    phase_marginalization=False,
    distance_marginalization=False,
    priors=priors,
)

# Run sampler.  This model has mostly been testing with the `dynesty` sampler.
result = bilby.run_sampler(
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=1000,
    dlogz=0.1,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

result.plot_corner()
