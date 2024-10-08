#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation with calibration
uncertainties marginalized over using a finite set of realizations.
"""

from copy import deepcopy

import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# Set the duration and sampling frequency of the data segment
# that we're going to create and inject the signal into.
duration = 4
sampling_frequency = 1024

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "calibration_marginalization"
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
start_time = injection_parameters["geocent_time"] - duration + 2

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
waveform_generator_rew = deepcopy(waveform_generator)

# Set up interferometers. In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).
# These default to their design sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for ifo in ifos:
    injection_parameters.update(
        {f"recalib_{ifo.name}_amplitude_{ii}": 0.0 for ii in range(10)}
    )
    injection_parameters.update(
        {f"recalib_{ifo.name}_phase_{ii}": 0.0 for ii in range(10)}
    )
    ifo.calibration_model = bilby.gw.calibration.CubicSpline(
        prefix=f"recalib_{ifo.name}_",
        minimum_frequency=ifo.minimum_frequency,
        maximum_frequency=ifo.maximum_frequency,
        n_points=10,
    )
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
)
ifos.inject_signal(
    parameters=injection_parameters, waveform_generator=waveform_generator
)
ifos_rew = deepcopy(ifos)

# Set up prior, which is a dictionary
# Here we fix the injected cbc parameters (except the distance)
# to the injected values.
priors = injection_parameters.copy()
priors["luminosity_distance"] = bilby.prior.Uniform(
    injection_parameters["luminosity_distance"] - 1000,
    injection_parameters["luminosity_distance"] + 1000,
    name="luminosity_distance",
    latex_label="$d_L$",
)
for key in injection_parameters:
    if "recalib" in key:
        priors[key] = injection_parameters[key]

# Convert to prior dictionary to replace the floats with delta function priors
priors = bilby.core.prior.PriorDict(priors)
priors_rew = deepcopy(priors)

# Initialise the likelihood by passing in the interferometer data (IFOs) and
# the waveform generator. Here we assume there is no calibration uncertainty
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator, priors=priors
)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    npoints=500,
    walks=20,
    nact=3,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

# Setting the log likelihood to actually be the log likelihood and not the log likelihood ratio...
# This is used the for reweighting
result.posterior["log_likelihood"] = (
    result.posterior["log_likelihood"] + result.log_noise_evidence
)

# Setting the priors we want on the calibration response curve parameters - as an example.
for name in ["recalib_H1_amplitude_1", "recalib_H1_amplitude_4"]:
    priors_rew[name] = bilby.prior.Gaussian(
        mu=0, sigma=0.03, name=name, latex_label=f"H1 $A_{name[-1]}$"
    )

# Setting up the calibration marginalized likelihood.
# We save the calibration response curve files into the output directory under {ifo.name}_calibration_file.h5
cal_likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos_rew,
    waveform_generator=waveform_generator_rew,
    calibration_marginalization=True,
    priors=priors_rew,
    number_of_response_curves=100,
    calibration_lookup_table={
        ifos[i].name: f"{outdir}/{ifos[i].name}_calibration_file.h5"
        for i in range(len(ifos))
    },
)

# Plot the magnitude of the curves to be used in the marginalization
plt.semilogx(
    ifos[0].frequency_array[ifos[0].frequency_mask][0:-1],
    cal_likelihood.calibration_draws[ifos[0].name][:, 0:-1].T,
)
plt.xlim(20, 1024)
plt.ylabel("Magnitude")
plt.xlabel("Frequency [Hz]")
plt.savefig(f"{outdir}/calibration_draws.pdf")
plt.clf()

# Reweight the posterior samples from a distribution with no calibration uncertainty to one with uncertainty.
# This method utilizes rejection sampling which can be inefficient at drawing samples at higher SNRs.
result_rew = bilby.core.result.reweight(
    result,
    new_likelihood=cal_likelihood,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
)

# Plot distance posterior with and without the calibration
for res, label in zip(
    [result, result_rew], ["No calibration uncertainty", "Calibration uncertainty"]
):
    plt.hist(
        res.posterior["luminosity_distance"],
        label=label,
        bins=50,
        histtype="step",
        density=True,
    )
plt.legend()
plt.xlabel("Luminosity distance [Mpc]")
plt.savefig(f"{outdir}/luminosity_distance_posterior.pdf")
plt.clf()

plt.hist(
    result_rew.posterior["recalib_index"],
    bins=np.linspace(
        0,
        cal_likelihood.number_of_response_curves - 1,
        cal_likelihood.number_of_response_curves,
    ),
    density=True,
)
plt.xlim(0, cal_likelihood.number_of_response_curves - 1)
plt.xlabel("Calibration index")
plt.savefig(f"{outdir}/calibration_index_histogram.pdf")
