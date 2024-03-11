#!/usr/bin/env python
"""
A script to demonstrate how to use your own source model
"""
import bilby
import numpy as np
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# First set up logging and some output directories and labels
outdir = "outdir"
label = "create_your_own_source_model"
sampling_frequency = 4096
duration = 1


# Here we define out source model - this is the sine-Gaussian model in the
# frequency domain.
def gaussian(frequency_array, amplitude, f0, tau, phi0):
    r"""
    Our custom source model, this is just a Gaussian in frequency with
    variable global phase.

    .. math::

        \tilde{h}_{\plus}(f) = \frac{A \tau}{2\sqrt{\pi}}}
        e^{- \pi \tau (f - f_{0})^2 + i \phi_{0}} \\
        \tilde{h}_{\times}(f) = \tilde{h}_{\plus}(f) e^{i \pi / 2}


    Parameters
    ----------
    frequency_array: array-like
        The frequencies to evaluate the model at. This is required for all
        Bilby source models.
    amplitude: float
        An overall amplitude prefactor.
    f0: float
        The central frequency.
    tau: float
        The damping rate.
    phi0: float
        The reference phase.

    Returns
    -------
    dict:
        A dictionary containing "plus" and "cross" entries.

    """
    arg = -((np.pi * tau * (frequency_array - f0)) ** 2) + 1j * phi0
    plus = np.sqrt(np.pi) * amplitude * tau * np.exp(arg) / 2.0
    cross = plus * np.exp(1j * np.pi / 2)
    return {"plus": plus, "cross": cross}


# We now define some parameters that we will inject
injection_parameters = dict(
    amplitude=1e-23, f0=100, tau=1, phi0=0, geocent_time=0, ra=0, dec=0, psi=0
)

# Now we pass our source function to the WaveformGenerator
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=gaussian,
)

# Set up interferometers.
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 0.5,
)
ifos.inject_signal(
    waveform_generator=waveform_generator,
    parameters=injection_parameters,
    raise_error=False,
)

# Here we define the priors for the search. We use the injection parameters
# except for the amplitude, f0, and geocent_time
prior = injection_parameters.copy()
prior["amplitude"] = bilby.core.prior.LogUniform(
    minimum=1e-25, maximum=1e-21, latex_label="$\\mathcal{A}$"
)
prior["f0"] = bilby.core.prior.Uniform(90, 110, latex_label="$f_{0}$")

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

result = bilby.core.sampler.run_sampler(
    likelihood,
    prior,
    sampler="dynesty",
    outdir=outdir,
    label=label,
    resume=False,
    sample="unif",
    injection_parameters=injection_parameters,
)
result.plot_corner()
