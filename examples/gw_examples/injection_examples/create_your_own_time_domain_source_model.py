#!/usr/bin/env python
"""
A script to show how to create your own time domain source model.
A simple damped Gaussian signal is defined in the time domain, injected into
noise in two interferometers (LIGO Livingston and Hanford at design
sensitivity), and then recovered.
"""

import bilby
import numpy as np
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)


# define the time-domain model
def time_domain_damped_sinusoid(time, amplitude, damping_time, frequency, phase, t0):
    r"""
    This example only creates a linearly polarised signal with only plus
    polarisation.

    .. math::

        h_{\plus}(t) =
            \Theta(t - t_{0}) A
            e^{-(t - t_{0}) / \tau}
            \sin \left( 2 \pi f t + \phi \right)

    Parameters
    ----------
    time: array-like
        The times at which to evaluate the model. This is required for all
        time-domain models.
    amplitude: float
        The peak amplitude.
    damping_time: float
        The damping time of the exponential.
    frequency: float
        The frequency of the oscillations.
    phase: float
        The initial phase of the signal.
    t0: float
        The offset of the start of the signal from the start time.

    Returns
    -------
    dict:
        A dictionary containing "plus" and "cross" entries.

    """
    plus = np.zeros(len(time))
    tidx = time >= t0
    plus[tidx] = (
        amplitude
        * np.exp(-(time[tidx] - t0) / damping_time)
        * np.sin(2 * np.pi * frequency * (time[tidx] - t0) + phase)
    )
    cross = np.zeros(len(time))
    return {"plus": plus, "cross": cross}


# define parameters to inject.
injection_parameters = dict(
    amplitude=5e-22,
    damping_time=0.1,
    frequency=50,
    phase=0,
    ra=0,
    dec=0,
    psi=0,
    t0=0.0,
    geocent_time=0.0,
)

duration = 1
sampling_frequency = 1024
outdir = "outdir"
label = "time_domain_source_model"

# call the waveform_generator to create our waveform model.
waveform = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    time_domain_source_model=time_domain_damped_sinusoid,
    start_time=injection_parameters["geocent_time"] - 0.5,
)

# inject the signal into three interferometers
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 0.5,
)
ifos.inject_signal(
    waveform_generator=waveform, parameters=injection_parameters, raise_error=False
)

#  create the priors
prior = injection_parameters.copy()
prior["amplitude"] = bilby.core.prior.LogUniform(1e-23, 1e-21, r"$h_0$")
prior["damping_time"] = bilby.core.prior.Uniform(0.01, 1, r"damping time", unit="$s$")
prior["frequency"] = bilby.core.prior.Uniform(0, 200, r"frequency", unit="Hz")
prior["phase"] = bilby.core.prior.Uniform(-np.pi / 2, np.pi / 2, r"$\phi$")

# define likelihood
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(ifos, waveform)

# launch sampler
result = bilby.core.sampler.run_sampler(
    likelihood,
    prior,
    sampler="dynesty",
    npoints=500,
    walks=5,
    nact=3,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

result.plot_corner()
