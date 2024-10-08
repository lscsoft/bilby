#!/usr/bin/env python
"""
A demonstration of a simple model with more than just the two polarization modes
allowed in general relativity.

We adapt the sine-Gaussian burst model to include vector polarizations with an
unknown contribution from the vector modes.
"""
import bilby
import numpy as np
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)


def vector_tensor_sine_gaussian(frequency_array, hrss, Q, frequency, epsilon):
    """
    Vector-Tensor sine-Gaussian burst model

    This is just like the bilby.gw.source.sinegaussian function but adds a
    vector-polarized component.

    Parameters
    ----------
    frequency_array: array-like
        Frequency array on which to calculate the waveform.
    hrss: float
    Q: float
    frequency: float
    epsilon: float
        Relative size of the vector modes compared to the tensor modes.
    """
    waveform_polarizations = bilby.gw.source.sinegaussian(
        frequency_array, hrss, Q, frequency
    )

    waveform_polarizations["x"] = epsilon * waveform_polarizations["plus"]
    waveform_polarizations["y"] = epsilon * waveform_polarizations["cross"]
    return waveform_polarizations


duration = 1
sampling_frequency = 512

outdir = "outdir"
label = "vector_tensor"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

injection_parameters = dict(
    hrss=1e-22,
    Q=5.0,
    frequency=200.0,
    ra=1.375,
    dec=-1.2108,
    geocent_time=1126259642.413,
    psi=2.659,
    epsilon=0.2,
)

waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=vector_tensor_sine_gaussian,
)

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
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

priors = bilby.core.prior.PriorDict()
for key in ["psi", "geocent_time", "hrss", "Q", "frequency"]:
    priors[key] = injection_parameters[key]
priors["ra"] = bilby.core.prior.Uniform(0, 2 * np.pi, latex_label="$\\alpha$")
priors["dec"] = bilby.core.prior.Cosine(latex_label="$\\delta$")
priors["epsilon"] = bilby.core.prior.Uniform(0, 1, latex_label="$\\epsilon$")

vector_tensor_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

# Run sampler.  In this case we're going to use the `dynesty` sampler
vector_tensor_result = bilby.core.sampler.run_sampler(
    likelihood=vector_tensor_likelihood,
    priors=priors,
    sampler="nestle",
    nlive=1000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label="vector_tensor",
)

vector_tensor_result.plot_corner()

tensor_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

priors["epsilon"] = 0

# Run sampler.  In this case we're going to use the `nestle` sampler
tensor_result = bilby.core.sampler.run_sampler(
    likelihood=tensor_likelihood,
    priors=priors,
    sampler="nestle",
    nlive=1000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label="tensor",
)

# make some plots of the outputs
tensor_result.plot_corner()

bilby.result.plot_multiple(
    [tensor_result, vector_tensor_result],
    labels=["Tensor", "Vector + Tensor"],
    parameters=dict(ra=1.375, dec=-1.2108),
    evidences=True,
)
