#!/usr/bin/env python
"""
Demonstrate using the FakeSampler to reweight a result to include higher-order
emission modes. This is a simplified version of the method presented in
arXiv:1905.05477, however, the method can be applied to a much wider range of
initial and more complex likelihoods.
"""

import bilby
import matplotlib.pyplot as plt
from bilby.core.utils.random import seed

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)


def make_comparison_histograms(result_1, result_2):
    pos_full = result_1.posterior
    pos_simple = result_2.posterior

    plt.figure()
    plt.hist(
        pos_full["log_likelihood"],
        50,
        label=result_1.label,
        histtype="step",
        density=True,
    )
    plt.hist(
        pos_simple["log_likelihood"],
        50,
        label=result_2.label,
        histtype="step",
        density=True,
    )
    plt.xlabel(r"delta_logl")
    plt.legend(loc=2)
    plt.savefig(f"{result_1.outdir}/delta_logl.pdf")
    plt.close()


def main():
    outdir = "outdir"

    duration = 4
    sampling_frequency = 1024

    injection_parameters = dict(
        chirp_mass=36.0,
        mass_ratio=0.2,
        chi_1=0.4,
        chi_2=0.3,
        luminosity_distance=2000.0,
        theta_jn=0.4,
        psi=0.659,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
    )

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXAS",
        reference_frequency=20.0,
        minimum_frequency=20.0,
    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    )

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])

    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )

    ifos.inject_signal(
        waveform_generator=waveform_generator,
        parameters=injection_parameters,
    )

    priors = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    for key in [
        "luminosity_distance",
        "theta_jn",
        "phase",
        "psi",
        "ra",
        "dec",
        "geocent_time",
    ]:
        priors[key] = injection_parameters[key]
    priors["chirp_mass"].maximum = 45

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator
    )

    # perform the initial sampling with our simple model
    original_result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        walks=10,
        nact=3,
        bound="single",
        injection_parameters=injection_parameters,
        outdir=outdir,
        label="primary_mode_only",
        save="hdf5",
    )

    # update the waveform generator to use our higher-order mode waveform
    likelihood.waveform_generator.waveform_arguments[
        "waveform_approximant"
    ] = "IMRPhenomXHM"

    # call the FakeSampler to compute the new likelihoods
    new_result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="fake_sampler",
        sample_file=f"{outdir}/{original_result.label}_result.hdf5",
        injection_parameters=injection_parameters,
        outdir=outdir,
        verbose=False,
        label="higher_order_mode",
        save="hdf5",
    )

    # make some comparison plots
    bilby.core.result.plot_multiple([original_result, new_result])
    make_comparison_histograms(new_result, original_result)


if __name__ == "__main__":
    main()
