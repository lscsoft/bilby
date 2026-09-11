import shutil
import os
import logging
from packaging import version

import unittest
import bilby
import pytest
import scipy
from scipy.stats import ks_2samp, kstest


pytestmark = pytest.mark.gw


def ks_2samp_wrapper(data1, data2):
    if version.parse(scipy.__version__) >= version.parse("1.3.0"):
        return ks_2samp(data1, data2, alternative="two-sided", mode="asymp")
    else:
        return ks_2samp(data1, data2)


class Test(unittest.TestCase):
    outdir = "outdir_for_tests"

    @classmethod
    def setUpClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(self.outdir))

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(self.outdir))

    def test_fifteen_dimensional_cbc(self):
        duration = 4.0
        sampling_frequency = 2048.0
        label = "full_15_parameters"
        bilby.core.utils.random.seed(8817021)

        waveform_arguments = dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=50.0,
            minimum_frequency=20.0,
        )
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
        )

        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration, start_time=0
        )

        priors = bilby.gw.prior.BBHPriorDict()
        priors.pop("mass_1")
        priors.pop("mass_2")
        priors["chirp_mass"] = bilby.prior.Uniform(
            name="chirp_mass",
            latex_label="$M$",
            minimum=10.0,
            maximum=100.0,
            unit="$M_{\\odot}$",
        )
        priors["mass_ratio"] = bilby.prior.Uniform(
            name="mass_ratio", latex_label="$q$", minimum=0.5, maximum=1.0
        )
        priors["geocent_time"] = bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)

        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=waveform_generator,
            priors=priors,
            distance_marginalization=False,
            phase_marginalization=False,
            time_marginalization=False,
        )

        likelihood = bilby.core.likelihood.ZeroLikelihood(likelihood)

        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="dynesty",
            nlive=1000,
            nact=10,
            outdir=self.outdir,
            label=label,
            save=False
        )
        pvalues = [
            ks_2samp_wrapper(
                result.priors[key].sample(10000), result.posterior[key].values
            ).pvalue
            for key in priors.keys()
        ]
        print("P values per parameter")
        for key, p in zip(priors.keys(), pvalues):
            print(key, p)
        self.assertGreater(kstest(pvalues, "uniform").pvalue, 0.01)


if __name__ == "__main__":
    unittest.main()
