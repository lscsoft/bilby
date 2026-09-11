import os
import unittest

import numpy as np
import pytest
from past.builtins import execfile

import bilby


pytestmark = pytest.mark.gw


class Test(unittest.TestCase):
    outdir = "./outdir"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(dir_path, os.path.pardir))

    # Run a script to produce standard data
    msd = {}  # A dictionary of variables saved in make_standard_data.py
    execfile(dir_path + "/integration/make_standard_data.py", msd)
    """
    @classmethod
    def setUpClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning(
                    "{} not removed prior to tests".format(self.outdir))

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning(
                    "{} not removed prior to tests".format(self.outdir))
    """

    def test_make_standard_data(self):
        " Load in the saved standard data and compare with new data "

        # Load in the saved standard data
        frequencies_saved, hf_real_saved, hf_imag_saved = np.loadtxt(
            self.dir_path + "/integration/standard_data.txt"
        ).T
        hf_signal_and_noise_saved = hf_real_saved + 1j * hf_imag_saved

        self.assertTrue(np.array_equal(self.msd["frequencies"], frequencies_saved))
        self.assertAlmostEqual(
            all(self.msd["hf_signal_and_noise"] - hf_signal_and_noise_saved),
            0.00000000,
            5,
        )

    def test_recover_luminosity_distance(self):
        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            [self.msd["IFO"]], self.msd["waveform_generator"]
        )

        priors = {}
        for key in self.msd["simulation_parameters"]:
            priors[key] = self.msd["simulation_parameters"][key]

        dL = self.msd["simulation_parameters"]["luminosity_distance"]
        priors["luminosity_distance"] = bilby.core.prior.Uniform(
            name="luminosity_distance", minimum=dL - 10, maximum=dL + 10
        )

        result = bilby.core.sampler.run_sampler(
            likelihood, priors, sampler="dynesty", verbose=False, npoints=100
        )
        self.assertAlmostEqual(
            np.mean(result.posterior.luminosity_distance),
            dL,
            delta=3 * np.std(result.posterior.luminosity_distance),
        )


if __name__ == "__main__":
    unittest.main()
