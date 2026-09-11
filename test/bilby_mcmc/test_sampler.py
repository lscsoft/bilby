import os
import shutil
import unittest

import bilby
from bilby.bilby_mcmc.sampler import Bilby_MCMC, BilbyMCMCSampler
from bilby.bilby_mcmc.utils import ConvergenceInputs
from bilby.core.sampler.base_sampler import SamplerError
import numpy as np
import pandas as pd


class TestBilbyMCMCSampler(unittest.TestCase):
    def setUp(self):
        default_kwargs = Bilby_MCMC.default_kwargs
        default_kwargs["target_nsamples"] = 100
        default_kwargs["L1steps"] = 1
        self.convergence_inputs = ConvergenceInputs(
            **{key: default_kwargs[key] for key in ConvergenceInputs._fields}
        )

        self.outdir = "bilby_mcmc_sampler_test"
        if os.path.isdir(self.outdir) is False:
            os.mkdir(self.outdir)

        def model(time, m, c):
            return time * m + c
        injection_parameters = dict(m=0.5, c=0.2)
        sampling_frequency = 10
        time_duration = 10
        time = np.arange(0, time_duration, 1 / sampling_frequency)
        N = len(time)
        sigma = np.random.normal(1, 0.01, N)
        data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)
        likelihood = bilby.likelihood.GaussianLikelihood(time, data, model, sigma)

        # From hereon, the syntax is exactly equivalent to other bilby examples
        # We make a prior
        priors = dict()
        priors['m'] = bilby.core.prior.Uniform(0, 5, 'm')
        priors['c'] = bilby.core.prior.Uniform(-2, 2, 'c')
        priors = bilby.core.prior.PriorDict(priors)

        search_parameter_keys = ['m', 'c']
        use_ratio = False

        bilby.core.sampler.base_sampler.initialize_global_variables(
            likelihood,
            priors,
            search_parameter_keys,
            use_ratio,
            priors.sample(),
        )

    def tearDown(self):
        if os.path.isdir(self.outdir):
            shutil.rmtree(self.outdir)

    def test_None_proposal_cycle(self):
        with self.assertRaises(SamplerError):
            BilbyMCMCSampler(
                convergence_inputs=self.convergence_inputs,
                proposal_cycle=None,
                beta=1,
                Tindex=0,
                Eindex=0,
                use_ratio=False
            )

    def test_default_proposal_cycle(self):
        sampler = BilbyMCMCSampler(
            convergence_inputs=self.convergence_inputs,
            proposal_cycle="default_noNFnoGMnoKD",
            beta=1,
            Tindex=0,
            Eindex=0,
            use_ratio=False
        )

        nsteps = 0
        while sampler.nsamples < 500:
            sampler.step()
            nsteps += 1
        self.assertEqual(sampler.chain.position, nsteps)
        self.assertEqual(sampler.accepted + sampler.rejected, nsteps)
        self.assertTrue(isinstance(sampler.samples, pd.DataFrame))


def test_get_expected_outputs():
    label = "par0"
    outdir = os.path.join("some", "bilby_pipe", "dir")
    filenames, directories = Bilby_MCMC.get_expected_outputs(
        outdir=outdir, label=label
    )
    assert len(filenames) == 1
    assert len(directories) == 0
    assert os.path.join(outdir, f"{label}_resume.pickle") in filenames


if __name__ == "__main__":
    unittest.main()
