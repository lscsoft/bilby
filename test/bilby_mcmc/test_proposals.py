import os
import copy
import shutil
import unittest
import inspect
import importlib
import sys
import time
import bilby
from bilby.bilby_mcmc.chain import Chain, Sample
from bilby.bilby_mcmc import proposals
from bilby.bilby_mcmc.utils import LOGLKEY, LOGPKEY
import numpy as np
import pytest


class GivenProposal(proposals.BaseProposal):
    """ A simple proposal class used for testing """
    def __init__(self, priors, weight=1, subset=None, sigma=0.01):
        super(GivenProposal, self).__init__(priors, weight, subset)

    def propose(self, chain):
        log_factor = 0
        return self.given_sample, log_factor


class TestBaseProposals(unittest.TestCase):
    def create_priors(self, ndim=2, boundary=None):
        priors = bilby.core.prior.PriorDict({
            f'x{i}': bilby.core.prior.Uniform(-10, 10, name=f'x{i}', boundary=boundary)
            for i in range(ndim)
        })
        priors["fixedA"] = bilby.core.prior.DeltaFunction(1)
        priors["infinite_support"] = bilby.core.prior.Normal(0, 1)
        priors["half_infinite_support"] = bilby.core.prior.HalfNormal(1)
        return priors

    def create_random_sample(self, ndim=2):
        p = {f"x{i}": np.random.normal(0, 1) for i in range(ndim)}
        p[LOGLKEY] = np.random.normal(0, 1)
        p[LOGPKEY] = -1
        p["fixedA"] = 1
        p["infinite_support"] = np.random.normal(0, 1)
        p["half_infinite_support"] = np.abs(np.random.normal(0, 1))
        return Sample(p)

    def create_chain(self, n=1000, ndim=2):
        initial_sample = self.create_random_sample(ndim)
        chain = Chain(initial_sample=initial_sample)
        for i in range(n):
            chain.append(self.create_random_sample(ndim))
        return chain

    def test_GivenProposal(self):
        priors = self.create_priors()
        chain = self.create_chain()
        proposal = GivenProposal(priors)
        proposal.given_sample = self.create_random_sample()
        prop, _ = proposal(chain)
        self.assertEqual(prop, proposal.given_sample)

    def test_noboundary(self):
        priors = self.create_priors()
        chain = self.create_chain()
        proposal = GivenProposal(priors)

        sample = self.create_random_sample()
        sample["x0"] = priors["x0"].maximum + 0.5
        proposal.given_sample = sample

        prop, _ = proposal(chain)
        self.assertEqual(prop, proposal.given_sample)
        self.assertEqual(prop["x0"], priors["x0"].maximum + 0.5)

    def test_periodic_boundary_above(self):
        priors = self.create_priors(boundary="periodic")
        chain = self.create_chain()
        proposal = GivenProposal(priors)

        sample = self.create_random_sample()
        sample["x0"] = priors["x0"].maximum + 0.5
        proposal.given_sample = copy.deepcopy(sample)

        prop, _ = proposal(chain)
        self.assertFalse(prop["x0"] == priors["x0"].maximum + 0.5)
        self.assertEqual(prop["x0"], priors["x0"].minimum + 0.5)

    def test_periodic_boundary_below(self):
        priors = self.create_priors(boundary="periodic")
        chain = self.create_chain()
        proposal = GivenProposal(priors)

        sample = self.create_random_sample()
        sample["x0"] = priors["x0"].minimum - 0.5
        proposal.given_sample = copy.deepcopy(sample)

        prop, _ = proposal(chain)
        self.assertFalse(prop["x0"] == priors["x0"].minimum - 0.5)
        self.assertEqual(prop["x0"], priors["x0"].maximum - 0.5)


class TestProposals(TestBaseProposals):
    def setUp(self):
        self.outdir = "chain_test"
        if os.path.isdir(self.outdir) is False:
            os.mkdir(self.outdir)

    def tearDown(self):
        if os.path.isdir(self.outdir):
            shutil.rmtree(self.outdir)

    def get_simple_proposals(self):
        clsmembers = inspect.getmembers(
            sys.modules[proposals.__name__], inspect.isclass
        )
        clsmembers_clean = []
        for name, cls in clsmembers:
            a = "Proposal" in name
            b = "Base" not in name
            c = "Ensemble" not in name
            d = "Phase" not in name
            e = "Polarisation" not in name
            f = "Cycle" not in name
            g = "KDE" not in name
            h = "NormalizingFlow" not in name
            if a * b * c * d * e * f * g * h:
                clsmembers_clean.append((name, cls))

        return clsmembers_clean

    def proposal_check(self, prop, ndim=2, N=100):
        chain = self.create_chain(ndim=ndim)
        if getattr(prop, 'needs_likelihood_and_priors', False):
            return

        print(f"Testing {prop.__class__.__name__}")
        # Timing and return type
        start = time.time()
        for _ in range(N):
            p, w = prop(chain)
            chain.append(p)
        dt = 1e3 * (time.time() - start) / N
        print(f"Testing {prop.__class__.__name__}: dt~{dt:0.2g} [ms]")

        self.assertTrue(isinstance(p, Sample))
        self.assertTrue(isinstance(w, (int, float)))

    def test_proposal_return_type(self):
        priors = self.create_priors()
        for name, cls in self.get_simple_proposals():
            prop = cls(priors)
            self.proposal_check(prop)

    def test_KDE_proposal(self):
        priors = self.create_priors()
        prop = proposals.KDEProposal(priors)
        self.proposal_check(prop, N=20000)

    def test_GMM_proposal(self):
        if importlib.util.find_spec("sklearn") is not None:
            priors = self.create_priors()
            prop = proposals.GMMProposal(priors)
            self.proposal_check(prop, N=20000)
            self.assertTrue(prop.trained)
        else:
            print("Unable to test GMM as sklearn is not installed")

    @pytest.mark.requires("glasflow")
    def test_NF_proposal(self):
        priors = self.create_priors()
        chain = self.create_chain(10000)
        prop = proposals.NormalizingFlowProposal(priors, first_fit=10000)
        prop.steps_since_refit = 9999
        start = time.time()
        p, w = prop(chain)
        dt = time.time() - start
        print(f"Training for {prop.__class__.__name__} took dt~{dt:0.2g} [s]")
        self.assertTrue(prop.trained)
        self.proposal_check(prop)

    @pytest.mark.requires("glasflow")
    def test_NF_proposal_15D(self):
        ndim = 15
        priors = self.create_priors(ndim)
        chain = self.create_chain(10000, ndim=ndim)
        prop = proposals.NormalizingFlowProposal(priors, first_fit=10000)
        prop.steps_since_refit = 9999
        start = time.time()
        p, w = prop(chain)
        dt = time.time() - start
        print(f"Training for {prop.__class__.__name__} took dt~{dt:0.2g} [s]")
        self.assertTrue(prop.trained)
        self.proposal_check(prop, ndim=ndim)


if __name__ == "__main__":
    unittest.main()
