import datetime
import os
import shutil
import unittest
from copy import deepcopy
from unittest import mock

import bilby
import bilby.core.sampler.dynesty
import dynesty
import numpy as np
import parameterized
from attr import define
from scipy.stats import gamma, ks_1samp, uniform, powerlaw
from unittest.mock import Mock, patch

from bilby.core.sampler import dynesty_utils


@define
class Dummy:
    u: np.ndarray
    axes: np.ndarray
    scale: float = 1
    rseed: float = 1234
    kwargs: dict = dict(
        walks=500, live=np.zeros((2, 4)), periodic=None, reflective=None
    )
    prior_transform: callable = lambda x: x
    loglikelihood: callable = lambda x: 0
    loglstar: float = -1
    live_points: np.ndarray = np.zeros((2, 4))


class DummyLikelihood(bilby.core.likelihood.Likelihood):
    """
    A trivial likelihood used for testing. Add some randomness so the likelihood
    isn't flat everywhere as that can cause issues for nested samplers.
    """

    def __init__(self):
        super().__init__()

    def log_likelihood(self, parameters):
        return np.random.uniform(0, 0.01)


class TestDynesty(unittest.TestCase):
    def setUp(self):
        self.likelihood = DummyLikelihood()
        self.priors = bilby.core.prior.PriorDict(
            dict(a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Uniform(0, 1))
        )
        self.init_sampler()

    def init_sampler(self, **kwargs):
        self.sampler = bilby.core.sampler.dynesty.Dynesty(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
            **kwargs,
        )
        bilby.core.sampler.base_sampler.initialize_global_variables(
            self.likelihood,
            self.priors,
            self.priors.keys(),
            False,
            self.priors.sample(),
        )
        self.dysampler = self.sampler.sampler_init(
            loglikelihood=bilby.core.sampler.dynesty._log_likelihood_wrapper,
            prior_transform=bilby.core.sampler.dynesty._prior_transform_wrapper,
            ndim=self.sampler.ndim,
            **self.sampler.sampler_init_kwargs,
        )

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler
        del self.dysampler

    def read_mock_checkpoint(self, metadata):
        checkpoint = mock.MagicMock(added_live=False)
        versions = dict(bilby=bilby.__version__, dynesty=dynesty.__version__)
        self.sampler.pool = None
        with (
            mock.patch("os.path.isfile", return_value=True),
            mock.patch("os.stat", return_value=mock.MagicMock(st_size=1)),
            mock.patch("builtins.open", mock.mock_open()),
            mock.patch(
                "dill.load", return_value=(checkpoint, versions, metadata.copy())
            ),
        ):
            return self.sampler.read_saved_state(continuing=True)

    def test_write_current_state_stores_search_parameter_keys(self):
        self.sampler.sampler = self.dysampler
        self.sampler.pool = None
        with (
            mock.patch("dill.pickles", return_value=True),
            mock.patch.object(
                bilby.core.sampler.dynesty, "safe_file_dump"
            ) as safe_file_dump,
        ):
            self.sampler.write_current_state()

        checkpoint = safe_file_dump.call_args.args[0]
        assert checkpoint[2]["search_parameter_keys"] == ["a", "b"]

    def test_read_saved_state_accepts_matching_search_parameter_keys(self):
        metadata = dict(
            search_parameter_keys=["a", "b"],
            sampling_time=datetime.timedelta(),
            start_time=datetime.datetime.now(),
        )
        assert self.read_mock_checkpoint(metadata)

    def test_read_saved_state_rejects_changed_search_parameter_order(self):
        metadata = dict(
            search_parameter_keys=["b", "a"],
            sampling_time=datetime.timedelta(),
            start_time=datetime.datetime.now(),
        )
        with self.assertRaisesRegex(
            bilby.core.sampler.base_sampler.ResumeError,
            "search parameter order differs",
        ):
            self.read_mock_checkpoint(metadata)

    def test_read_saved_state_allows_legacy_checkpoint(self):
        metadata = dict(
            sampling_time=datetime.timedelta(),
            start_time=datetime.datetime.now(),
        )
        with mock.patch.object(bilby.core.sampler.dynesty.logger, "warning") as warning:
            assert self.read_mock_checkpoint(metadata)
        assert (
            "does not contain search parameter order metadata"
            in warning.call_args_list[0].args[0]
        )

    def test_default_kwargs(self):
        """Only test the kwargs where we specify different defaults to dynesty"""
        expected = dict(
            sample="act-walk",
            facc=0.2,
            save_bounds=False,
            dlogz=0.1,
            bound="live",
            update_interval=600,
        )
        for key in expected:
            self.assertEqual(expected[key], self.sampler.kwargs[key])

    def test_translate_kwargs(self):
        expected = 1000
        for equiv in bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs:
            new_kwargs = deepcopy(self.sampler.kwargs)
            del new_kwargs["nlive"]
            new_kwargs[equiv] = expected
            self.sampler._translate_kwargs(new_kwargs)
            self.assertEqual(new_kwargs["nlive"], expected)

    def test_prior_boundary(self):
        self.priors["a"] = bilby.core.prior.Prior(boundary="periodic")
        self.priors["b"] = bilby.core.prior.Prior(boundary="reflective")
        self.priors["c"] = bilby.core.prior.Prior(boundary=None)
        self.priors["d"] = bilby.core.prior.Prior(boundary="reflective")
        self.priors["e"] = bilby.core.prior.Prior(boundary="periodic")
        self.init_sampler()
        self.assertEqual(
            [0, 4], self.dysampler.internal_sampler_next.sampler_kwargs["periodic"]
        )
        self.assertEqual(
            [1, 3],
            self.dysampler.internal_sampler_next.sampler_kwargs["reflective"],
        )

    def test_sampler_kwargs_act(self):
        self.init_sampler(sample="act-walk", nact=5, maxmcmc=200)
        self.assertIsInstance(
            self.dysampler.internal_sampler_next,
            dynesty_utils.ACTTrackingEnsembleWalk,
        )
        self.assertEqual(self.dysampler.internal_sampler_next.thin, 5)
        self.assertEqual(self.dysampler.internal_sampler_next.maxmcmc, 200 * 50)

    def test_sampler_kwargs_rwalk(self):
        self.init_sampler(sample="rwalk", nact=5, maxmcmc=200)
        self.assertIsInstance(
            self.dysampler.internal_sampler_next,
            dynesty_utils.AcceptanceTrackingRWalk,
        )
        self.assertEqual(self.dysampler.internal_sampler_next.nact, 5)
        self.assertEqual(self.dysampler.internal_sampler_next.maxmcmc, 200)

    def test_sampler_kwargs_acceptance_walk(self):
        self.init_sampler(sample="acceptance-walk", naccept=5, maxmcmc=200)
        self.assertIsInstance(
            self.dysampler.internal_sampler_next, dynesty_utils.EnsembleWalkSampler
        )
        self.assertEqual(self.dysampler.internal_sampler_next.naccept, 5)
        self.assertEqual(self.dysampler.internal_sampler_next.maxmcmc, 200)

    def test_run_test_runs(self):
        self.sampler._run_test()

    def test_plot_current_state_only_closes_figures_it_creates(self):
        import matplotlib.pyplot as plt

        user_figure = plt.figure()
        self.addCleanup(plt.close, user_figure)
        user_figure_number = user_figure.number
        created_figures = []

        def make_plot(*args, **kwargs):
            self.assertEqual(set(plt.get_fignums()), {user_figure_number})
            figure = plt.figure()
            created_figures.append(figure)
            return figure, None

        trace_plot_calls = 0

        def make_trace_plot(*args, **kwargs):
            nonlocal trace_plot_calls
            trace_plot_calls += 1
            plot = make_plot()
            if trace_plot_calls == 1:
                raise ValueError("Failed after creating a figure")
            return plot

        self.sampler.sampler = mock.Mock(
            results={"samples_u": np.zeros((1, self.sampler.ndim))}
        )
        with (
            mock.patch("dynesty.plotting.traceplot", side_effect=make_trace_plot),
            mock.patch("dynesty.plotting.runplot", side_effect=make_plot),
            mock.patch(
                "dynesty.utils.results_substitute",
                side_effect=lambda results, substitutions: results,
            ),
            mock.patch.object(
                bilby.core.sampler.dynesty,
                "dynesty_stats_plot",
                side_effect=make_plot,
            ),
            mock.patch("matplotlib.figure.Figure.savefig"),
        ):
            self.sampler.plot_current_state()

        self.assertTrue(plt.fignum_exists(user_figure_number))
        self.assertTrue(
            all(not plt.fignum_exists(figure.number) for figure in created_figures)
        )

    @parameterized.parameterized.expand((
        ("unif", "single"),
        ("unif", "multi"),
        ("rslice", "single"),
        ("rslice", "multi"),
    ))
    def test_dynesty_native_methods_initialize(self, sample, bound):
        """
        Make sure that we can initialize the native dynesty samplers.
        This is not an exhaustive test.
        """
        self.init_sampler(sample=sample, bound=bound)

    def test_plotting_exception_does_not_swallow_interruption(self):
        # Emulate error reported in https://github.com/bilby-dev/bilby/issues/758
        error = SystemError(
            "numpy.ndarray.__deepcopy__ returned a result with an error set"
        )

        self.sampler.sampler = Mock()
        self.sampler._interrupted = True

        with patch("dynesty.plotting.traceplot", side_effect=error):
            with self.assertRaises(SystemExit) as context:
                self.sampler.plot_current_state()

        self.assertEqual(context.exception.code, self.sampler.exit_code)
        self.assertIs(context.exception.__cause__, error)


def test_get_expected_outputs():
    label = "par0"
    outdir = os.path.join("some", "bilby_pipe", "dir")
    filenames, directories = bilby.core.sampler.dynesty.Dynesty.get_expected_outputs(
        outdir=outdir, label=label
    )
    assert len(filenames) == 2
    assert len(directories) == 0
    assert os.path.join(outdir, f"{label}_resume.pickle") in filenames
    assert os.path.join(outdir, f"{label}_dynesty.pickle") in filenames


class ProposalsTest(unittest.TestCase):

    def test_boundaries(self):
        inputs = np.array([0.1, 1.1, -1.3])
        expected = np.array([0.1, 0.1, 0.7])
        periodic = [1]
        reflective = [2]
        self.assertLess(
            max(
                abs(
                    dynesty_utils.apply_boundaries_(inputs, periodic, reflective)
                    - expected
                )
            ),
            1e-10,
        )

    def test_boundaries_returns_none_outside_bound(self):
        inputs = np.array([0.1, 1.1, -1.3])
        self.assertIsNone(dynesty_utils.apply_boundaries_(inputs, None, None))

    def test_propose_volumetric(self):
        proposal_func = dynesty_utils.proposal_funcs["volumetric"]
        rng = np.random.default_rng(12345)
        axes = np.array([[2, 0], [0, 2]])
        start = np.zeros(4)
        new_samples = list()
        for _ in range(1000):
            new_samples.append(proposal_func(start, axes, 1, 4, 2, rng))
        new_samples = np.array(new_samples)
        self.assertGreater(
            ks_1samp(new_samples[:, 2:].flatten(), uniform(0, 1).cdf).pvalue, 0.01
        )
        self.assertGreater(
            ks_1samp(
                np.linalg.norm(new_samples[:, :2], axis=-1), powerlaw(2, scale=2).cdf
            ).pvalue,
            0.01,
        )

    def test_propose_differential_evolution_mode_hopping(self):
        proposal_func = dynesty_utils.proposal_funcs["diff"]
        rng = np.random.default_rng(12345)
        live = np.array([[1, 1], [0, 0]])
        start = np.zeros(4)
        new_samples = list()
        for _ in range(1000):
            new_samples.append(proposal_func(start, live, 4, 2, rng, mix=0))
        new_samples = np.array(new_samples)
        self.assertGreater(
            ks_1samp(new_samples[:, 2:].flatten(), uniform(0, 1).cdf).pvalue, 0.01
        )
        self.assertLess(np.max(abs(new_samples[:, :2]) - np.array([1, 1])), 1e-10)

    @parameterized.parameterized.expand(((1,), (None,), (5,)))
    def test_propose_differential_evolution(self, scale):
        proposal_func = dynesty_utils.proposal_funcs["diff"]
        rng = np.random.default_rng(12345)
        live = np.array([[1, 1], [0, 0]])
        start = np.zeros(4)
        new_samples = list()
        for _ in range(1000):
            new_samples.append(
                proposal_func(start, live, 4, 2, rng, mix=1, scale=scale)
            )
        new_samples = np.array(new_samples)
        if scale is None:
            scale = 1.17
        self.assertGreater(
            ks_1samp(new_samples[:, 2:].flatten(), uniform(0, 1).cdf).pvalue, 0.01
        )
        self.assertGreater(
            ks_1samp(
                np.abs(new_samples[:, :2].flatten()), gamma(4, scale=scale / 4).cdf
            ).pvalue,
            0.01,
        )

    def test_get_proposal_kwargs_diff(self):
        args = Dummy(u=-np.ones(4), axes=np.zeros((2, 2)), scale=4)
        args.kwargs["proposals"] = ["diff"]
        args.kwargs["ncdim"] = 2
        proposals, common, specific = dynesty_utils._get_proposal_kwargs(args)
        del common["rstate"]
        self.assertTrue(
            np.array_equal(proposals, np.array(["diff"] * args.kwargs["walks"]))
        )
        self.assertDictEqual(common, dict(n=len(args.u), n_cluster=len(args.axes)))
        assert np.array_equal(args.kwargs["live"][:1, :2], specific["diff"]["live"])
        del specific["diff"]["live"]
        self.assertDictEqual(specific, dict(diff=dict(mix=0.5, scale=1.19)))

    def test_get_proposal_kwargs_volumetric(self):
        args = Dummy(u=-np.ones(4), axes=np.zeros((2, 2)), scale=4)
        args.kwargs["proposals"] = ["volumetric"]
        args.live_points = np.zeros((2, 2))
        proposals, common, specific = dynesty_utils._get_proposal_kwargs(args)
        del common["rstate"]
        self.assertTrue(
            np.array_equal(proposals, np.array(["volumetric"] * args.kwargs["walks"]))
        )
        self.assertDictEqual(common, dict(n=len(args.u), n_cluster=len(args.axes)))
        self.assertDictEqual(
            specific, dict(volumetric=dict(axes=args.axes, scale=args.scale))
        )

    def test_proposal_functions_run_new(self):
        args = Dummy(u=np.ones(4) / 2, axes=np.ones((2, 2)))
        args.live_points[0] += 1
        args.kwargs["ncdim"] = 2
        # args.kwargs["live"][0] += 1
        for proposals in [
            ["diff"],
            ["volumetric"],
            ["diff", "volumetric"],
            {"diff": 5, "volumetric": 1},
        ]:
            args.kwargs["proposals"] = proposals
            dynesty_utils.EnsembleWalkSampler().sample(args)
            dynesty_utils.AcceptanceTrackingRWalk().sample(args)
            dynesty_utils.ACTTrackingEnsembleWalk().sample(args)


class TestEstimateNMCMC(unittest.TestCase):
    def test_converges_to_correct_value(self):
        """
        NMCMC convergence should convergence to
        safety * (2 / accept_ratio - 1)
        """
        sampler = dynesty_utils.AcceptanceTrackingRWalk()
        dynesty_utils.AcceptanceTrackingRWalk.old_act = None
        for _ in range(10):
            accept_ratio = np.random.uniform()
            safety = np.random.randint(2, 8)
            expected = safety * (2 / accept_ratio - 1)
            for _ in range(1000):
                estimated = sampler.estimate_nmcmc(
                    accept_ratio=accept_ratio,
                    safety=safety,
                    tau=1000,
                )
            self.assertAlmostEqual(estimated, expected)


class TestReproducibility(unittest.TestCase):

    @staticmethod
    def model(x, m, c):
        return m * x + c

    def setUp(self):
        bilby.core.utils.random.seed(42)
        bilby.core.utils.command_line_args.bilby_test_mode = False
        rng = bilby.core.utils.random.rng
        self.x = np.linspace(0, 1, 11)
        self.injection_parameters = dict(m=0.5, c=0.2)
        self.sigma = 0.1
        self.y = self.model(self.x, **self.injection_parameters) + rng.normal(
            0, self.sigma, len(self.x)
        )
        self.likelihood = bilby.likelihood.GaussianLikelihood(
            self.x, self.y, self.model, self.sigma
        )

        self.priors = bilby.core.prior.PriorDict()
        self.priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
        self.priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")
        # Evaluate prior once to ensure normalization constant have been set
        theta = self.priors.sample()
        self.priors.ln_prob(theta)
        self._remove_tree()
        bilby.core.utils.check_directory_exists_and_if_not_mkdir("outdir")

    def tearDown(self):
        del self.likelihood
        del self.priors
        bilby.core.utils.command_line_args.bilby_test_mode = False
        self._remove_tree()

    def _remove_tree(self):
        try:
            shutil.rmtree("outdir")
        except OSError:
            pass

    def _run_sampler(self, **kwargs):
        bilby.core.utils.random.seed(42)
        return bilby.run_sampler(
            likelihood=self.likelihood,
            priors=self.priors,
            sampler="dynesty",
            save=False,
            resume=False,
            dlogz=1.0,
            nlive=20,
            sample="acceptance-walk",
            **kwargs,
        )

    def test_reproducibility_seed(self):
        res0 = self._run_sampler(seed=1234)
        res1 = self._run_sampler(seed=1234)
        assert res0.log_evidence == res1.log_evidence

    def test_reproducibility_state(self):
        rstate = np.random.default_rng(1234)
        res0 = self._run_sampler(rstate=rstate)
        rstate = np.random.default_rng(1234)
        res1 = self._run_sampler(rstate=rstate)
        assert res0.log_evidence == res1.log_evidence

    def test_reproducibility_state_and_seed(self):
        rstate = np.random.default_rng(1234)
        res0 = self._run_sampler(rstate=rstate)
        res1 = self._run_sampler(seed=1234)
        assert res0.log_evidence == res1.log_evidence


if __name__ == "__main__":
    unittest.main()
