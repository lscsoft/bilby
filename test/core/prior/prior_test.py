import os
import unittest
from copy import deepcopy

import array_api_compat as aac
import bilby
import numpy as np
import pytest
import scipy.stats as ss
from scipy.integrate import trapezoid


aligned_prior_complex = bilby.gw.prior.AlignedSpin(
    a_prior=bilby.core.prior.Beta(alpha=2.0, beta=2.0),
    z_prior=bilby.core.prior.Beta(alpha=2.0, beta=2.0, minimum=-1),
    name="test",
    unit="unit",
    num_interp=1000,
)

hp_map_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "prior_files/GW150914_testing_skymap.fits",
)
hp_dist = bilby.gw.prior.HealPixMapPriorDist(
    hp_map_file, names=["testra", "testdec"]
)
hp_3d_dist = bilby.gw.prior.HealPixMapPriorDist(
    hp_map_file, names=["testra", "testdec", "testdistance"], distance=True
)


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestPriorClasses(unittest.TestCase):
    def setUp(self):
        # set multivariate Gaussian
        mvg = bilby.core.prior.MultivariateGaussianDist(
            names=["testa", "testb"],
            mus=[1, 1],
            covs=np.array([[2.0, 0.5], [0.5, 2.0]]),
            weights=1.0,
        )
        mvn = bilby.core.prior.MultivariateGaussianDist(
            names=["testa", "testb"],
            mus=[1, 1],
            covs=np.array([[2.0, 0.5], [0.5, 2.0]]),
            weights=1.0,
        )

        # need to reset this for the repr test to get equality correct
        hp_dist.requested_parameters = {"testra": None, "testdec": None}
        hp_3d_dist.requested_parameters = {"testra": None, "testdec": None, "testdistance": None}

        def condition_func(reference_params, test_param):
            return reference_params.copy()

        self.priors = [
            bilby.core.prior.DeltaFunction(name="test", unit="unit", peak=1),
            bilby.core.prior.Gaussian(name="test", unit="unit", mu=0, sigma=1),
            bilby.core.prior.Normal(name="test", unit="unit", mu=0, sigma=1),
            bilby.core.prior.PowerLaw(
                name="test", unit="unit", alpha=0, minimum=0, maximum=1
            ),
            bilby.core.prior.PowerLaw(
                name="test", unit="unit", alpha=-1, minimum=0.5, maximum=1
            ),
            bilby.core.prior.PowerLaw(
                name="test", unit="unit", alpha=2, minimum=1, maximum=1e2
            ),
            bilby.core.prior.Uniform(name="test", unit="unit", minimum=0, maximum=1),
            bilby.core.prior.LogUniform(
                name="test", unit="unit", minimum=5e0, maximum=1e2
            ),
            bilby.gw.prior.UniformComovingVolume(
                name="redshift", minimum=0.1, maximum=1.0
            ),
            bilby.gw.prior.UniformSourceFrame(
                name="redshift", minimum=0.1, maximum=1.0
            ),
            bilby.core.prior.Sine(name="test", unit="unit"),
            bilby.core.prior.Cosine(name="test", unit="unit"),
            bilby.core.prior.Interped(
                name="test",
                unit="unit",
                xx=np.linspace(0, 10, 1000),
                yy=np.linspace(0, 10, 1000) ** 4,
                minimum=3,
                maximum=5,
            ),
            bilby.core.prior.TruncatedGaussian(
                name="test", unit="unit", mu=1, sigma=0.4, minimum=-1, maximum=1
            ),
            bilby.core.prior.TruncatedNormal(
                name="test", unit="unit", mu=1, sigma=0.4, minimum=-1, maximum=1
            ),
            bilby.core.prior.HalfGaussian(name="test", unit="unit", sigma=1),
            bilby.core.prior.HalfNormal(name="test", unit="unit", sigma=1),
            bilby.core.prior.LogGaussian(name="test", unit="unit", mu=0, sigma=1),
            bilby.core.prior.LogNormal(name="test", unit="unit", mu=0, sigma=1),
            bilby.core.prior.Exponential(name="test", unit="unit", mu=1),
            bilby.core.prior.StudentT(name="test", unit="unit", df=3, mu=0, scale=1),
            bilby.core.prior.Beta(name="test", unit="unit", alpha=2.0, beta=2.0),
            bilby.core.prior.Logistic(name="test", unit="unit", mu=0, scale=1),
            bilby.core.prior.Cauchy(name="test", unit="unit", alpha=0, beta=1),
            bilby.core.prior.Lorentzian(name="test", unit="unit", alpha=0, beta=1),
            bilby.core.prior.Gamma(name="test", unit="unit", k=1, theta=1),
            bilby.core.prior.ChiSquared(name="test", unit="unit", nu=2),
            bilby.core.prior.FermiDirac(name="test", unit="unit", mu=1, sigma=1),
            bilby.core.prior.WeightedDiscreteValues(
                name="test", unit="unit", values=[1, 2, 3, 4], weights=[1, 2, 3, 4]
            ),
            bilby.core.prior.DiscreteValues(
                name="test", unit="unit", values=[1, 2, 3, 4]
            ),
            bilby.core.prior.WeightedCategorical(
                name="test", unit="unit", ncategories=4, weights=[1, 2, 3, 4]
            ),
            bilby.core.prior.Categorical(name="test", unit="unit", ncategories=5),
            bilby.core.prior.SymmetricLogUniform(
                name="test", unit="unit", minimum=1e-2, maximum=1e2
            ),
            bilby.gw.prior.AlignedSpin(name="test", unit="unit"),
            aligned_prior_complex,
            bilby.core.prior.MultivariateGaussian(dist=mvg, name="testa", unit="unit"),
            bilby.core.prior.MultivariateGaussian(dist=mvg, name="testb", unit="unit"),
            bilby.core.prior.MultivariateNormal(dist=mvn, name="testa", unit="unit"),
            bilby.core.prior.MultivariateNormal(dist=mvn, name="testb", unit="unit"),
            bilby.core.prior.ConditionalDeltaFunction(
                condition_func=condition_func, name="test", unit="unit", peak=1
            ),
            bilby.core.prior.ConditionalGaussian(
                condition_func=condition_func, name="test", unit="unit", mu=0, sigma=1
            ),
            bilby.core.prior.ConditionalPowerLaw(
                condition_func=condition_func,
                name="test",
                unit="unit",
                alpha=0,
                minimum=0,
                maximum=1,
            ),
            bilby.core.prior.ConditionalPowerLaw(
                condition_func=condition_func,
                name="test",
                unit="unit",
                alpha=-1,
                minimum=0.5,
                maximum=1,
            ),
            bilby.core.prior.ConditionalPowerLaw(
                condition_func=condition_func,
                name="test",
                unit="unit",
                alpha=2,
                minimum=1,
                maximum=1e2,
            ),
            bilby.core.prior.ConditionalUniform(
                condition_func=condition_func,
                name="test",
                unit="unit",
                minimum=0,
                maximum=1,
            ),
            bilby.core.prior.ConditionalLogUniform(
                condition_func=condition_func,
                name="test",
                unit="unit",
                minimum=5e0,
                maximum=1e2,
            ),
            bilby.core.prior.Triangular(
                name="test",
                unit="unit",
                minimum=-1.1,
                maximum=3.14,
                mode=0.0,
            ),
            bilby.core.prior.Triangular(
                name="test",
                unit="unit",
                minimum=0.0,
                maximum=4.0,
                mode=4.0,
            ),
            bilby.core.prior.Triangular(
                name="test",
                unit="unit",
                minimum=2.0,
                maximum=5.0,
                mode=2.0,
            ),
            bilby.gw.prior.ConditionalUniformComovingVolume(
                condition_func=condition_func, name="redshift", minimum=0.1, maximum=1.0
            ),
            bilby.gw.prior.ConditionalUniformSourceFrame(
                condition_func=condition_func, name="redshift", minimum=0.1, maximum=1.0
            ),
            bilby.core.prior.ConditionalSine(
                condition_func=condition_func, name="test", unit="unit"
            ),
            bilby.core.prior.ConditionalCosine(
                condition_func=condition_func, name="test", unit="unit"
            ),
            bilby.core.prior.ConditionalTruncatedGaussian(
                condition_func=condition_func,
                name="test",
                unit="unit",
                mu=1,
                sigma=0.4,
                minimum=-1,
                maximum=1,
            ),
            bilby.core.prior.ConditionalHalfGaussian(
                condition_func=condition_func, name="test", unit="unit", sigma=1
            ),
            bilby.core.prior.ConditionalLogNormal(
                condition_func=condition_func, name="test", unit="unit", mu=0, sigma=1
            ),
            bilby.core.prior.ConditionalExponential(
                condition_func=condition_func, name="test", unit="unit", mu=1
            ),
            bilby.core.prior.ConditionalStudentT(
                condition_func=condition_func,
                name="test",
                unit="unit",
                df=3,
                mu=0,
                scale=1,
            ),
            bilby.core.prior.ConditionalBeta(
                condition_func=condition_func,
                name="test",
                unit="unit",
                alpha=2.0,
                beta=2.0,
            ),
            bilby.core.prior.ConditionalLogistic(
                condition_func=condition_func, name="test", unit="unit", mu=0, scale=1
            ),
            bilby.core.prior.ConditionalCauchy(
                condition_func=condition_func, name="test", unit="unit", alpha=0, beta=1
            ),
            bilby.core.prior.ConditionalGamma(
                condition_func=condition_func, name="test", unit="unit", k=1, theta=1
            ),
            bilby.core.prior.ConditionalChiSquared(
                condition_func=condition_func, name="test", unit="unit", nu=2
            ),
            bilby.gw.prior.HealPixPrior(dist=hp_dist, name="testra", unit="unit"),
            bilby.gw.prior.HealPixPrior(dist=hp_dist, name="testdec", unit="unit"),
            bilby.gw.prior.HealPixPrior(dist=hp_3d_dist, name="testra", unit="unit"),
            bilby.gw.prior.HealPixPrior(dist=hp_3d_dist, name="testdec", unit="unit"),
            bilby.gw.prior.HealPixPrior(
                dist=hp_3d_dist, name="testdistance", unit="unit"
            ),
        ]
        if aac.is_torch_namespace(self.xp):
            self.priors = [
                p for p in self.priors
                if not isinstance(p, bilby.core.prior.Interped)
            ]
        elif aac.is_jax_namespace(self.xp):
            self.priors = [
                p for p in self.priors
                if not isinstance(p, bilby.core.prior.StudentT)
            ]

    def tearDown(self):
        del self.priors

    def test_jits(self):
        if not aac.is_jax_namespace(self.xp):
            pytest.skip("Jitting test only works with JAX")

        import jax
        from bilby.compat import pytrees  # noqa

        @jax.jit
        def evaluate_prior(prior_, val):
            return prior_.prob(val)

        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.JointPrior):
                continue
            cache_size = evaluate_prior._cache_size()
            sample = jax.numpy.asarray(prior.sample(3))
            evaluate_prior(prior, sample)
            alt_prior = deepcopy(prior)
            sample = jax.numpy.asarray(alt_prior.sample(3))
            evaluate_prior(alt_prior, sample)
            new_cache_size = evaluate_prior._cache_size()
            message = f"Cache size increased by more than 1 for {prior.__class__.__name__}"
            assert new_cache_size <= cache_size + 1, message

    def _validate_return_type(self, val):
        if not isinstance(val, (int, float)):
            self.assertEqual(aac.get_namespace(val), self.xp)

    def test_minimum_rescaling(self):
        """Test the the rescaling works as expected."""
        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.analytical.SymmetricLogUniform):
                # SymmetricLogUniform has support down to -maximum
                continue
            if isinstance(prior, bilby.gw.prior.AlignedSpin):
                # the edge of the prior is extremely suppressed for these priors
                # and so the rescale function doesn't quite return the lower bound
                continue
            if isinstance(prior, bilby.gw.prior.HealPixPrior) and aac.is_torch_namespace(self.xp):
                # HealPix rescaling requires interpolation
                continue
            with self.subTest(prior=prior):
                if bilby.core.prior.JointPrior in prior.__class__.__mro__:
                    minimum_sample = prior.rescale(self.xp.asarray(0))
                    if prior.dist.filled_rescale():
                        self.assertAlmostEqual(np.asarray(minimum_sample[0]), prior.minimum)
                        self.assertAlmostEqual(np.asarray(minimum_sample[1]), prior.minimum)
                else:
                    minimum_sample = prior.rescale(self.xp.asarray(0))
                    self.assertAlmostEqual(np.asarray(minimum_sample), prior.minimum)

    def test_maximum_rescaling(self):
        """Test the the rescaling works as expected."""
        for prior in self.priors:
            if isinstance(prior, bilby.gw.prior.HealPixPrior) and aac.is_torch_namespace(self.xp):
                # HealPix rescaling requires interpolation
                continue
            with self.subTest(prior=prior):
                if bilby.core.prior.JointPrior in prior.__class__.__mro__:
                    maximum_sample = prior.rescale(self.xp.asarray(0))
                    if prior.dist.filled_rescale():
                        self.assertAlmostEqual(np.asarray(maximum_sample[0]), prior.maximum)
                        self.assertAlmostEqual(np.asarray(maximum_sample[1]), prior.maximum)
                elif isinstance(prior, bilby.gw.prior.AlignedSpin):
                    maximum_sample = prior.rescale(self.xp.asarray(1))
                    self.assertGreater(np.asarray(maximum_sample), 0.997)
                else:
                    maximum_sample = prior.rescale(self.xp.asarray(1))
                    self.assertAlmostEqual(np.asarray(maximum_sample), prior.maximum)

    def test_many_sample_rescaling(self):
        """Test the the rescaling works as expected."""
        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.analytical.SymmetricLogUniform):
                # SymmetricLogUniform has support down to -maximum
                continue
            if isinstance(prior, bilby.gw.prior.HealPixPrior) and aac.is_torch_namespace(self.xp):
                # HealPix rescaling requires interpolation
                continue
            many_samples = prior.rescale(self.xp.asarray(np.random.uniform(0, 1, 1000)))
            if bilby.core.prior.JointPrior in prior.__class__.__mro__:
                if not prior.dist.filled_rescale():
                    continue
            with self.subTest(prior=prior):
                self.assertTrue(
                    all((many_samples >= prior.minimum) & (many_samples <= prior.maximum))
                )
                self._validate_return_type(many_samples)

    def test_least_recently_sampled(self):
        for prior in self.priors:
            with self.subTest(prior=prior):
                least_recently_sampled_expected = prior.sample(random_state=self.rng)
                self.assertEqual(
                    least_recently_sampled_expected, prior.least_recently_sampled
                )
                self._validate_return_type(least_recently_sampled_expected)

    def test_sampling_single(self):
        """Test that sampling from the prior always returns values within its domain."""
        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.analytical.SymmetricLogUniform):
                # SymmetricLogUniform has support down to -maximum
                continue
            with self.subTest(prior=prior):
                single_sample = prior.sample(random_state=self.rng)
                self.assertGreaterEqual(single_sample, prior.minimum)
                self.assertLessEqual(single_sample, prior.maximum)
                self._validate_return_type(single_sample)

    def test_sampling_many(self):
        """Test that sampling from the prior always returns values within its domain."""
        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.analytical.SymmetricLogUniform):
                # SymmetricLogUniform has support down to -maximum
                continue
            with self.subTest(prior=prior):
                many_samples = prior.sample(5000, random_state=self.rng)
                self.assertGreaterEqual(min(many_samples), prior.minimum)
                self.assertLessEqual(max(many_samples), prior.maximum)
                self._validate_return_type(many_samples)

    def test_probability_above_domain(self):
        """Test that the prior probability is non-negative in domain of validity and zero outside."""
        for prior in self.priors:
            if prior.maximum != np.inf:
                outside_domain = self.xp.linspace(
                    prior.maximum + 1, prior.maximum + 1e4, 1000
                )
                if bilby.core.prior.JointPrior in prior.__class__.__mro__:
                    if not prior.dist.filled_request():
                        prior.dist.requested_parameters[prior.name] = outside_domain
                        continue
                self.assertTrue(all(prior.prob(outside_domain) == 0))

    def test_probability_below_domain(self):
        """Test that the prior probability is non-negative in domain of validity and zero outside."""
        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.analytical.SymmetricLogUniform):
                # SymmetricLogUniform has support down to -maximum
                continue
            if prior.minimum != -np.inf:
                outside_domain = self.xp.linspace(
                    prior.minimum - 1e4, prior.minimum - 1, 1000
                )
                if bilby.core.prior.JointPrior in prior.__class__.__mro__:
                    if not prior.dist.filled_request():
                        prior.dist.requested_parameters[prior.name] = outside_domain
                        continue
                self.assertTrue(all(prior.prob(outside_domain) == 0))

    def test_least_recently_sampled_2(self):
        for prior in self.priors:
            with self.subTest(prior=prior):
                lrs = prior.sample(random_state=self.rng)
                self.assertEqual(lrs, prior.least_recently_sampled)
                self._validate_return_type(lrs)

    def test_prob_and_ln_prob(self):
        for prior in self.priors:
            sample = prior.sample(random_state=self.rng)
            if bilby.core.prior.JointPrior in prior.__class__.__mro__:  # noqa
                # due to the way that the Multivariate Gaussian prior must sequentially call
                # the prob and ln_prob functions, it must be ignored in this test.
                continue
            with self.subTest(prior=prior):
                lnprob = prior.ln_prob(sample)
                prob = prior.prob(sample)
                self._validate_return_type(lnprob)
                self._validate_return_type(prob)
                # lower precision for jax running tests with float32
                lnprob = np.asarray(lnprob)
                prob = np.asarray(prob)
                self.assertAlmostEqual(np.log(prob), lnprob, 6)

    def test_many_prob_and_many_ln_prob(self):
        for prior in self.priors:
            samples = prior.sample(10, random_state=self.rng)
            if bilby.core.prior.JointPrior in prior.__class__.__mro__:  # noqa
                continue
            with self.subTest(prior=prior):
                ln_probs = prior.ln_prob(samples)
                probs = prior.prob(samples)
                self._validate_return_type(ln_probs)
                self._validate_return_type(probs)
                ln_probs = np.asarray(ln_probs)
                probs = np.asarray(probs)
                for sample, logp, p in zip(samples, ln_probs, probs):
                    new_lnprob = np.asarray(prior.ln_prob(sample))
                    new_prob = np.asarray(prior.prob(sample))
                    self.assertAlmostEqual(new_lnprob, logp, 6)
                    self.assertAlmostEqual(new_prob, p, 6)

    def test_cdf_is_inverse_of_rescaling(self):
        domain = self.xp.linspace(0, 1, 100)
        threshold = 1e-9
        for prior in self.priors:
            if (
                isinstance(prior, bilby.core.prior.DeltaFunction)
                or bilby.core.prior.JointPrior in prior.__class__.__mro__
            ):
                continue
            elif isinstance(prior, bilby.core.prior.StudentT) and "jax" in str(self.xp):
                # JAX implementation of StudentT prior rescale is not accurate enough
                continue
            with self.subTest(prior=prior):
                if isinstance(prior, bilby.core.prior.WeightedDiscreteValues):
                    rescaled = prior.rescale(domain)
                    cdf_vals = prior.cdf(rescaled)
                    rescaled_2 = prior.rescale(cdf_vals)
                    cdf_vals_2 = prior.cdf(rescaled_2)
                    self.assertTrue(np.array_equal(rescaled, rescaled_2))
                    max_difference = max(np.abs(cdf_vals - cdf_vals_2))
                    for arr in [rescaled, rescaled_2, cdf_vals, cdf_vals_2]:
                        self._validate_return_type(arr)
                else:
                    rescaled = prior.rescale(domain)
                    max_difference = max(np.abs(domain - prior.cdf(rescaled)))
                    self._validate_return_type(rescaled)
                self.assertLess(max_difference, threshold)

    def test_cdf_one_above_domain(self):
        for prior in self.priors:
            if isinstance(prior, bilby.gw.prior.HealPixPrior) and aac.is_torch_namespace(self.xp):
                # HealPix rescaling requires interpolation
                continue
            if prior.maximum == np.inf:
                continue
            with self.subTest(prior=prior):
                outside_domain = self.xp.linspace(
                    prior.maximum + 1, prior.maximum + 1e4, 1000
                )
                self.assertTrue(all(prior.cdf(outside_domain) == 1))

    def test_cdf_zero_below_domain(self):
        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.analytical.SymmetricLogUniform):
                # SymmetricLogUniform has support down to -maximum
                continue
            if isinstance(prior, bilby.gw.prior.HealPixPrior) and aac.is_torch_namespace(self.xp):
                # HealPix rescaling requires interpolation
                continue
            if (
                bilby.core.prior.JointPrior in prior.__class__.__mro__
                and prior.maximum == np.inf
            ):
                continue
            if prior.minimum == -np.inf:
                continue
            with self.subTest(prior=prior):
                outside_domain = self.xp.linspace(
                    prior.minimum - 1e4, prior.minimum - 1, 1000
                )
                self.assertTrue(all(np.nan_to_num(prior.cdf(outside_domain)) == 0))

    def test_cdf_float_with_float_input(self):
        for prior in self.priors:
            if bilby.core.prior.JointPrior in prior.__class__.__mro__:
                continue
            with self.subTest(prior=prior):
                self.assertIsInstance(prior.cdf(prior.sample()), float)

    def test_log_normal_fail(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.LogNormal(name="test", unit="unit", mu=0, sigma=-1)

    def test_studentt_fail(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.StudentT(name="test", unit="unit", df=3, mu=0, scale=-1)
        with self.assertRaises(ValueError):
            bilby.core.prior.StudentT(name="test", unit="unit", df=0, mu=0, scale=1)

    def test_beta_fail(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.Beta(name="test", unit="unit", alpha=-2.0, beta=2.0),

        with self.assertRaises(ValueError):
            bilby.core.prior.Beta(name="test", unit="unit", alpha=2.0, beta=-2.0),

    def test_multivariate_gaussian_fail(self):
        with self.assertRaises(ValueError):
            # bounds is wrong length
            bilby.core.prior.MultivariateGaussianDist(["a", "b"], bounds=[(-1.0, 1.0)])
        with self.assertRaises(ValueError):
            # bounds has lower value greater than upper
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"], bounds=[(-1.0, 1.0), (1.0, -1)]
            )
        with self.assertRaises(TypeError):
            # bound is not a list/tuple
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"], bounds=[(-1.0, 1.0), 2]
            )
        with self.assertRaises(ValueError):
            # bound contains too many values
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"], bounds=[(-1.0, 1.0, 4), 2]
            )
        with self.assertRaises(ValueError):
            # means is not a list
            bilby.core.prior.MultivariateGaussianDist(["a", "b"], mus=1.0)
        with self.assertRaises(ValueError):
            # sigmas is not a list
            bilby.core.prior.MultivariateGaussianDist(["a", "b"], sigmas=1.0)
        with self.assertRaises(TypeError):
            # covariances is not a list
            bilby.core.prior.MultivariateGaussianDist(["a", "b"], covs=1.0)
        with self.assertRaises(TypeError):
            # correlation coefficients is not a list
            bilby.core.prior.MultivariateGaussianDist(["a", "b"], corrcoefs=1.0)
        with self.assertRaises(ValueError):
            # wrong number of weights
            bilby.core.prior.MultivariateGaussianDist(["a", "b"], weights=[0.5, 0.5])
        with self.assertRaises(ValueError):
            # not enough modes set
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"], mus=[[1.0, 2.0]], nmodes=2
            )
        with self.assertRaises(ValueError):
            # covariance is the wrong shape
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"], covs=np.array([[[1.0, 1.0], [1.0, 1.0]]])
            )
        with self.assertRaises(ValueError):
            # covariance is the wrong shape
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"], covs=np.array([[[1.0, 1.0]]])
            )
        with self.assertRaises(ValueError):
            # correlation coefficient matrix is the wrong shape
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"],
                sigmas=[1.0, 1.0],
                corrcoefs=np.array([[[[1.0, 1.0], [1.0, 1.0]]]]),
            )
        with self.assertRaises(ValueError):
            # correlation coefficient matrix is the wrong shape
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"], sigmas=[1.0, 1.0], corrcoefs=np.array([[[1.0, 1.0]]])
            )
        with self.assertRaises(ValueError):
            # correlation coefficient has non-unity diagonal value
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"],
                sigmas=[1.0, 1.0],
                corrcoefs=np.array([[1.0, 1.0], [1.0, 2.0]]),
            )
        with self.assertRaises(ValueError):
            # correlation coefficient matrix is not symmetric
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"],
                sigmas=[1.0, 2.0],
                corrcoefs=np.array([[1.0, -1.2], [-0.3, 1.0]]),
            )
        with self.assertRaises(ValueError):
            # correlation coefficient matrix is not positive definite
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"],
                sigmas=[1.0, 2.0],
                corrcoefs=np.array([[1.0, -1.3], [-1.3, 1.0]]),
            )
        with self.assertRaises(ValueError):
            # wrong number of sigmas
            bilby.core.prior.MultivariateGaussianDist(
                ["a", "b"],
                sigmas=[1.0, 2.0, 3.0],
                corrcoefs=np.array([[1.0, 0.3], [0.3, 1.0]]),
            )

    def test_multivariate_gaussian_covariance(self):
        """Test that the correlation coefficient/covariance matrices are correct"""
        cov = np.array([[4.0, 0], [0.0, 9.0]])
        mvg = bilby.core.prior.MultivariateGaussianDist(["a", "b"], covs=cov)
        self.assertEqual(mvg.nmodes, 1)
        self.assertTrue(np.allclose(mvg.covs[0], cov))
        self.assertTrue(np.allclose(mvg.sigmas[0], np.sqrt(np.diag(cov))))
        self.assertTrue(np.allclose(mvg.corrcoefs[0], np.eye(2)))

        corrcoef = np.array([[1.0, 0.5], [0.5, 1.0]])
        sigma = [2.0, 2.0]
        mvg = bilby.core.prior.MultivariateGaussianDist(
            ["a", "b"], corrcoefs=corrcoef, sigmas=sigma
        )
        self.assertTrue(np.allclose(mvg.corrcoefs[0], corrcoef))
        self.assertTrue(np.allclose(mvg.sigmas[0], sigma))
        self.assertTrue(np.allclose(np.diag(mvg.covs[0]), np.square(sigma)))
        self.assertTrue(np.allclose(np.diag(np.fliplr(mvg.covs[0])), 2.0 * np.ones(2)))

    def test_fermidirac_fail(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.FermiDirac(name="test", unit="unit", sigma=1.0)

        with self.assertRaises(ValueError):
            bilby.core.prior.FermiDirac(name="test", unit="unit", sigma=1.0, mu=-1)

    def test_probability_in_domain(self):
        """Test that the prior probability is non-negative in domain of validity and zero outside."""
        for prior in self.priors:
            with self.subTest(prior=prior):
                if prior.minimum == -np.inf:
                    minimum = -1e5
                else:
                    minimum = prior.minimum
                if prior.maximum == np.inf:
                    maximum = 1e5
                else:
                    maximum = prior.maximum
                domain = self.xp.linspace(minimum, maximum, 1000)
                prob = prior.prob(domain)
                self._validate_return_type(prob)
                prob = np.asarray(prob)
                self.assertTrue(all(prob >= 0))

    def test_probability_surrounding_domain(self):
        """Test that the prior probability is non-negative in domain of validity and zero outside."""
        for prior in self.priors:
            # skip delta function prior in this case
            if isinstance(prior, bilby.core.prior.DeltaFunction):
                continue
            if isinstance(prior, bilby.core.prior.analytical.SymmetricLogUniform):
                # SymmetricLogUniform has support down to -maximum
                continue
            with np.errstate(invalid="ignore"):
                surround_domain = self.xp.linspace(prior.minimum - 1, prior.maximum + 1, 1000)
                indomain = (surround_domain >= prior.minimum) | (
                    surround_domain <= prior.maximum
                )
                outdomain = (surround_domain < prior.minimum) | (
                    surround_domain > prior.maximum
                )
            if bilby.core.prior.JointPrior in prior.__class__.__mro__:
                if not prior.dist.filled_request():
                    continue
            with self.subTest(prior=prior):
                self.assertTrue(all(prior.prob(surround_domain[indomain]) >= 0))
                self.assertTrue(all(prior.prob(surround_domain[outdomain]) == 0))

    def test_normalized(self):
        """
        Test that each of the priors are normalised.
        This needs extra care for priors defined on infinite domains and the
        Cauchy, DeltaFunction, and SymmetricLogUniform priors are skipped
        because they are too sharply peaked to be tested efficiently in this way.
        """
        for prior in self.priors:
            if isinstance(
                prior,
                (
                    bilby.core.prior.DeltaFunction,
                    bilby.core.prior.Cauchy,
                    bilby.core.prior.SymmetricLogUniform,
                ),
            ):
                continue
            if bilby.core.prior.JointPrior in prior.__class__.__mro__:
                continue
            elif isinstance(prior, bilby.core.prior.Gaussian):
                domain = np.linspace(-1e2, 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.Cauchy):
                domain = np.linspace(-1e2, 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.StudentT):
                domain = np.linspace(-1e2, 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.HalfGaussian):
                domain = np.linspace(0.0, 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.Gamma):
                domain = np.linspace(0.0, 1e2, 5000)
            elif isinstance(prior, bilby.core.prior.LogNormal):
                domain = np.linspace(0.0, 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.Exponential):
                domain = np.linspace(0.0, 1e2, 5000)
            elif isinstance(prior, bilby.core.prior.Logistic):
                domain = np.linspace(-1e2, 1e2, 1000)
            elif isinstance(prior, bilby.core.prior.FermiDirac):
                domain = np.linspace(0.0, 1e2, 1000)
            elif isinstance(prior, bilby.gw.prior.AlignedSpin):
                domain = np.linspace(prior.minimum, prior.maximum, 10000)
            elif isinstance(prior, bilby.core.prior.WeightedDiscreteValues):
                domain = prior.values
                continue
            else:
                domain = np.linspace(prior.minimum, prior.maximum, 1000)
            with self.subTest(prior=prior):
                if isinstance(prior, bilby.core.prior.WeightedDiscreteValues):
                    probs = prior.prob(self.xp.asarray(domain))
                    self._validate_return_type(probs)
                    self.assertTrue(np.sum(np.asarray(probs)) == 1)
                else:
                    probs = prior.prob(self.xp.asarray(domain))
                    self.assertAlmostEqual(trapezoid(np.array(probs), domain), 1, 3)
                    self._validate_return_type(probs)

    def test_accuracy(self):
        """Test that each of the priors' functions is calculated accurately, as compared to scipy's calculations"""
        for prior in self.priors:
            rescale_domain = np.linspace(0, 1, 1000)
            if isinstance(prior, bilby.core.prior.Uniform):
                domain = np.linspace(-5, 5, 100)
                scipy_prob = ss.uniform.pdf(domain, loc=0, scale=1)
                scipy_lnprob = ss.uniform.logpdf(domain, loc=0, scale=1)
                scipy_cdf = ss.uniform.cdf(domain, loc=0, scale=1)
                scipy_rescale = ss.uniform.ppf(rescale_domain, loc=0, scale=1)
            elif isinstance(prior, bilby.core.prior.Gaussian):
                domain = np.linspace(-1e2, 1e2, 1000)
                scipy_prob = ss.norm.pdf(domain, loc=0, scale=1)
                scipy_lnprob = ss.norm.logpdf(domain, loc=0, scale=1)
                scipy_cdf = ss.norm.cdf(domain, loc=0, scale=1)
                scipy_rescale = ss.norm.ppf(rescale_domain, loc=0, scale=1)
            elif isinstance(prior, bilby.core.prior.Cauchy):
                domain = np.linspace(-1e2, 1e2, 1000)
                scipy_prob = ss.cauchy.pdf(domain, loc=0, scale=1)
                scipy_lnprob = ss.cauchy.logpdf(domain, loc=0, scale=1)
                scipy_cdf = ss.cauchy.cdf(domain, loc=0, scale=1)
                scipy_rescale = ss.cauchy.ppf(rescale_domain, loc=0, scale=1)
            elif isinstance(prior, bilby.core.prior.StudentT):
                domain = np.linspace(-1e2, 1e2, 1000)
                scipy_prob = ss.t.pdf(domain, 3, loc=0, scale=1)
                scipy_lnprob = ss.t.logpdf(domain, 3, loc=0, scale=1)
                scipy_cdf = ss.t.cdf(domain, 3, loc=0, scale=1)
                scipy_rescale = ss.t.ppf(rescale_domain, 3, loc=0, scale=1)
            elif isinstance(prior, bilby.core.prior.Gamma) and not isinstance(
                prior, bilby.core.prior.ChiSquared
            ):
                domain = np.linspace(0.0, 1e2, 5000)
                scipy_prob = ss.gamma.pdf(domain, 1, loc=0, scale=1)
                scipy_lnprob = ss.gamma.logpdf(domain, 1, loc=0, scale=1)
                scipy_cdf = ss.gamma.cdf(domain, 1, loc=0, scale=1)
                scipy_rescale = ss.gamma.ppf(rescale_domain, 1, loc=0, scale=1)
            elif isinstance(prior, bilby.core.prior.LogNormal):
                domain = np.linspace(0.0, 1e2, 1000)
                scipy_prob = ss.lognorm.pdf(domain, 1, scale=1)
                scipy_lnprob = ss.lognorm.logpdf(domain, 1, scale=1)
                scipy_cdf = ss.lognorm.cdf(domain, 1, scale=1)
                scipy_rescale = ss.lognorm.ppf(rescale_domain, 1, scale=1)
            elif isinstance(prior, bilby.core.prior.Exponential):
                domain = np.linspace(0.0, 1e2, 5000)
                scipy_prob = ss.expon.pdf(domain, scale=1)
                scipy_lnprob = ss.expon.logpdf(domain, scale=1)
                scipy_cdf = ss.expon.cdf(domain, scale=1)
                scipy_rescale = ss.expon.ppf(rescale_domain, scale=1)
            elif isinstance(prior, bilby.core.prior.Logistic):
                domain = np.linspace(-1e2, 1e2, 1000)
                scipy_prob = ss.logistic.pdf(domain, loc=0, scale=1)
                scipy_lnprob = ss.logistic.logpdf(domain, loc=0, scale=1)
                scipy_cdf = ss.logistic.cdf(domain, loc=0, scale=1)
                scipy_rescale = ss.logistic.ppf(rescale_domain, loc=0, scale=1)
            elif isinstance(prior, bilby.core.prior.ChiSquared):
                domain = np.linspace(0.0, 1e2, 5000)
                scipy_prob = ss.gamma.pdf(domain, 1, loc=0, scale=2)
                scipy_lnprob = ss.gamma.logpdf(domain, 1, loc=0, scale=2)
                scipy_cdf = ss.gamma.cdf(domain, 1, loc=0, scale=2)
                scipy_rescale = ss.gamma.ppf(rescale_domain, 1, loc=0, scale=2)
            elif isinstance(prior, bilby.core.prior.Beta):
                domain = np.linspace(-5, 5, 5000)
                scipy_prob = ss.beta.pdf(domain, 2, 2, loc=0, scale=1)
                scipy_lnprob = ss.beta.logpdf(domain, 2, 2, loc=0, scale=1)
                scipy_cdf = ss.beta.cdf(domain, 2, 2, loc=0, scale=1)
                scipy_rescale = ss.beta.ppf(rescale_domain, 2, 2, loc=0, scale=1)
            elif isinstance(prior, bilby.core.prior.WeightedDiscreteValues):
                domain = prior.values
                rescale_domain = prior.weights
                scipy_dist = ss.rv_discrete(
                    a=np.min(domain),
                    b=np.max(domain),
                    values=(domain, rescale_domain),
                )
                scipy_prob = scipy_dist.pmf(domain)
                scipy_lnprob = scipy_dist.logpmf(domain)
                scipy_cdf = scipy_dist.cdf(domain)
                scipy_rescale = scipy_dist.ppf(rescale_domain)
            else:
                continue
            testTuple = (
                bilby.core.prior.Uniform,
                bilby.core.prior.Gaussian,
                bilby.core.prior.Cauchy,
                bilby.core.prior.StudentT,
                bilby.core.prior.Exponential,
                bilby.core.prior.Logistic,
                bilby.core.prior.LogNormal,
                bilby.core.prior.Gamma,
                bilby.core.prior.Beta,
                bilby.core.prior.WeightedDiscreteValues,
            )
            if isinstance(prior, (testTuple)):
                with self.subTest(prior=prior):
                    np.testing.assert_almost_equal(prior.prob(self.xp.asarray(domain)), scipy_prob)
                    np.testing.assert_almost_equal(prior.ln_prob(self.xp.asarray(domain)), scipy_lnprob)
                    np.testing.assert_almost_equal(prior.cdf(self.xp.asarray(domain)), scipy_cdf)
                    if isinstance(prior, bilby.core.prior.StudentT) and "jax" in str(self.xp):
                        # JAX implementation of StudentT prior rescale is not accurate enough
                        continue
                    np.testing.assert_almost_equal(
                        prior.rescale(self.xp.asarray(rescale_domain)), scipy_rescale
                    )

    def test_unit_setting(self):
        for prior in self.priors:
            with self.subTest(prior=prior):
                if isinstance(prior, bilby.gw.prior.Cosmological):
                    self.assertEqual(None, prior.unit)
                else:
                    self.assertEqual("unit", prior.unit)

    def test_eq_different_classes(self):
        for i in range(len(self.priors)):
            for j in range(len(self.priors)):
                with self.subTest(i=self.priors[i], j=self.priors[j]):
                    if i == j:
                        self.assertEqual(self.priors[i], self.priors[j])
                    else:
                        self.assertNotEqual(self.priors[i], self.priors[j])

    def test_eq_other_condition(self):
        prior_1 = bilby.core.prior.PowerLaw(
            name="test", unit="unit", alpha=0, minimum=0, maximum=1
        )
        prior_2 = bilby.core.prior.PowerLaw(
            name="test", unit="unit", alpha=0, minimum=0, maximum=1.5
        )
        self.assertNotEqual(prior_1, prior_2)

    def test_eq_different_keys(self):
        prior_1 = bilby.core.prior.PowerLaw(
            name="test", unit="unit", alpha=0, minimum=0, maximum=1
        )
        prior_2 = bilby.core.prior.PowerLaw(
            name="test", unit="unit", alpha=0, minimum=0, maximum=1
        )
        prior_2.other_key = 5
        self.assertNotEqual(prior_1, prior_2)

    def test_repr(self):
        for prior in self.priors:
            if isinstance(prior, bilby.core.prior.Interped):
                continue  # we cannot test this because of the numpy arrays
            elif isinstance(prior, bilby.core.prior.MultivariateGaussian):
                repr_prior_string = "bilby.core.prior." + repr(prior)
                repr_prior_string = repr_prior_string.replace(
                    "MultivariateGaussianDist",
                    "bilby.core.prior.MultivariateGaussianDist",
                )
            elif isinstance(prior, bilby.gw.prior.HealPixPrior):
                repr_prior_string = repr(prior)
                repr_prior_string = repr_prior_string.replace(
                    "HealPixMapPriorDist", "bilby.gw.prior.HealPixMapPriorDist"
                )
                prior.dist.rescale_parameters = {key: None for key in prior.dist.names}
            elif isinstance(prior, bilby.gw.prior.UniformComovingVolume):
                repr_prior_string = "bilby.gw.prior." + repr(prior)
            elif "Conditional" in prior.__class__.__name__:
                continue  # This feature does not exist because we cannot recreate the condition function
            else:
                repr_prior_string = "bilby.core.prior." + repr(prior)

            with self.subTest(prior=prior):
                repr_prior = eval(repr_prior_string, None, dict(inf=np.inf))
                self.assertEqual(prior, repr_prior)

    def test_set_maximum_setting(self):
        for prior in self.priors:
            if isinstance(
                prior,
                (
                    bilby.core.prior.DeltaFunction,
                    bilby.core.prior.Gaussian,
                    bilby.core.prior.HalfGaussian,
                    bilby.core.prior.LogNormal,
                    bilby.core.prior.Exponential,
                    bilby.core.prior.StudentT,
                    bilby.core.prior.Logistic,
                    bilby.core.prior.Cauchy,
                    bilby.core.prior.Gamma,
                    bilby.core.prior.MultivariateGaussian,
                    bilby.core.prior.FermiDirac,
                    bilby.core.prior.WeightedDiscreteValues,
                    bilby.core.prior.Triangular,
                    bilby.gw.prior.HealPixPrior,
                ),
            ):
                continue
            with self.subTest(prior=prior):
                prior.maximum = (prior.maximum + prior.minimum) / 2
                self.assertTrue(max(prior.sample(10000, random_state=self.rng)) < prior.maximum)

    def test_set_minimum_setting(self):
        for prior in self.priors:
            if isinstance(
                prior,
                (
                    bilby.core.prior.DeltaFunction,
                    bilby.core.prior.Gaussian,
                    bilby.core.prior.HalfGaussian,
                    bilby.core.prior.LogNormal,
                    bilby.core.prior.Exponential,
                    bilby.core.prior.StudentT,
                    bilby.core.prior.Logistic,
                    bilby.core.prior.Cauchy,
                    bilby.core.prior.Gamma,
                    bilby.core.prior.MultivariateGaussian,
                    bilby.core.prior.FermiDirac,
                    bilby.core.prior.WeightedDiscreteValues,
                    bilby.core.prior.Triangular,
                    bilby.core.prior.SymmetricLogUniform,
                    bilby.gw.prior.HealPixPrior,
                ),
            ):
                continue
            with self.subTest(prior=prior):
                prior.minimum = (prior.maximum + prior.minimum) / 2
                self.assertTrue(min(prior.sample(10000, random_state=self.rng)) > prior.minimum)


if __name__ == "__main__":
    unittest.main()
