import os
import unittest
from unittest.mock import Mock, patch

import array_api_compat as aac
import numpy as np
import pytest

import bilby


# needs to be defined on module-level for later re-initialization
class MVNSubclass(bilby.core.prior.MultivariateNormalDist):
    def __init__(self, names, mus, covs, weights):
        super().__init__(names=names, mus=mus, covs=covs, weights=weights)


class FakeJointPriorDist(bilby.core.prior.BaseJointPriorDist):

    def __init__(self, names, bounds=None):
        super().__init__(names=names, bounds=bounds)


setattr(bilby.core.prior, "FakeJointPriorDist", FakeJointPriorDist)


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestPriorDict(unittest.TestCase):

    def setUp(self):

        self.first_prior = bilby.core.prior.Uniform(
            name="a", minimum=0, maximum=1, unit="kg", boundary=None
        )
        self.second_prior = bilby.core.prior.PowerLaw(
            name="b", alpha=3, minimum=1, maximum=2, unit="m/s", boundary=None
        )
        self.third_prior = bilby.core.prior.DeltaFunction(name="c", peak=42, unit="m")
        self.priors = dict(
            mass=self.first_prior, speed=self.second_prior, length=self.third_prior
        )
        self.prior_set_from_dict = bilby.core.prior.PriorDict(dictionary=self.priors)
        self.joint_prior_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/joint_prior.prior",
        )
        self.joint_prior_from_file = bilby.core.prior.PriorDict(
            filename=self.joint_prior_file
        )

    def tearDown(self):
        del self.first_prior
        del self.second_prior
        del self.third_prior
        del self.priors
        del self.prior_set_from_dict

    def test_copy(self):
        priors = bilby.core.prior.PriorDict(self.priors)
        self.assertEqual(priors, priors.copy())

    def test_prior_set_is_dict(self):
        self.assertIsInstance(self.prior_set_from_dict, dict)

    def test_prior_set_has_correct_length(self):
        self.assertEqual(3, len(self.prior_set_from_dict))

    def test_prior_set_has_expected_priors(self):
        self.assertDictEqual(self.priors, dict(self.prior_set_from_dict))

    def test_assignment_rejects_invalid_prior_value(self):
        expected_error = "Unable to parse prior, bad entry"
        with self.assertRaisesRegex(TypeError, expected_error):
            self.prior_set_from_dict["Om0"] = np.array(0.30966)

    def test_read_from_file(self):
        fake_dist = FakeJointPriorDist(names=["testAfake", "testBfake"])
        testAfake = bilby.core.prior.JointPrior(dist=fake_dist, name="testAfake", unit="unit")
        testBfake = bilby.core.prior.JointPrior(dist=fake_dist, name="testBfake", unit="unit")
        base_dist = bilby.core.prior.BaseJointPriorDist(names=["testAbase", "testBbase"])
        testAbase = bilby.core.prior.JointPrior(dist=base_dist, name="testAbase", unit="unit")
        testBbase = bilby.core.prior.JointPrior(dist=base_dist, name="testBbase", unit="unit")
        expected_joint = dict(testAfake=testAfake, testBfake=testBfake, testAbase=testAbase, testBbase=testBbase)
        self.assertDictEqual(expected_joint, self.joint_prior_from_file)
        self.assertTrue(id(self.joint_prior_from_file["testAfake"].dist)
                        == id(self.joint_prior_from_file["testBfake"].dist))
        self.assertTrue(id(self.joint_prior_from_file["testAbase"].dist)
                        == id(self.joint_prior_from_file["testBbase"].dist))

    def test_to_file(self):
        """
        We compare that the strings contain all of the same characters in not
        necessarily the same order as python2 doesn't conserve the order of the
        arguments.
        """
        expected = [
            "length = DeltaFunction(peak=42, name='c', latex_label='c', unit='m')\n",
            "speed = PowerLaw(alpha=3, minimum=1, maximum=2, name='b', latex_label='b', "
            "unit='m/s', boundary=None)\n",
            "mass = Uniform(minimum=0, maximum=1, name='a', latex_label='a', "
            "unit='kg', boundary=None)\n",
        ]
        self.prior_set_from_dict.to_file(outdir="prior_files", label="to_file_test")
        with open("prior_files/to_file_test.prior") as f:
            for i, line in enumerate(f.readlines()):
                self.assertTrue(
                    any([sorted(line) == sorted(expect) for expect in expected])
                )

    def test_from_dict_with_string(self):
        string_prior = (
            "PowerLaw(name='b', alpha=3, minimum=1, maximum=2, unit='m/s', "
            "boundary=None)"
        )
        self.priors["speed"] = string_prior
        from_dict = bilby.core.prior.PriorDict(dictionary=self.priors)
        self.assertDictEqual(self.prior_set_from_dict, from_dict)

    def test_convert_floats_to_delta_functions(self):
        self.prior_set_from_dict["d"] = 5
        self.prior_set_from_dict["e"] = 7.3
        self.prior_set_from_dict.convert_floats_to_delta_functions()
        expected = dict(
            mass=bilby.core.prior.Uniform(
                name="a", minimum=0, maximum=1, unit="kg", boundary=None
            ),
            speed=bilby.core.prior.PowerLaw(
                name="b", alpha=3, minimum=1, maximum=2, unit="m/s", boundary=None
            ),
            length=bilby.core.prior.DeltaFunction(name="c", peak=42, unit="m"),
            d=bilby.core.prior.DeltaFunction(peak=5),
            e=bilby.core.prior.DeltaFunction(peak=7.3),
        )
        self.assertDictEqual(expected, self.prior_set_from_dict)

    def test_dict_argument_is_not_string_or_dict(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.PriorDict(dictionary=list())

    def test_sample_subset_correct_size(self):
        size = 7
        samples = self.prior_set_from_dict.sample_subset(
            keys=self.prior_set_from_dict.keys(), size=size,
            random_state=self.rng,
        )
        self.assertEqual(len(self.prior_set_from_dict), len(samples))
        for key in samples:
            self.assertEqual(size, len(samples[key]))
            self.assertEqual(aac.get_namespace(samples[key]), self.xp)

    def test_sample_subset_with_actual_subset(self):
        size = 3
        samples = self.prior_set_from_dict.sample_subset(
            keys=["length"], size=size, random_state=self.rng
        )
        expected = dict(length=self.xp.asarray([42.0, 42.0, 42.0]))
        self.assertTrue(np.array_equal(expected["length"], samples["length"]))
        self.assertEqual(aac.get_namespace(samples["length"]), self.xp)

    def test_sample_subset_constrained_as_array(self):
        size = 3
        keys = ["mass", "speed"]
        out = self.prior_set_from_dict.sample_subset_constrained_as_array(
            keys, size, random_state=self.rng
        )
        self.assertEqual(aac.get_namespace(out), self.xp)
        self.assertTrue(out.shape == (len(keys), size))

    def test_sample_subset_constrained(self):

        def conversion_function(parameters):
            converted_parameters = parameters.copy()
            converted_parameters["delta_mass"] = (
                parameters["mass_1"] - parameters["mass_2"]
            )
            return converted_parameters

        N = 1_000

        priors1 = bilby.core.prior.PriorDict(conversion_function=conversion_function)
        priors1["mass_1"] = bilby.core.prior.Uniform(minimum=1.38, maximum=2)
        priors1["mass_2"] = bilby.core.prior.Uniform(minimum=1, maximum=1.4)
        priors1["delta_mass"] = bilby.core.prior.Constraint(minimum=-2, maximum=0)

        with patch("bilby.core.prior.logger.warning") as mock_warning:
            samples1 = priors1.sample_subset_constrained(
                keys=list(priors1.keys()), size=N, random_state=self.rng
            )
            self.assertEqual(len(priors1) - 1, len(samples1))
            for key in samples1:
                self.assertEqual(N, len(samples1[key]))
            mock_warning.assert_called()

        priors2 = bilby.core.prior.PriorDict()
        priors2["mass_1"] = bilby.core.prior.Uniform(minimum=1.38, maximum=2)
        priors2["mass_2"] = bilby.core.prior.Uniform(minimum=1, maximum=1.4)

        with patch("bilby.core.prior.logger.warning") as mock_warning:
            samples2 = priors2.sample_subset_constrained(
                keys=list(priors2.keys()), size=N, random_state=self.rng
            )
            self.assertEqual(len(priors2), len(samples2))
            for key in samples2:
                self.assertEqual(N, len(samples2[key]))
            mock_warning.assert_not_called()

    def test_sample_with_random_seed(self):
        """
        This test uses the default RNG, so don't specify random_state.
        """
        size = 7
        bilby.core.utils.random.seed(42)
        samples1 = self.prior_set_from_dict.sample_subset(
            keys=self.prior_set_from_dict.keys(), size=size
        )
        bilby.core.utils.random.seed(42)
        samples2 = self.prior_set_from_dict.sample(size=size)
        self.assertEqual(set(samples1.keys()), set(samples2.keys()))
        for key in samples1:
            np.testing.assert_array_equal(samples1[key], samples2[key])

    def test_sample_returns_correct_type(self):
        """
        This test uses the default RNG, so don't specify random_state.
        """
        size = 7
        samples = self.prior_set_from_dict.sample_subset(
            keys=self.prior_set_from_dict.keys(), size=size, random_state=self.rng
        )
        for key in samples:
            self.assertEqual(aac.get_namespace(samples[key]), self.xp)

    def test_prob(self):
        samples = self.prior_set_from_dict.sample_subset(keys=["mass", "speed"], random_state=self.rng)
        expected = self.first_prior.prob(samples["mass"]) * self.second_prior.prob(
            samples["speed"]
        )
        self.assertEqual(expected, self.prior_set_from_dict.prob(samples))
        self.assertEqual(aac.get_namespace(expected), self.xp)

    def test_ln_prob(self):
        samples = self.prior_set_from_dict.sample_subset(keys=["mass", "speed"], random_state=self.rng)
        expected = self.first_prior.ln_prob(
            samples["mass"]
        ) + self.second_prior.ln_prob(samples["speed"])
        self.assertEqual(expected, self.prior_set_from_dict.ln_prob(samples))
        self.assertEqual(aac.get_namespace(expected), self.xp)

    def test_rescale(self):
        theta = [0.5, 0.5, 0.5]
        expected = [
            self.first_prior.rescale(0.5),
            self.second_prior.rescale(0.5),
            self.third_prior.rescale(0.5),
        ]
        self.assertListEqual(
            sorted(expected),
            sorted(
                self.prior_set_from_dict.rescale(
                    keys=self.prior_set_from_dict.keys(), theta=theta
                )
            ),
        )

    def test_cdf(self):
        """
        Test that the CDF method is the inverse of the rescale method.

        Note that the format of inputs/outputs is different between the two methods.
        """
        sample = self.prior_set_from_dict.sample(random_state=self.rng)
        original = self.xp.asarray(list(sample.values()))
        new = self.xp.asarray(self.prior_set_from_dict.rescale(
            sample.keys(),
            self.prior_set_from_dict.cdf(sample=sample).values()
        ))
        self.assertLess(max(abs(original - new)), 1e-10)
        self.assertEqual(aac.get_namespace(new), self.xp)

    def test_redundancy(self):
        for key in self.prior_set_from_dict.keys():
            self.assertFalse(self.prior_set_from_dict.test_redundancy(key=key))


class TestJsonIO(unittest.TestCase):
    def setUp(self):
        mvg = bilby.core.prior.MultivariateGaussianDist(
            names=["testa", "testb"],
            mus=[1, 1],
            covs=np.array([[2.0, 0.5], [0.5, 2.0]]),
            weights=1.0,
        )
        mvn = bilby.core.prior.MultivariateNormalDist(
            names=["testA", "testB"],
            mus=[1, 1],
            covs=np.array([[2.0, 0.5], [0.5, 2.0]]),
            weights=1.0,
        )

        mvn_subclass = MVNSubclass(
            names=["testAsubclass", "testBsubclass"],
            mus=[1, 1],
            covs=np.array([[2.0, 0.5], [0.5, 2.0]]),
            weights=1.0,
        )

        fake_joint_prior = FakeJointPriorDist(names=["testAfake", "testBfake"])

        self.priors = bilby.core.prior.PriorDict(
            dict(
                aa=bilby.core.prior.DeltaFunction(name="test", unit="unit", peak=1),
                bb=bilby.core.prior.Gaussian(name="test", unit="unit", mu=0, sigma=1),
                cc=bilby.core.prior.Normal(name="test", unit="unit", mu=0, sigma=1),
                dd=bilby.core.prior.PowerLaw(
                    name="test", unit="unit", alpha=0, minimum=0, maximum=1
                ),
                ee=bilby.core.prior.PowerLaw(
                    name="test", unit="unit", alpha=-1, minimum=0.5, maximum=1
                ),
                ff=bilby.core.prior.PowerLaw(
                    name="test", unit="unit", alpha=2, minimum=1, maximum=1e2
                ),
                gg=bilby.core.prior.Uniform(
                    name="test", unit="unit", minimum=0, maximum=1
                ),
                hh=bilby.core.prior.LogUniform(
                    name="test", unit="unit", minimum=5e0, maximum=1e2
                ),
                kk=bilby.core.prior.Sine(name="test", unit="unit"),
                ll=bilby.core.prior.Cosine(name="test", unit="unit"),
                m=bilby.core.prior.Interped(
                    name="test",
                    unit="unit",
                    xx=np.linspace(0, 10, 1000),
                    yy=np.linspace(0, 10, 1000) ** 4,
                    minimum=3,
                    maximum=5,
                ),
                nn=bilby.core.prior.TruncatedGaussian(
                    name="test", unit="unit", mu=1, sigma=0.4, minimum=-1, maximum=1
                ),
                oo=bilby.core.prior.TruncatedNormal(
                    name="test", unit="unit", mu=1, sigma=0.4, minimum=-1, maximum=1
                ),
                pp=bilby.core.prior.HalfGaussian(name="test", unit="unit", sigma=1),
                qq=bilby.core.prior.HalfNormal(name="test", unit="unit", sigma=1),
                rr=bilby.core.prior.LogGaussian(name="test", unit="unit", mu=0, sigma=1),
                ss=bilby.core.prior.LogNormal(name="test", unit="unit", mu=0, sigma=1),
                tt=bilby.core.prior.Exponential(name="test", unit="unit", mu=1),
                uu=bilby.core.prior.StudentT(
                    name="test", unit="unit", df=3, mu=0, scale=1
                ),
                vv=bilby.core.prior.Beta(name="test", unit="unit", alpha=2.0, beta=2.0),
                xx=bilby.core.prior.Logistic(name="test", unit="unit", mu=0, scale=1),
                yy=bilby.core.prior.Cauchy(name="test", unit="unit", alpha=0, beta=1),
                zz=bilby.core.prior.Lorentzian(
                    name="test", unit="unit", alpha=0, beta=1
                ),
                a_=bilby.core.prior.Gamma(name="test", unit="unit", k=1, theta=1),
                ab=bilby.core.prior.ChiSquared(name="test", unit="unit", nu=2),
                testa=bilby.core.prior.MultivariateGaussian(
                    dist=mvg, name="testa", unit="unit"
                ),
                testb=bilby.core.prior.MultivariateGaussian(
                    dist=mvg, name="testb", unit="unit"
                ),
                testA=bilby.core.prior.MultivariateNormal(
                    dist=mvn, name="testA", unit="unit"
                ),
                testB=bilby.core.prior.MultivariateNormal(
                    dist=mvn, name="testB", unit="unit"
                ),
                testAsubclass=bilby.core.prior.JointPrior(
                    dist=mvn_subclass, name="testAsubclass", unit="unit"
                ),
                testBsubclass=bilby.core.prior.JointPrior(
                    dist=mvn_subclass, name="testBsubclass", unit="unit"
                ),
                testAfake=bilby.core.prior.JointPrior(
                    dist=fake_joint_prior, name="testAfake", unit="unit"
                ),
                testBfake=bilby.core.prior.JointPrior(
                    dist=fake_joint_prior, name="testBfake", unit="unit"
                ),
            )
        )

    def test_read_write_to_json(self):
        """ Interped prior is removed as there is numerical error in the recovered prior."""
        self.priors.to_json(outdir="prior_files", label="json_test")
        new_priors = bilby.core.prior.PriorDict.from_json(
            filename="prior_files/json_test_prior.json"
        )
        old_interped = self.priors.pop("m")
        new_interped = new_priors.pop("m")
        self.assertDictEqual(self.priors, new_priors)
        self.assertLess(max(abs(old_interped.xx - new_interped.xx)), 1e-15)
        self.assertLess(max(abs(old_interped.yy - new_interped.yy)), 1e-15)
        self.assertTrue(id(new_priors["testa"].dist) == id(new_priors["testb"].dist))
        self.assertTrue(id(new_priors["testAfake"].dist) == id(new_priors["testBfake"].dist))


class TestLoadPrior(unittest.TestCase):
    def test_load_prior_with_float(self):
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/prior_with_floats.prior",
        )
        prior = bilby.core.prior.PriorDict(filename)
        self.assertTrue("mass_1" in prior)
        self.assertTrue("mass_2" in prior)
        self.assertTrue(prior["mass_2"].peak == 20)

    def test_load_prior_with_parentheses(self):
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/prior_with_parentheses.prior",
        )
        prior = bilby.core.prior.PriorDict(filename)
        self.assertTrue(isinstance(prior["logA"], bilby.core.prior.Uniform))


class TestCreateDefaultPrior(unittest.TestCase):
    def test_none_behaviour(self):
        self.assertIsNone(
            bilby.core.prior.create_default_prior(name="name", default_priors_file=None)
        )


class TestFillPrior(unittest.TestCase):
    def setUp(self):
        self.likelihood = Mock()
        self.likelihood.non_standard_sampling_parameter_keys = dict(t=8)
        self.priors = dict(a=1, b=1.1, c="string", d=bilby.core.prior.Uniform(0, 1))
        self.priors = bilby.core.prior.PriorDict(dictionary=self.priors)
        self.default_prior_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/prior_with_floats.prior",
        )
        self.priors.fill_priors(self.likelihood, self.default_prior_file)

    def tearDown(self):
        del self.likelihood
        del self.priors

    def test_prior_instances_are_not_changed_by_parsing(self):
        self.assertIsInstance(self.priors["d"], bilby.core.prior.Uniform)

    def test_parsing_ints_to_delta_priors_class(self):
        self.assertIsInstance(self.priors["a"], bilby.core.prior.DeltaFunction)

    def test_parsing_ints_to_delta_priors_with_right_value(self):
        self.assertEqual(self.priors["a"].peak, 1)

    def test_parsing_floats_to_delta_priors_class(self):
        self.assertIsInstance(self.priors["b"], bilby.core.prior.DeltaFunction)

    def test_parsing_floats_to_delta_priors_with_right_value(self):
        self.assertAlmostEqual(self.priors["b"].peak, 1.1, 1e-8)

    def test_without_available_default_priors_no_prior_is_set(self):
        with self.assertRaises(KeyError):
            print(self.priors["asdf"])


if __name__ == "__main__":
    unittest.main()
