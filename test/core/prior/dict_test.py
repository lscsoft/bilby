import os
import unittest
from unittest.mock import Mock

import numpy as np

import bilby


# needs to be defined on module-level for later re-initialization
class MVNSubclass(bilby.core.prior.MultivariateNormalDist):
    def __init__(self, names, mus, covs, weights):
        super().__init__(names=names, mus=mus, covs=covs, weights=weights)


class FakeJointPriorDist(bilby.core.prior.BaseJointPriorDist):

    def __init__(self, names, bounds=None):
        super().__init__(names=names, bounds=bounds)


setattr(bilby.core.prior, "FakeJointPriorDist", FakeJointPriorDist)


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
        self.default_prior_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/precessing_spins_bbh.prior",
        )
        self.joint_prior_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/joint_prior.prior",
        )
        self.prior_set_from_file = bilby.core.prior.PriorDict(
            filename=self.default_prior_file
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
        del self.default_prior_file
        del self.prior_set_from_file

    def test_copy(self):
        priors = bilby.core.prior.PriorDict(self.priors)
        self.assertEqual(priors, priors.copy())

    def test_prior_set_is_dict(self):
        self.assertIsInstance(self.prior_set_from_dict, dict)

    def test_prior_set_has_correct_length(self):
        self.assertEqual(3, len(self.prior_set_from_dict))

    def test_prior_set_has_expected_priors(self):
        self.assertDictEqual(self.priors, dict(self.prior_set_from_dict))

    def test_read_from_file(self):
        expected = dict(
            mass_1=bilby.core.prior.Constraint(
                name="mass_1",
                minimum=5,
                maximum=100,
            ),
            mass_2=bilby.core.prior.Constraint(
                name="mass_2",
                minimum=5,
                maximum=100,
            ),
            chirp_mass=bilby.core.prior.Uniform(
                name="chirp_mass",
                minimum=25,
                maximum=100,
                latex_label=r"$\mathcal{M}$",
            ),
            mass_ratio=bilby.core.prior.Uniform(
                name="mass_ratio",
                minimum=0.125,
                maximum=1,
                latex_label="$q$",
                unit=None,
            ),
            a_1=bilby.core.prior.Uniform(
                name="a_1", minimum=0, maximum=0.99
            ),
            a_2=bilby.core.prior.Uniform(
                name="a_2", minimum=0, maximum=0.99
            ),
            tilt_1=bilby.core.prior.Sine(name="tilt_1"),
            tilt_2=bilby.core.prior.Sine(name="tilt_2"),
            phi_12=bilby.core.prior.Uniform(
                name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            phi_jl=bilby.core.prior.Uniform(
                name="phi_jl", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            luminosity_distance=bilby.gw.prior.UniformSourceFrame(
                name="luminosity_distance",
                minimum=1e2,
                maximum=5e3,
                unit="Mpc",
                boundary=None,
            ),
            dec=bilby.core.prior.Cosine(name="dec"),
            ra=bilby.core.prior.Uniform(
                name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            theta_jn=bilby.core.prior.Sine(name="theta_jn"),
            psi=bilby.core.prior.Uniform(
                name="psi", minimum=0, maximum=np.pi, boundary="periodic"
            ),
            phase=bilby.core.prior.Uniform(
                name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
        )
        self.assertDictEqual(expected, self.prior_set_from_file)

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
        self.prior_set_from_dict["f"] = "unconvertable"
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
            f="unconvertable",
        )
        self.assertDictEqual(expected, self.prior_set_from_dict)

    def test_prior_set_from_dict_but_using_a_string(self):
        prior_set = bilby.core.prior.PriorDict(dictionary=self.default_prior_file)
        expected = bilby.core.prior.PriorDict(
            dict(
                mass_1=bilby.core.prior.Constraint(
                    name="mass_1",
                    minimum=5,
                    maximum=100,
                ),
                mass_2=bilby.core.prior.Constraint(
                    name="mass_2",
                    minimum=5,
                    maximum=100,
                ),
                chirp_mass=bilby.core.prior.Uniform(
                    name="chirp_mass",
                    minimum=25,
                    maximum=100,
                    latex_label=r"$\mathcal{M}$",
                ),
                mass_ratio=bilby.core.prior.Uniform(
                    name="mass_ratio",
                    minimum=0.125,
                    maximum=1,
                    latex_label="$q$",
                    unit=None,
                ),
                a_1=bilby.core.prior.Uniform(
                    name="a_1", minimum=0, maximum=0.99,
                ),
                a_2=bilby.core.prior.Uniform(
                    name="a_2", minimum=0, maximum=0.99,
                ),
                tilt_1=bilby.core.prior.Sine(name="tilt_1"),
                tilt_2=bilby.core.prior.Sine(name="tilt_2"),
                phi_12=bilby.core.prior.Uniform(
                    name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic"
                ),
                phi_jl=bilby.core.prior.Uniform(
                    name="phi_jl", minimum=0, maximum=2 * np.pi, boundary="periodic"
                ),
                luminosity_distance=bilby.gw.prior.UniformSourceFrame(
                    name="luminosity_distance",
                    minimum=1e2,
                    maximum=5e3,
                    unit="Mpc",
                    boundary=None,
                ),
                dec=bilby.core.prior.Cosine(name="dec"),
                ra=bilby.core.prior.Uniform(
                    name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
                ),
                theta_jn=bilby.core.prior.Sine(name="theta_jn"),
                psi=bilby.core.prior.Uniform(
                    name="psi", minimum=0, maximum=np.pi, boundary="periodic"
                ),
                phase=bilby.core.prior.Uniform(
                    name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
                ),
            )
        )
        all_keys = set(prior_set.keys()).union(set(expected.keys()))
        for key in all_keys:
            self.assertEqual(expected[key], prior_set[key])

    def test_dict_argument_is_not_string_or_dict(self):
        with self.assertRaises(ValueError):
            bilby.core.prior.PriorDict(dictionary=list())

    def test_sample_subset_correct_size(self):
        size = 7
        samples = self.prior_set_from_dict.sample_subset(
            keys=self.prior_set_from_dict.keys(), size=size
        )
        self.assertEqual(len(self.prior_set_from_dict), len(samples))
        for key in samples:
            self.assertEqual(size, len(samples[key]))

    def test_sample_subset_correct_size_when_non_priors_in_dict(self):
        self.prior_set_from_dict["asdf"] = "not_a_prior"
        samples = self.prior_set_from_dict.sample_subset(
            keys=self.prior_set_from_dict.keys()
        )
        self.assertEqual(len(self.prior_set_from_dict) - 1, len(samples))

    def test_sample_subset_with_actual_subset(self):
        size = 3
        samples = self.prior_set_from_dict.sample_subset(keys=["length"], size=size)
        expected = dict(length=np.array([42.0, 42.0, 42.0]))
        self.assertTrue(np.array_equal(expected["length"], samples["length"]))

    def test_sample_subset_constrained_as_array(self):
        size = 3
        keys = ["mass", "speed"]
        out = self.prior_set_from_dict.sample_subset_constrained_as_array(keys, size)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertTrue(out.shape == (len(keys), size))

    def test_sample(self):
        size = 7
        bilby.core.utils.random.seed(42)
        samples1 = self.prior_set_from_dict.sample_subset(
            keys=self.prior_set_from_dict.keys(), size=size
        )
        bilby.core.utils.random.seed(42)
        samples2 = self.prior_set_from_dict.sample(size=size)
        self.assertEqual(set(samples1.keys()), set(samples2.keys()))
        for key in samples1:
            self.assertTrue(np.array_equal(samples1[key], samples2[key]))

    def test_prob(self):
        samples = self.prior_set_from_dict.sample_subset(keys=["mass", "speed"])
        expected = self.first_prior.prob(samples["mass"]) * self.second_prior.prob(
            samples["speed"]
        )
        self.assertEqual(expected, self.prior_set_from_dict.prob(samples))

    def test_ln_prob(self):
        samples = self.prior_set_from_dict.sample_subset(keys=["mass", "speed"])
        expected = self.first_prior.ln_prob(
            samples["mass"]
        ) + self.second_prior.ln_prob(samples["speed"])
        self.assertEqual(expected, self.prior_set_from_dict.ln_prob(samples))

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
        sample = self.prior_set_from_dict.sample()
        original = np.array(list(sample.values()))
        new = np.array(self.prior_set_from_dict.rescale(
            sample.keys(),
            self.prior_set_from_dict.cdf(sample=sample).values()
        ))
        self.assertLess(max(abs(original - new)), 1e-10)

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

        hp_map_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/GW150914_testing_skymap.fits",
        )
        hp_dist = bilby.gw.prior.HealPixMapPriorDist(
            hp_map_file, names=["testra", "testdec"]
        )
        hp_3d_dist = bilby.gw.prior.HealPixMapPriorDist(
            hp_map_file, names=["testRA", "testDEC", "testdistance"], distance=True
        )

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
                ii=bilby.gw.prior.UniformComovingVolume(
                    name="redshift", minimum=0.1, maximum=1.0
                ),
                jj=bilby.gw.prior.UniformSourceFrame(
                    name="luminosity_distance", minimum=1.0, maximum=1000.0
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
                ac=bilby.gw.prior.AlignedSpin(name="test", unit="unit"),
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
                testra=bilby.gw.prior.HealPixPrior(
                    dist=hp_dist, name="testra", unit="unit"
                ),
                testdec=bilby.gw.prior.HealPixPrior(
                    dist=hp_dist, name="testdec", unit="unit"
                ),
                testRA=bilby.gw.prior.HealPixPrior(
                    dist=hp_3d_dist, name="testRA", unit="unit"
                ),
                testDEC=bilby.gw.prior.HealPixPrior(
                    dist=hp_3d_dist, name="testDEC", unit="unit"
                ),
                testdistance=bilby.gw.prior.HealPixPrior(
                    dist=hp_3d_dist, name="testdistance", unit="unit"
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

    def test_load_prior_with_function(self):
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/prior_with_function.prior",
        )
        prior = bilby.core.prior.ConditionalPriorDict(filename)
        self.assertTrue("mass_1" in prior)
        self.assertTrue("mass_2" in prior)
        samples = prior.sample(10000)
        self.assertTrue(all(samples["mass_1"] > samples["mass_2"]))


class TestCreateDefaultPrior(unittest.TestCase):
    def test_none_behaviour(self):
        self.assertIsNone(
            bilby.core.prior.create_default_prior(name="name", default_priors_file=None)
        )

    def test_bbh_params(self):
        prior_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/precessing_spins_bbh.prior",
        )
        prior_set = bilby.core.prior.PriorDict(filename=prior_file)
        for prior in prior_set:
            self.assertEqual(
                prior_set[prior],
                bilby.core.prior.create_default_prior(
                    name=prior, default_priors_file=prior_file
                ),
            )

    def test_unknown_prior(self):
        prior_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/precessing_spins_bbh.prior",
        )
        self.assertIsNone(
            bilby.core.prior.create_default_prior(
                name="name", default_priors_file=prior_file
            )
        )


class TestFillPrior(unittest.TestCase):
    def setUp(self):
        self.likelihood = Mock()
        self.likelihood.parameters = dict(a=0, b=0, c=0, d=0, asdf=0, ra=1)
        self.likelihood.non_standard_sampling_parameter_keys = dict(t=8)
        self.priors = dict(a=1, b=1.1, c="string", d=bilby.core.prior.Uniform(0, 1))
        self.priors = bilby.core.prior.PriorDict(dictionary=self.priors)
        self.default_prior_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/precessing_spins_bbh.prior",
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

    def test_with_available_default_priors_a_default_prior_is_set(self):
        self.assertIsInstance(self.priors["ra"], bilby.core.prior.Uniform)


class TestLoadPriorWithCosmologicalParameters(unittest.TestCase):

    def test_load(self):
        prior_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "prior_files/prior_with_cosmo_params.prior"
        )
        prior_dict = bilby.gw.prior.BBHPriorDict(filename=prior_file)
        cosmology = prior_dict["luminosity_distance"].cosmology
        # These values are based on Plank15_LAL as defined in:
        # https://dcc.ligo.org/DocDB/0167/T2000185/005/LVC_symbol_convention.pdf
        self.assertEqual(cosmology.H0.value, 67.90)
        self.assertEqual(cosmology.Om0, 0.3065)

        dl = 1000.0
        ln_prob = prior_dict["luminosity_distance"].ln_prob(dl)
        self.assertEqual(ln_prob, -9.360343006800193)


if __name__ == "__main__":
    unittest.main()
