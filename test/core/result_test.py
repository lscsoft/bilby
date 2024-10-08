import unittest
import numpy as np
import pandas as pd
import shutil
import os
import json
from unittest.mock import patch

import bilby
from bilby.core.result import ResultError


class TestJson(unittest.TestCase):
    def setUp(self):
        self.encoder = bilby.core.utils.BilbyJsonEncoder
        self.decoder = bilby.core.utils.decode_bilby_json

    def test_list_encoding(self):
        data = dict(x=[1, 2, 3.4])
        encoded = json.dumps(data, cls=self.encoder)
        decoded = json.loads(encoded, object_hook=self.decoder)
        self.assertEqual(data.keys(), decoded.keys())
        self.assertEqual(type(data["x"]), type(decoded["x"]))
        self.assertTrue(np.all(data["x"] == decoded["x"]))

    def test_array_encoding(self):
        data = dict(x=np.array([1, 2, 3.4]))
        encoded = json.dumps(data, cls=self.encoder)
        decoded = json.loads(encoded, object_hook=self.decoder)
        self.assertEqual(data.keys(), decoded.keys())
        self.assertEqual(type(data["x"]), type(decoded["x"]))
        self.assertTrue(np.all(data["x"] == decoded["x"]))

    def test_complex_encoding(self):
        data = dict(x=1 + 3j)
        encoded = json.dumps(data, cls=self.encoder)
        decoded = json.loads(encoded, object_hook=self.decoder)
        self.assertEqual(data.keys(), decoded.keys())
        self.assertEqual(type(data["x"]), type(decoded["x"]))
        self.assertTrue(np.all(data["x"] == decoded["x"]))

    def test_dataframe_encoding(self):
        data = dict(data=pd.DataFrame(dict(x=[3, 4, 5], y=[5, 6, 7])))
        encoded = json.dumps(data, cls=self.encoder)
        decoded = json.loads(encoded, object_hook=self.decoder)
        self.assertEqual(data.keys(), decoded.keys())
        self.assertEqual(type(data["data"]), type(decoded["data"]))
        self.assertTrue(np.all(data["data"]["x"] == decoded["data"]["x"]))
        self.assertTrue(np.all(data["data"]["y"] == decoded["data"]["y"]))


class TestResult(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        bilby.utils.command_line_args.bilby_test_mode = False
        priors = bilby.prior.PriorDict(
            dict(
                x=bilby.prior.Uniform(0, 1, "x", latex_label="$x$", unit="s"),
                y=bilby.prior.Uniform(0, 1, "y", latex_label="$y$", unit="m"),
                c=1,
                d=2,
            )
        )
        self.outdir = "test_outdir"
        result = bilby.core.result.Result(
            label="label",
            outdir=self.outdir,
            sampler="nestle",
            search_parameter_keys=["x", "y"],
            fixed_parameter_keys=["c", "d"],
            priors=priors,
            sampler_kwargs=dict(test="test", func=lambda x: x),
            injection_parameters=dict(x=0.5, y=0.5),
            meta_data=dict(test="test"),
            sampling_time=100.0,
        )

        n = 100
        posterior = pd.DataFrame(
            dict(x=np.random.normal(0, 1, n), y=np.random.normal(0, 1, n))
        )
        result.posterior = posterior
        result.log_evidence = 10
        result.log_evidence_err = 11
        result.log_bayes_factor = 12
        result.log_noise_evidence = 13
        self.result = result
        pass

    def tearDown(self):
        bilby.utils.command_line_args.bilby_test_mode = True
        try:
            shutil.rmtree(self.outdir)
        except OSError:
            pass
        del self.result
        pass

    def test_result_file_name_default(self):
        outdir = "outdir"
        label = "label"
        self.assertEqual(
            bilby.core.result.result_file_name(outdir, label),
            "{}/{}_result.json".format(outdir, label),
        )

    def test_result_file_name_hdf5(self):
        outdir = "outdir"
        label = "label"
        self.assertEqual(
            bilby.core.result.result_file_name(outdir, label, extension="hdf5"),
            "{}/{}_result.hdf5".format(outdir, label),
        )

    def test_result_file_name_pkl(self):
        outdir = "outdir"
        label = "label"
        self.assertEqual(
            bilby.core.result.result_file_name(outdir, label, extension="pkl"),
            "{}/{}_result.pkl".format(outdir, label),
        )

    def test_result_file_name_pickle(self):
        outdir = "outdir"
        label = "label"
        self.assertEqual(
            bilby.core.result.result_file_name(outdir, label, extension="pickle"),
            "{}/{}_result.pkl".format(outdir, label),
        )

    def test_fail_save_and_load(self):
        with self.assertRaises(ValueError):
            bilby.core.result.read_in_result()

        with self.assertRaises(ValueError):
            bilby.core.result.read_in_result(filename="no_file_extension")

        with self.assertRaises(IOError):
            bilby.core.result.read_in_result(filename="not/a/file.json")

        with self.assertRaises(IOError):
            incomplete_json = """
{
  "label": "label",
  "outdir": "outdir",
  "sampler": "dynesty",
  "log_evidence": 0,
  "log_evidence_err": 0,
  "log_noise_evidence": 0,
  "log_bayes_factor": 0,
  "priors": {
    "chirp_mass": {
"""
            with open("{}/incomplete.json".format(self.result.outdir), "wb") as ff:
                ff.write(incomplete_json)
            bilby.core.result.read_in_result(
                filename="{}/incomplete.json".format(self.result.outdir)
            )

    def test_unset_priors(self):
        result = bilby.core.result.Result(
            label="label",
            outdir="outdir",
            sampler="nestle",
            search_parameter_keys=["x", "y"],
            fixed_parameter_keys=["c", "d"],
            priors=None,
            sampler_kwargs=dict(test="test"),
            injection_parameters=dict(x=0.5, y=0.5),
            meta_data=dict(test="test"),
        )
        with self.assertRaises(ValueError):
            _ = result.priors
        self.assertEqual(result.parameter_labels, result.search_parameter_keys)
        self.assertEqual(
            result.parameter_labels_with_unit, result.search_parameter_keys
        )

    def test_unknown_priors_fail(self):
        with self.assertRaises(ValueError):
            bilby.core.result.Result(
                label="label",
                outdir="outdir",
                sampler="nestle",
                search_parameter_keys=["x", "y"],
                fixed_parameter_keys=["c", "d"],
                priors=["a", "b"],
                sampler_kwargs=dict(test="test"),
                injection_parameters=dict(x=0.5, y=0.5),
                meta_data=dict(test="test"),
            )

    def test_set_samples(self):
        samples = [1, 2, 3]
        self.result.samples = samples
        self.assertEqual(samples, self.result.samples)

    def test_set_nested_samples(self):
        nested_samples = [1, 2, 3]
        self.result.nested_samples = nested_samples
        self.assertEqual(nested_samples, self.result.nested_samples)

    def test_set_walkers(self):
        walkers = [1, 2, 3]
        self.result.walkers = walkers
        self.assertEqual(walkers, self.result.walkers)

    def test_set_nburn(self):
        nburn = 1
        self.result.nburn = nburn
        self.assertEqual(nburn, self.result.nburn)

    def test_unset_posterior(self):
        self.result.posterior = None
        with self.assertRaises(ValueError):
            _ = self.result.posterior

    def test_save_and_load_json(self):
        self._save_and_load_test(extension='json')

    def test_save_and_load_json_gzip(self):
        self._save_and_load_test(extension='json', gzip=True)

    def test_save_and_load_pkl(self):
        self._save_and_load_test(extension='pkl')

    def test_save_and_load_hdf5(self):
        self._save_and_load_test(extension='hdf5')

    def _save_and_load_test(self, extension, gzip=False):
        self.result.save_to_file(extension=extension, gzip=gzip)
        loaded_result = bilby.core.result.read_in_result(
            outdir=self.result.outdir, label=self.result.label, extension=extension, gzip=gzip
        )
        self.assertTrue(
            np.array_equal(
                self.result.posterior.sort_values(by=["x"]),
                loaded_result.posterior.sort_values(by=["x"]),
            )
        )
        self.assertTrue(
            self.result.fixed_parameter_keys == loaded_result.fixed_parameter_keys
        )
        self.assertTrue(
            self.result.search_parameter_keys == loaded_result.search_parameter_keys
        )
        self.assertEqual(self.result.meta_data, loaded_result.meta_data)
        self.assertEqual(
            self.result.injection_parameters, loaded_result.injection_parameters
        )
        self.assertEqual(self.result.log_evidence, loaded_result.log_evidence)
        self.assertEqual(
            self.result.log_noise_evidence, loaded_result.log_noise_evidence
        )
        self.assertEqual(self.result.log_evidence_err, loaded_result.log_evidence_err)
        self.assertEqual(self.result.log_bayes_factor, loaded_result.log_bayes_factor)
        self.assertEqual(self.result.priors["x"], loaded_result.priors["x"])
        self.assertEqual(self.result.priors["y"], loaded_result.priors["y"])
        self.assertEqual(self.result.priors["c"], loaded_result.priors["c"])
        self.assertEqual(self.result.priors["d"], loaded_result.priors["d"])
        self.assertEqual(self.result.sampling_time, loaded_result.sampling_time)

    def test_save_and_dont_overwrite_json(self):
        self._save_and_dont_overwrite_test(extension='json')

    def test_save_and_dont_overwrite_pkl(self):
        self._save_and_dont_overwrite_test(extension='pkl')

    def test_save_and_dont_overwrite_hdf5(self):
        self._save_and_dont_overwrite_test(extension='hdf5')

    def _save_and_dont_overwrite_test(self, extension):
        self.result.save_to_file(overwrite=False, extension=extension)
        self.result.save_to_file(overwrite=False, extension=extension)
        self.assertTrue(os.path.isfile(f"{self.result.outdir}/{self.result.label}_result.{extension}.old"))

    def test_save_and_overwrite_json(self):
        self._save_and_overwrite_test(extension='json')

    def test_save_and_overwrite_pkl(self):
        self._save_and_overwrite_test(extension='pkl')

    def test_save_and_overwrite_hdf5(self):
        self._save_and_overwrite_test(extension='hdf5')

    def _save_and_overwrite_test(self, extension):
        self.result.save_to_file(overwrite=True, extension=extension)
        self.result.save_to_file(overwrite=True, extension=extension)
        self.assertFalse(os.path.isfile(f"{self.result.outdir}/{self.result.label}_result.{extension}.old"))

    def test_save_samples(self):
        self.result.save_posterior_samples()
        filename = "{}/{}_posterior_samples.dat".format(
            self.result.outdir, self.result.label
        )
        self.assertTrue(os.path.isfile(filename))
        df = pd.read_csv(filename, sep=" ")
        self.assertTrue(np.allclose(self.result.posterior.values, df.values))

    def test_save_samples_from_filename(self):
        filename = "{}/{}_posterior_samples_OTHER.dat".format(
            self.result.outdir, self.result.label
        )
        self.result.save_posterior_samples(filename=filename)
        self.assertTrue(os.path.isfile(filename))
        df = pd.read_csv(filename, sep=" ")
        self.assertTrue(np.allclose(self.result.posterior.values, df.values))

    def test_save_samples_numpy_load(self):
        self.result.save_posterior_samples()
        filename = "{}/{}_posterior_samples.dat".format(
            self.result.outdir, self.result.label
        )
        self.assertTrue(os.path.isfile(filename))
        data = np.genfromtxt(filename, names=True)
        df = pd.read_csv(filename, sep=" ")
        self.assertTrue(len(data.dtype) == len(df.keys()))

    def test_samples_to_posterior_simple(self):
        self.result.posterior = None
        x = [1, 2, 3]
        y = [4, 6, 8]
        self.result.samples = np.array([x, y]).T
        self.result.samples_to_posterior()
        self.assertTrue(all(self.result.posterior["x"] == x))
        self.assertTrue(all(self.result.posterior["y"] == y))
        self.assertTrue(np.all(None == self.result.posterior.log_likelihood.values))  # noqa

    def test_samples_to_posterior(self):
        self.result.posterior = None
        x = [1, 2, 3]
        y = [4, 6, 8]
        log_likelihood = np.array([6, 7, 8])
        self.result.samples = np.array([x, y]).T
        self.result.log_likelihood_evaluations = log_likelihood
        self.result.samples_to_posterior(priors=self.result.priors)
        self.assertTrue(all(self.result.posterior["x"] == x))
        self.assertTrue(all(self.result.posterior["y"] == y))
        self.assertTrue(
            np.array_equal(self.result.posterior.log_likelihood.values, log_likelihood)
        )
        self.assertTrue(
            all(self.result.posterior.c.values == self.result.priors["c"].peak)
        )
        self.assertTrue(
            all(self.result.posterior.d.values == self.result.priors["d"].peak)
        )

    def test_calculate_prior_values(self):
        self.result.calculate_prior_values(priors=self.result.priors)
        self.assertEqual(len(self.result.posterior), len(self.result.prior_values))

    def test_plot_multiple(self):
        filename = "{}/multiple.png".format(self.result.outdir)
        bilby.core.result.plot_multiple([self.result, self.result], filename=filename)
        self.assertTrue(os.path.isfile(filename))
        os.remove(filename)

    def test_plot_walkers(self):
        self.result.walkers = np.random.uniform(0, 1, (10, 11, 2))
        self.result.nburn = 5
        self.result.plot_walkers()
        self.assertTrue(
            os.path.isfile(
                "{}/{}_walkers.png".format(self.result.outdir, self.result.label)
            )
        )

    def test_plot_with_data(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)

        def model(xx, theta):
            return xx

        self.result.posterior = pd.DataFrame(dict(theta=[1, 2, 3]))
        self.result.plot_with_data(model, x, y, ndraws=10)
        self.assertTrue(
            os.path.isfile(
                "{}/{}_plot_with_data.png".format(self.result.outdir, self.result.label)
            )
        )
        self.result.posterior["log_likelihood"] = np.random.uniform(
            0, 1, len(self.result.posterior)
        )
        self.result.plot_with_data(model, x, y, ndraws=10, xlabel="a", ylabel="y")

    def test_plot_corner(self):
        self.result.injection_parameters = dict(x=0.8, y=1.1)
        self.result.plot_corner()
        self.result.plot_corner(parameters=["x", "y"])
        self.result.plot_corner(parameters=["x", "y"], truths=[1, 1])
        self.result.plot_corner(parameters=dict(x=1, y=1))
        self.result.plot_corner(truths=dict(x=1, y=1))
        self.result.plot_corner(truth=dict(x=1, y=1))
        self.result.plot_corner(truths=None)
        self.result.plot_corner(truths=False)
        self.result.plot_corner(truths=True)
        with self.assertRaises(ValueError):
            self.result.plot_corner(truths=dict(x=1, y=1), parameters=dict(x=1, y=1))
        with self.assertRaises(ValueError):
            self.result.plot_corner(truths=[1, 1], parameters=dict(x=1, y=1))
        with self.assertRaises(ValueError):
            self.result.plot_corner(parameters=["x", "y"], truths=dict(x=1, y=1))

    def test_plot_corner_with_injection_parameters(self):
        self.result.plot_corner()
        self.result.plot_corner(parameters=["x", "y"])
        self.result.plot_corner(parameters=["x", "y"], truths=[1, 1])
        self.result.plot_corner(parameters=dict(x=1, y=1))

    def test_plot_corner_with_priors(self):
        priors = bilby.core.prior.PriorDict()
        priors["x"] = bilby.core.prior.Uniform(-1, 1, "x")
        priors["y"] = bilby.core.prior.Uniform(-1, 1, "y")
        self.result.plot_corner(priors=priors)
        self.result.priors = priors
        self.result.plot_corner(priors=True)
        with self.assertRaises(ValueError):
            self.result.plot_corner(priors="test")

    def test_get_credible_levels(self):
        levels = self.result.get_all_injection_credible_levels()
        self.assertDictEqual(levels, dict(x=0.68, y=0.72))

    def test_get_credible_levels_raises_error_if_no_injection_parameters(self):
        self.result.injection_parameters = None
        with self.assertRaises(TypeError):
            self.result.get_all_injection_credible_levels()

    def test_kde(self):
        kde = self.result.kde
        import scipy.stats

        self.assertEqual(type(kde), scipy.stats.kde.gaussian_kde)
        self.assertEqual(kde.d, 2)

    def test_posterior_probability(self):
        sample = dict(x=0, y=0.1)
        self.assertTrue(
            isinstance(self.result.posterior_probability(sample), np.ndarray)
        )
        self.assertTrue(len(self.result.posterior_probability(sample)), 1)
        self.assertEqual(
            self.result.posterior_probability(sample)[0], self.result.kde([0, 0.1])
        )

    def test_multiple_posterior_probability(self):
        sample = [dict(x=0, y=0.1), dict(x=0.8, y=0)]
        self.assertTrue(
            isinstance(self.result.posterior_probability(sample), np.ndarray)
        )
        self.assertTrue(
            np.array_equal(
                self.result.posterior_probability(sample),
                self.result.kde([[0, 0.1], [0.8, 0]]),
            )
        )

    def test_to_arviz(self):
        with self.assertRaises(TypeError):
            self.result.to_arviz(prior=dict())

        Nprior = 100

        log_likelihood = np.random.rand(len(self.result.posterior))
        self.result.log_likelihood_evaluations = log_likelihood

        az = self.result.to_arviz(prior=Nprior)

        self.assertTrue("x" in az.posterior and "y" in az.posterior)
        for var in ["x", "y"]:
            self.assertTrue(np.array_equal(az.posterior[var].values.squeeze(),
                                           self.result.posterior[var].values))
            self.assertTrue(len(az.prior[var][0]) == Nprior)

        self.assertTrue(np.array_equal(az.log_likelihood["log_likelihood"].values.squeeze(),
                                       log_likelihood))

        self.assertTrue(
            az.posterior.attrs["inference_library"] == "bilby: {}".format(
                self.result.sampler
            )
        )
        self.assertTrue(
            az.posterior.attrs["inference_library_version"]
            == bilby.utils.get_version_information()
        )

        # add log likelihood to samples and extract from there
        del az
        self.result.posterior["log_likelihood"] = log_likelihood
        az = self.result.to_arviz()
        self.assertTrue(np.array_equal(az.log_likelihood["log_likelihood"].values.squeeze(),
                                       log_likelihood))

    @patch("builtins.__import__")
    def test_to_arviz_not_installed(self, mock_import):

        def import_side_effect(name, *args):
            if name == "arviz":
                raise ImportError
            return __import__(name, *args)

        mock_import.side_effect = import_side_effect

        with self.assertRaises(ResultError) as excinfo:
            self.result.to_arviz()

        self.assertEqual(
            str(excinfo.exception),
            "ArviZ is not installed, so cannot convert to InferenceData."
        )

    def test_result_caching(self):

        class SimpleLikelihood(bilby.Likelihood):
            def __init__(self):
                super().__init__(parameters={"x": None})

            def log_likelihood(self):
                return -self.parameters["x"]**2

        likelihood = SimpleLikelihood()
        priors = dict(x=bilby.core.prior.Uniform(-5, 5, "x"))

        # Trivial subclass of Result

        class NotAResult(bilby.core.result.Result):
            pass

        result = bilby.run_sampler(
            likelihood, priors, sampler='bilby_mcmc', nsamples=10, L1steps=1,
            proposal_cycle="default_noGMnoKD", printdt=1,
            check_point_plot=False,
            result_class=NotAResult)
        # result should be specified result_class
        assert isinstance(result, NotAResult)

        cached_result = bilby.run_sampler(
            likelihood, priors, sampler='bilby_mcmc', nsamples=10, L1steps=1,
            proposal_cycle="default_noGMnoKD", printdt=1,
            check_point_plot=False,
            result_class=NotAResult)

        # so should a result loaded from cache
        assert isinstance(cached_result, NotAResult)


class TestResultListError(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        bilby.utils.command_line_args.bilby_test_mode = False
        self.priors = bilby.prior.PriorDict(
            dict(
                x=bilby.prior.Uniform(0, 1, "x", latex_label="$x$", unit="s"),
                y=bilby.prior.Uniform(0, 1, "y", latex_label="$y$", unit="m"),
                c=1,
                d=2,
            )
        )

        # create two cpnest results
        self.nested_results = bilby.result.ResultList([])
        self.mcmc_results = bilby.result.ResultList([])
        self.expected_nested = []

        self.outdir = "outdir"
        self.label = "label"
        n = 100
        for i in range(2):
            result = bilby.core.result.Result(
                label=self.label + str(i),
                outdir=self.outdir,
                sampler="cpnest",
                search_parameter_keys=["x", "y"],
                fixed_parameter_keys=["c", "d"],
                priors=self.priors,
                sampler_kwargs=dict(test="test", func=lambda x: x, nlive=10),
                injection_parameters=dict(x=0.5, y=0.5),
                meta_data=dict(test="test"),
            )

            posterior = pd.DataFrame(
                dict(
                    x=np.random.normal(0, 1, n),
                    y=np.random.normal(0, 1, n),
                    log_likelihood=sorted(np.random.normal(0, 1, n)),
                )
            )
            result.posterior = posterior[-10:]  # use last 10 samples as posterior
            result.nested_samples = posterior
            result.log_evidence = 10
            result.log_evidence_err = 11
            result.log_bayes_factor = 12
            result.log_noise_evidence = 13
            self.nested_results.append(result)
            self.expected_nested.append(result)
        for i in range(2):
            result = bilby.core.result.Result(
                label=self.label + str(i) + "mcmc",
                outdir=self.outdir,
                sampler="emcee",
                search_parameter_keys=["x", "y"],
                fixed_parameter_keys=["c", "d"],
                priors=self.priors,
                sampler_kwargs=dict(test="test", func=lambda x: x, nlive=10),
                injection_parameters=dict(x=0.5, y=0.5),
                meta_data=dict(test="test"),
            )

            posterior = pd.DataFrame(
                dict(
                    x=np.random.normal(0, 1, n),
                    y=np.random.normal(0, 1, n),
                    log_likelihood=sorted(np.random.normal(0, 1, n)),
                )
            )
            result.posterior = posterior[-10:]
            result.log_evidence = 10
            result.log_evidence_err = 11
            result.log_bayes_factor = 12
            result.log_noise_evidence = 13
            self.mcmc_results.append(result)
        for res in self.nested_results:
            res.save_to_file()

    def tearDown(self):
        bilby.utils.command_line_args.bilby_test_mode = True
        try:
            shutil.rmtree(self.nested_results[0].outdir)
        except OSError:
            pass

        del self.nested_results
        del self.label
        del self.outdir
        del self.expected_nested
        del self.mcmc_results
        del self.priors

    def test_append_illegal_type(self):
        with self.assertRaises(TypeError):
            _ = bilby.core.result.ResultList(1)

    def test_append_from_string(self):
        self.nested_results.append(self.outdir + "/" + self.label + "1_result.json")
        self.assertEqual(3, len(self.nested_results))

    def test_append_result_type(self):
        self.nested_results.append(self.nested_results[1])
        self.expected_nested.append(self.expected_nested[1])
        self.assertListEqual(self.nested_results, self.nested_results)

    def test_combine_inconsistent_samplers(self):
        self.nested_results[0].sampler = "dynesty"
        with self.assertRaises(bilby.result.ResultListError):
            self.nested_results.combine()

    def test_combine_inconsistent_priors_length(self):
        self.nested_results[0].priors = bilby.prior.PriorDict(
            dict(
                x=bilby.prior.Uniform(0, 1, "x", latex_label="$x$", unit="s"),
                y=bilby.prior.Uniform(0, 1, "y", latex_label="$y$", unit="m"),
                c=1,
            )
        )
        with self.assertRaises(bilby.result.ResultListError):
            self.nested_results.combine()

    def test_combine_inconsistent_priors_types(self):
        self.nested_results[0].priors = bilby.prior.PriorDict(
            dict(
                x=bilby.prior.Uniform(0, 1, "x", latex_label="$x$", unit="s"),
                y=bilby.prior.Uniform(0, 1, "y", latex_label="$y$", unit="m"),
                c=1,
                d=bilby.core.prior.Cosine(),
            )
        )
        with self.assertRaises(bilby.result.ResultListError):
            self.nested_results.combine()

    def test_combine_inconsistent_search_parameters(self):
        self.nested_results[0].search_parameter_keys = ["y"]
        with self.assertRaises(bilby.result.ResultListError):
            self.nested_results.combine()

    def test_combine_inconsistent_data(self):
        self.nested_results[0].log_noise_evidence = -7
        with self.assertRaises(bilby.result.ResultListError):
            self.nested_results.combine()

    def test_combine_data_all_nan_consistent(self):
        self.nested_results[0].log_noise_evidence = np.nan
        self.nested_results[1].log_noise_evidence = np.nan
        self.nested_results.combine()

    def test_combine_inconsistent_data_one_nan(self):
        self.nested_results[0].log_noise_evidence = np.nan
        with self.assertRaises(bilby.result.ResultListError):
            self.nested_results.combine()

    def test_combine_inconsistent_sampling_data(self):
        result = bilby.core.result.Result(
            label=self.label,
            outdir=self.outdir,
            sampler="cpnest",
            search_parameter_keys=["x", "y"],
            fixed_parameter_keys=["c", "d"],
            priors=self.priors,
            sampler_kwargs=dict(test="test", func=lambda x: x, nlive=10),
            injection_parameters=dict(x=0.5, y=0.5),
            meta_data=dict(test="test"),
        )

        posterior = pd.DataFrame(
            dict(
                x=np.random.normal(0, 1, 100),
                y=np.random.normal(0, 1, 100),
                log_likelihood=sorted(np.random.normal(0, 1, 100)),
            )
        )
        result.posterior = posterior[-10:]  # use last 10 samples as posterior
        result.log_evidence = 10
        result.log_evidence_err = 11
        result.log_bayes_factor = 12
        result.log_noise_evidence = 13
        result._nested_samples = None
        self.nested_results.append(result)
        with self.assertRaises(bilby.result.ResultListError):
            self.nested_results.combine()


class TestMiscResults(unittest.TestCase):
    def test_sanity_check_labels(self):
        labels = ["a", "$a$", "a_1", "$a_1$"]
        labels_checked = bilby.core.result.sanity_check_labels(labels)
        self.assertEqual(labels_checked, ["a", "$a$", "a-1", "$a_1$"])


class TestPPPlots(unittest.TestCase):

    def setUp(self):
        priors = bilby.core.prior.PriorDict(dict(
            a=bilby.core.prior.Uniform(0, 1, latex_label="$a$"),
            b=bilby.core.prior.Uniform(0, 1, latex_label="$b$"),
        ))
        self.results = [
            bilby.core.result.Result(
                label=str(ii),
                outdir='.',
                search_parameter_keys=list(priors.keys()),
                priors=priors,
                injection_parameters=priors.sample(),
                posterior=pd.DataFrame(priors.sample(500)),
            )
            for ii in range(10)
        ]

    def test_make_pp_plot(self):
        _ = bilby.core.result.make_pp_plot(self.results, save=False)

    def test_pp_plot_raises_error_with_wrong_number_of_lines(self):
        with self.assertRaises(ValueError):
            _ = bilby.core.result.make_pp_plot(self.results, save=False, lines=["-"])

    def test_pp_plot_raises_error_with_wrong_number_of_confidence_intervals(self):
        with self.assertRaises(ValueError):
            _ = bilby.core.result.make_pp_plot(
                self.results, save=False, confidence_interval_alpha=[0.1]
            )


class SimpleGaussianLikelihood(bilby.core.likelihood.Likelihood):
    def __init__(self, mean=0, sigma=1):
        """
        A very simple Gaussian likelihood for testing
        """
        from scipy.stats import norm
        super().__init__(parameters=dict())
        self.mean = mean
        self.sigma = sigma
        self.dist = norm(loc=mean, scale=sigma)

    def log_likelihood(self):
        return self.dist.logpdf(self.parameters["mu"])


class TestReweight(unittest.TestCase):

    def setUp(self):
        self.priors = bilby.core.prior.PriorDict(dict(
            mu=bilby.core.prior.TruncatedNormal(0, 1, minimum=-5, maximum=5),
        ))
        self.result = bilby.core.result.Result(
            search_parameter_keys=list(self.priors.keys()),
            priors=self.priors,
            posterior=pd.DataFrame(self.priors.sample(1000)),
            log_evidence=-np.log(10),
        )

    def _run_reweighting(self, sigma):
        likelihood_1 = SimpleGaussianLikelihood()
        likelihood_2 = SimpleGaussianLikelihood(sigma=sigma)
        original_ln_likelihoods = list()
        for ii in range(len(self.result.posterior)):
            likelihood_1.parameters = self.result.posterior.iloc[ii]
            original_ln_likelihoods.append(likelihood_1.log_likelihood())
        self.result.posterior["log_prior"] = self.priors.ln_prob(self.result.posterior)
        self.result.posterior["log_likelihood"] = original_ln_likelihoods
        self.original_ln_likelihoods = original_ln_likelihoods
        return bilby.core.result.reweight(
            self.result, likelihood_1, likelihood_2, verbose_output=True
        )

    def test_reweight_same_likelihood_weights_1(self):
        """
        When the likelihoods are the same, the weights should be 1.
        """
        _, weights, _, _, _, _ = self._run_reweighting(sigma=1)
        self.assertLess(min(abs(weights - 1)), 1e-10)

    def test_reweight_different_likelihood_weights_correct(self):
        """
        Test the known case where the target likelihood is a Gaussian with
        sigma=0.5. The weights can be calculated analytically and the evidence
        should be close to the original evidence within statistical error.
        """
        from scipy.stats import norm
        new, weights, _, _, _, _ = self._run_reweighting(sigma=0.5)
        expected_weights = (
            norm(0, 0.5).pdf(self.result.posterior["mu"])
            / norm(0, 1).pdf(self.result.posterior["mu"])
        )
        self.assertLess(min(abs(weights - expected_weights)), 1e-10)
        self.assertLess(abs(new.log_evidence - self.result.log_evidence), 0.05)
        self.assertNotEqual(new.log_evidence, self.result.log_evidence)


if __name__ == "__main__":
    unittest.main()
