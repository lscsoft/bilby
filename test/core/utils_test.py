import unittest
import os

import array_api_compat as aac
import array_api_extra as xpx
import dill
import numpy as np
from astropy import constants
import importlib
import logging
import matplotlib.pyplot as plt
import h5py
import json
import pytest
import sys
import types
import warnings

import bilby
from bilby.core import utils
from bilby.core.utils import global_meta_data
from bilby.core.utils.plotting import _close_new_figures


class TestConstants(unittest.TestCase):
    def test_speed_of_light(self):
        self.assertLess(
            abs(utils.speed_of_light - constants.c.value) / utils.speed_of_light, 1e-16
        )

    def test_parsec(self):
        self.assertLess(abs(utils.parsec - constants.pc.value) / utils.parsec, 1e-11)

    def test_solar_mass(self):
        self.assertLess(
            abs(utils.solar_mass - constants.M_sun.value) / utils.solar_mass, 1e-4
        )

    def test_radius_of_earth(self):
        self.assertLess(
            abs(utils.radius_of_earth - constants.R_earth.value)
            / utils.radius_of_earth,
            1e-5,
        )


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestFFT(unittest.TestCase):
    def setUp(self):
        self.sampling_frequency = self.xp.asarray(10)

    def tearDown(self):
        del self.sampling_frequency

    def test_nfft_sine_function(self):
        xp = self.xp
        injected_frequency = xp.asarray(2.7324)
        duration = xp.asarray(100)
        times = utils.create_time_series(xp.asarray(self.sampling_frequency), duration)

        time_domain_strain = xp.sin(2 * np.pi * times * injected_frequency + 0.4)

        frequency_domain_strain, frequencies = bilby.core.utils.nfft(
            time_domain_strain, self.sampling_frequency
        )
        frequency_at_peak = frequencies[xp.argmax(abs(frequency_domain_strain))]
        self.assertEqual(aac.get_namespace(frequency_at_peak), xp)
        frequency_at_peak = np.asarray(frequency_at_peak)
        injected_frequency = np.asarray(injected_frequency)
        self.assertAlmostEqual(injected_frequency, frequency_at_peak, places=1)

    def test_nfft_infft(self):
        xp = self.xp
        time_domain_strain = xp.asarray(np.random.normal(0, 1, 10))
        frequency_domain_strain, _ = bilby.core.utils.nfft(
            time_domain_strain, self.sampling_frequency
        )
        new_time_domain_strain = bilby.core.utils.infft(
            frequency_domain_strain, self.sampling_frequency
        )
        self.assertTrue(xp.allclose(time_domain_strain, new_time_domain_strain))


class TestInferParameters(unittest.TestCase):
    def setUp(self):
        def source_function(freqs, a, b, *args, **kwargs):
            return None

        class TestClass:
            def test_method(self, a, b, *args, **kwargs):
                pass

        class TestClass2:
            def test_method(self, freqs, a, b, *args, **kwargs):
                pass

        self.source1 = source_function
        test_obj = TestClass()
        self.source2 = test_obj.test_method
        test_obj2 = TestClass2()
        self.source3 = test_obj2.test_method

    def tearDown(self):
        del self.source1
        del self.source2

    def test_builtin_function(self):
        expected = [
            "mode",
            "buffering",
            "encoding",
            "errors",
            "newline",
            "closefd",
            "opener"
        ]
        actual = utils.infer_parameters_from_function(open)
        self.assertListEqual(expected, actual)

    def test_args_kwargs_handling(self):
        expected = ["a", "b"]
        actual = utils.infer_parameters_from_function(self.source1)
        self.assertListEqual(expected, actual)

    def test_self_handling(self):
        expected = ["a", "b"]
        actual = utils.infer_args_from_method(self.source2)
        self.assertListEqual(expected, actual)

    def test_self_handling_method_as_function(self):
        expected = ["a", "b"]
        actual = utils.infer_parameters_from_function(self.source3)
        self.assertListEqual(expected, actual)


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestTimeAndFrequencyArrays(unittest.TestCase):
    def setUp(self):
        self.start_time = self.xp.asarray(1.3)
        self.sampling_frequency = self.xp.asarray(5)
        self.duration = self.xp.asarray(1.6)
        self.frequency_array = utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.time_array = utils.create_time_series(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            starting_time=self.start_time,
        )

    def tearDown(self):
        del self.start_time
        del self.sampling_frequency
        del self.duration
        del self.frequency_array
        del self.time_array

    def test_create_time_array(self):
        expected_time_array = self.xp.asarray([1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7])
        time_array = utils.create_time_series(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            starting_time=self.start_time,
        )
        self.assertEqual(aac.get_namespace(time_array), self.xp)
        self.assertTrue(np.allclose(expected_time_array, time_array))

    def test_create_frequency_array(self):
        expected_frequency_array = np.array([0.0, 0.625, 1.25, 1.875, 2.5])
        frequency_array = utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.assertTrue(np.allclose(expected_frequency_array, frequency_array))

    def test_get_sampling_frequency_from_time_array(self):
        (
            new_sampling_freq,
            _,
        ) = utils.get_sampling_frequency_and_duration_from_time_array(self.time_array)
        self.assertEqual(self.sampling_frequency, new_sampling_freq)

    def test_get_sampling_frequency_from_time_array_unequally_sampled(self):
        self.time_array = xpx.at(self.time_array, -1).set(self.time_array[-1] + 0.0001)
        with self.assertRaises(ValueError):
            _, _ = utils.get_sampling_frequency_and_duration_from_time_array(
                self.time_array
            )

    def test_get_duration_from_time_array(self):
        _, new_duration = utils.get_sampling_frequency_and_duration_from_time_array(
            self.time_array
        )
        self.assertEqual(self.duration, new_duration)

    def test_get_start_time_from_time_array(self):
        new_start_time = self.time_array[0]
        self.assertEqual(self.start_time, new_start_time)

    def test_get_sampling_frequency_from_frequency_array(self):
        (
            new_sampling_freq,
            _,
        ) = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array
        )
        self.assertEqual(self.sampling_frequency, new_sampling_freq)

    def test_get_sampling_frequency_from_frequency_array_unequally_sampled(self):
        self.frequency_array = xpx.at(
            self.frequency_array, -1
        ).set(self.frequency_array[-1] + 0.0001)
        with self.assertRaises(ValueError):
            _, _ = utils.get_sampling_frequency_and_duration_from_frequency_array(
                self.frequency_array
            )

    def test_get_duration_from_frequency_array(self):
        (
            _,
            new_duration,
        ) = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array
        )
        self.assertEqual(self.duration, new_duration)

    def test_consistency_time_array_to_time_array(self):
        (
            new_sampling_frequency,
            new_duration,
        ) = utils.get_sampling_frequency_and_duration_from_time_array(self.time_array)
        new_start_time = self.time_array[0]
        new_time_array = utils.create_time_series(
            sampling_frequency=new_sampling_frequency,
            duration=new_duration,
            starting_time=new_start_time,
        )
        self.assertTrue(np.allclose(self.time_array, new_time_array))

    def test_consistency_frequency_array_to_frequency_array(self):
        (
            new_sampling_frequency,
            new_duration,
        ) = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array
        )
        new_frequency_array = utils.create_frequency_series(
            sampling_frequency=new_sampling_frequency, duration=new_duration
        )
        self.assertTrue(np.allclose(self.frequency_array, new_frequency_array))

    def test_illegal_sampling_frequency_and_duration(self):
        with self.assertRaises(utils.IllegalDurationAndSamplingFrequencyException):
            _ = utils.create_time_series(
                sampling_frequency=self.xp.asarray(7.7),
                duration=self.xp.asarray(1.3),
                starting_time=self.xp.asarray(0),
            )


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestReflect(unittest.TestCase):
    def test_in_range(self):
        xprime = self.xp.asarray([0.1, 0.5, 0.9])
        x = self.xp.asarray([0.1, 0.5, 0.9])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_one_to_two(self):
        xprime = self.xp.asarray([1.1, 1.5, 1.9])
        x = self.xp.asarray([0.9, 0.5, 0.1])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_two_to_three(self):
        xprime = self.xp.asarray([2.1, 2.5, 2.9])
        x = self.xp.asarray([0.1, 0.5, 0.9])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_minus_one_to_zero(self):
        xprime = self.xp.asarray([-0.9, -0.5, -0.1])
        x = self.xp.asarray([0.9, 0.5, 0.1])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)

    def test_in_minus_two_to_minus_one(self):
        xprime = self.xp.asarray([-1.9, -1.5, -1.1])
        x = self.xp.asarray([0.1, 0.5, 0.9])
        self.assertTrue(np.testing.assert_allclose(utils.reflect(xprime), x) is None)


class TestCloseNewFigures(unittest.TestCase):
    def setUp(self):
        self.existing_figure = plt.figure()

    def tearDown(self):
        plt.close(self.existing_figure)

    def test_closes_only_new_figures(self):
        with _close_new_figures():
            new_figure = plt.figure()

        self.assertTrue(plt.fignum_exists(self.existing_figure.number))
        self.assertFalse(plt.fignum_exists(new_figure.number))

    def test_closes_new_figures_after_error(self):
        with self.assertRaises(RuntimeError):
            with _close_new_figures():
                new_figure = plt.figure()
                raise RuntimeError

        self.assertTrue(plt.fignum_exists(self.existing_figure.number))
        self.assertFalse(plt.fignum_exists(new_figure.number))


class TestLatexPlotFormat(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 1)
        self.y = np.sin(self.x)
        self.filename = "test_plot.png"

    def tearDown(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def test_default(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot()
        self.assertTrue(os.path.isfile(self.filename))

    def test_mathedefault_one(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot(BILBY_MATHDEFAULT=1)
        self.assertTrue(os.path.isfile(self.filename))

    def test_mathedefault_zero(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot(BILBY_MATHDEFAULT=0)
        self.assertTrue(os.path.isfile(self.filename))

    def test_matplotlib_style(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot(BILBY_STYLE="fivethirtyeight")
        self.assertTrue(os.path.isfile(self.filename))

    def test_user_style(self):
        @bilby.core.utils.latex_plot_format
        def plot():
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            fig.savefig(self.filename)

        plot(BILBY_STYLE="test/test.mplstyle")
        self.assertTrue(os.path.isfile(self.filename))


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestUnsortedInterp2d(unittest.TestCase):
    def setUp(self):
        if aac.is_torch_namespace(self.xp):
            pytest.skip("Skipping Interp2d tests for torch backend")
        self.xx = np.linspace(0, 1, 10)
        self.yy = np.linspace(0, 1, 10)
        self.zz = np.random.random((10, 10))
        self.interpolant = bilby.core.utils.BoundedRectBivariateSpline(self.xx, self.yy, self.zz)

    def tearDown(self):
        pass

    def test_returns_float_for_floats(self):
        self.assertIsInstance(self.interpolant(0.5, 0.5), float)

    def test_returns_none_for_floats_outside_range(self):
        self.assertIsNone(self.interpolant(0.5, -0.5))
        self.assertIsNone(self.interpolant(-0.5, 0.5))

    def test_returns_float_for_float_and_array(self):
        input_array = self.xp.asarray(np.random.random(10))
        self.assertEqual(aac.get_namespace(self.interpolant(input_array, 0.5)), self.xp)
        self.assertEqual(aac.get_namespace(
            self.interpolant(input_array, input_array)), self.xp
        )
        self.assertEqual(aac.get_namespace(self.interpolant(0.5, input_array)), self.xp)

    def test_raises_for_mismatched_arrays(self):
        with self.assertRaises(ValueError):
            self.interpolant(
                self.xp.asarray(np.random.random(10)),
                self.xp.asarray(np.random.random(20)),
            )

    def test_returns_fill_in_correct_place(self):
        x_data = self.xp.asarray(np.random.random(10))
        y_data = self.xp.asarray(np.random.random(10))
        x_data = xpx.at(x_data, 3).set(-1)
        self.assertTrue(self.xp.isnan(self.interpolant(x_data, y_data)[3]))


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestTrapeziumRuleIntegration(unittest.TestCase):
    def setUp(self):
        self.x = self.xp.linspace(0, 1, 100)
        self.dxs = self.xp.diff(self.x)
        self.dx = self.dxs[0]
        with np.errstate(divide="ignore"):
            self.lnfunc1 = self.xp.log(self.x)
        self.func1int = (self.x[-1] ** 2 - self.x[0] ** 2) / 2
        with np.errstate(divide="ignore"):
            self.lnfunc2 = self.xp.log(self.x ** 2)
        self.func2int = (self.x[-1] ** 3 - self.x[0] ** 3) / 3

        self.irregularx = self.xp.asarray(
            [
                self.x[0],
                self.x[12],
                self.x[19],
                self.x[33],
                self.x[49],
                self.x[55],
                self.x[59],
                self.x[61],
                self.x[73],
                self.x[89],
                self.x[93],
                self.x[97],
                self.x[-1],
            ]
        )
        with np.errstate(divide="ignore"):
            self.lnfunc1irregular = self.xp.log(self.irregularx)
            self.lnfunc2irregular = self.xp.log(self.irregularx ** 2)
        self.irregulardxs = self.xp.diff(self.irregularx)

    def test_incorrect_step_type(self):
        with self.assertRaises(TypeError):
            utils.logtrapzexp(self.lnfunc1, "blah")

    def test_inconsistent_step_length(self):
        with self.assertRaises(ValueError):
            utils.logtrapzexp(self.lnfunc1, self.x[0 : len(self.x) // 2])

    def test_integral_func1(self):
        res1 = utils.logtrapzexp(self.lnfunc1, self.dx)
        res2 = utils.logtrapzexp(self.lnfunc1, self.dxs)

        self.assertTrue(np.abs(res1 - res2) < 1e-12)
        self.assertTrue(np.abs((self.xp.exp(res1) - self.func1int) / self.func1int) < 1e-12)

    def test_integral_func2(self):
        res = utils.logtrapzexp(self.lnfunc2, self.dxs)
        self.assertTrue(np.abs((self.xp.exp(res) - self.func2int) / self.func2int) < 1e-4)

    def test_integral_func1_irregular_steps(self):
        res = utils.logtrapzexp(self.lnfunc1irregular, self.irregulardxs)
        self.assertTrue(np.abs((self.xp.exp(res) - self.func1int) / self.func1int) < 1e-12)

    def test_integral_func2_irregular_steps(self):
        res = utils.logtrapzexp(self.lnfunc2irregular, self.irregulardxs)
        self.assertTrue(np.abs((self.xp.exp(res) - self.func2int) / self.func2int) < 1e-2)


class TestSavingNumpyRandomGenerator(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def init_outdir(self, tmp_path):
        # Use pytest's tmp_path fixture to create a temporary directory
        self.outdir = tmp_path / "test"
        self.outdir.mkdir()

    def setUp(self):
        self.filename = "test_random_state.npy"
        self.data = {
            "rng": np.random.default_rng(),
            "seed": 1234,
        }

    def test_hdf5(self):
        with h5py.File(self.outdir / "test.h5", "w") as f:
            bilby.core.utils.recursively_save_dict_contents_to_group(
                f, "/", self.data
            )
        a = self.data["rng"].random()

        with h5py.File(self.outdir / "test.h5", "r") as f:
            data = bilby.core.utils.recursively_load_dict_contents_from_group(f, "/")

        b = data["rng"].random()
        self.assertEqual(a, b)

    def test_json(self):
        with open(self.outdir / "test.json", 'w') as file:
            json.dump(self.data, file, indent=2, cls=bilby.core.utils.BilbyJsonEncoder)

        a = self.data["rng"].random()

        with open(self.outdir / "test.json", 'r') as file:
            data = json.load(file, object_hook=bilby.core.utils.decode_bilby_json)

        b = data["rng"].random()
        self.assertEqual(a, b)

    def test_hdf5_seed_sequence(self):
        seed_sequence = np.random.SeedSequence(1234).spawn(1)[0]
        data = {"seed_sequence": seed_sequence}

        with h5py.File(self.outdir / "test_seed_sequence.h5", "w") as f:
            bilby.core.utils.recursively_save_dict_contents_to_group(
                f, "/", data
            )

        with h5py.File(self.outdir / "test_seed_sequence.h5", "r") as f:
            loaded = bilby.core.utils.recursively_load_dict_contents_from_group(
                f, "/"
            )

        self.assertIsInstance(loaded["seed_sequence"], np.random.SeedSequence)
        self.assertEqual(loaded["seed_sequence"].state, seed_sequence.state)

    def test_json_seed_sequence(self):
        seed_sequence = np.random.SeedSequence(1234).spawn(1)[0]
        data = {"seed_sequence": seed_sequence}

        with open(self.outdir / "test_seed_sequence.json", "w") as file:
            json.dump(data, file, indent=2, cls=bilby.core.utils.BilbyJsonEncoder)

        with open(self.outdir / "test_seed_sequence.json", "r") as file:
            loaded = json.load(
                file, object_hook=bilby.core.utils.decode_bilby_json
            )

        self.assertIsInstance(loaded["seed_sequence"], np.random.SeedSequence)
        self.assertEqual(loaded["seed_sequence"].state, seed_sequence.state)

    def test_pickle(self):
        with open(self.outdir / "test.pkl", 'wb') as file:
            dill.dump(self.data, file)
        a = self.data["rng"].random()

        with open(self.outdir / "test.pkl", 'rb') as file:
            data = dill.load(file)
        b = data["rng"].random()
        self.assertEqual(a, b)


class TestGlobalMetaData(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def set_caplog(self, caplog):
        self._caplog = caplog

    def setUp(self):
        global_meta_data.clear()
        global_meta_data["rng"] = bilby.core.utils.random.rng
        bilby.gw.cosmology.DEFAULT_COSMOLOGY = None
        bilby.gw.cosmology.COSMOLOGY = [None, str(None)]

    def tearDown(self):
        global_meta_data.clear()
        global_meta_data["rng"] = bilby.core.utils.random.rng
        bilby.gw.cosmology.DEFAULT_COSMOLOGY = None
        bilby.gw.cosmology.COSMOLOGY = [None, str(None)]

    def test_set_item(self):
        global_meta_data["test"] = 123
        self.assertEqual(global_meta_data["test"], 123)

    def test_set_rng(self):
        bilby.core.utils.random.seed(1234)
        self.assertTrue(global_meta_data["rng"] is bilby.core.utils.random.rng)
        self.assertEqual(global_meta_data["seed"], 1234)

    @pytest.mark.gw
    @pytest.mark.requires("lal")
    def test_set_cosmology(self):
        bilby.gw.cosmology.set_cosmology("Planck15_LAL")
        self.assertTrue(global_meta_data["cosmology"] is bilby.gw.cosmology.COSMOLOGY[0])

    def test_update(self):
        bilby.core.utils.meta_data.logger.propagate = True
        with self._caplog.at_level(logging.DEBUG, logger="bilby"):
            global_meta_data.update({"test": 123})
        assert "Setting meta data key test with value 123" in str(self._caplog.text)

    def test_init(self):
        bilby.core.utils.meta_data.logger.propagate = True
        with self._caplog.at_level(logging.DEBUG, logger="bilby"):
            bilby.core.utils.GlobalMetaData({"test": 123})
        assert "Setting meta data key test with value 123" in str(self._caplog.text)
        assert "GlobalMetaData has already been instantiated" in str(self._caplog.text)


class TestRandomUtils(unittest.TestCase):

    def setUp(self):
        # Ensure a clean import of the random module
        if "bilby.core.utils.random" in sys.modules:
            importlib.reload(sys.modules["bilby.core.utils.random"])

    def tearDown(self):
        # Reload the module to restore default state after each test
        if "bilby.core.utils.random" in sys.modules:
            importlib.reload(sys.modules["bilby.core.utils.random"])

    def test_no_warning_when_accessed_via_module(self):
        from bilby.core.utils import random

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            random.seed(42)

        for warning in w:
            self.assertNotIn("Detected that `rng` was likely imported directly", str(warning.message))

    def test_warning_when_imported_directly(self):
        from bilby.core.utils.random import rng
        from bilby.core.utils import random

        # Simulate direct import of rng in a fake module
        fake_module = types.ModuleType("fake_module")
        fake_module.rng = rng
        sys.modules["fake_module"] = fake_module

        try:
            with self.assertWarnsRegex(RuntimeWarning, "Detected that `rng` was likely imported directly"):
                random.seed(123)
        finally:
            # always clean up the fake module even if the test fails
            del sys.modules["fake_module"]

    def test_rng_is_updated_after_seed(self):
        from bilby.core.utils import random

        old_rng = random.rng
        random.seed(123)
        new_rng = random.rng

        self.assertIsNot(old_rng, new_rng)


if __name__ == "__main__":
    unittest.main()
