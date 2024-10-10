import os
import unittest
from parameterized import parameterized

import numpy as np
from bilby.core.prior import PriorDict
from bilby.gw import calibration, detector, prior


class TestBaseClass(unittest.TestCase):
    def setUp(self):
        self.model = calibration.Recalibrate()

    def tearDown(self):
        del self.model

    def test_repr(self):
        expected = "Recalibrate(prefix={})".format("'recalib_'")
        actual = repr(self.model)
        self.assertEqual(expected, actual)

    def test_calibration_factor(self):
        frequency_array = np.linspace(20, 1024, 1000)
        cal_factor = self.model.get_calibration_factor(frequency_array)
        assert np.alltrue(cal_factor.real == np.ones_like(frequency_array))


class TestCubicSpline(unittest.TestCase):
    def setUp(self):
        self.prefix = "recalib_"
        self.minimum_frequency = 20
        self.maximum_frequency = 1024
        self.n_points = 5
        self.model = calibration.CubicSpline(
            prefix=self.prefix,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            n_points=self.n_points,
        )
        self.parameters = {
            "recalib_{}_{}".format(param, ii): 0.0
            for ii in range(5)
            for param in ["amplitude", "phase"]
        }

    def tearDown(self):
        del self.prefix
        del self.minimum_frequency
        del self.maximum_frequency
        del self.n_points
        del self.model
        del self.parameters

    def test_calibration_factor(self):
        frequency_array = np.linspace(20, 1024, 1000)
        cal_factor = self.model.get_calibration_factor(
            frequency_array, **self.parameters
        )
        assert np.alltrue(cal_factor.real == np.ones_like(frequency_array))

    def test_repr(self):
        expected = "CubicSpline(prefix='{}', minimum_frequency={}, maximum_frequency={}, n_points={})".format(
            self.prefix, self.minimum_frequency, self.maximum_frequency, self.n_points
        )
        actual = repr(self.model)
        self.assertEqual(expected, actual)


class TestCubicSplineRequiresFourNodes(unittest.TestCase):
    def test_cannot_instantiate_with_too_few_nodes(self):
        for ii in range(6):
            if ii < 4:
                with self.assertRaises(ValueError):
                    calibration.CubicSpline("test", 1, 10, ii)
            else:
                calibration.CubicSpline("test", 1, 10, ii)


class TestReadWriteCalibrationDraws(unittest.TestCase):
    def setUp(self):
        self.filename = "calibration_draws.h5"
        self.number_of_draws = 100
        self.ifo = detector.get_empty_interferometer("H1")
        self.ifo.set_strain_data_from_power_spectral_density(
            duration=1, sampling_frequency=512
        )
        self.ifo.calibration_model = calibration.CubicSpline(
            prefix="recalib_H1_",
            n_points=10,
            minimum_frequency=self.ifo.minimum_frequency,
            maximum_frequency=self.ifo.maximum_frequency,
        )
        self.priors = prior.CalibrationPriorDict.constant_uncertainty_spline(
            amplitude_sigma=0.1,
            phase_sigma=0.1,
            minimum_frequency=self.ifo.minimum_frequency,
            maximum_frequency=self.ifo.maximum_frequency,
            n_nodes=10,
            label="H1",
        )

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_generate_draws(self):
        draws, parameters = calibration._generate_calibration_draws(
            self.ifo, self.priors, self.number_of_draws
        )
        self.assertEqual(draws.shape, (self.number_of_draws, sum(self.ifo.frequency_mask)))
        self.assertListEqual(list(self.priors.keys()), list(parameters.keys()))

    @parameterized.expand([("template",), ("data",), (None,)])
    def test_read_write_matches(self, correction):
        draws, parameters = calibration._generate_calibration_draws(
            self.ifo, self.priors, self.number_of_draws
        )
        frequencies = self.ifo.frequency_array[self.ifo.frequency_mask]
        calibration.write_calibration_file(
            filename=self.filename,
            frequency_array=frequencies,
            calibration_draws=draws,
            calibration_parameter_draws=parameters,
            correction=correction,
        )
        self.assertTrue(os.path.exists(self.filename))
        loaded_draws, loaded_parameters = calibration.read_calibration_file(
            filename=self.filename,
            frequency_array=frequencies,
            number_of_response_curves=self.number_of_draws,
            correction=correction,
        )
        self.assertLess(np.max(np.abs(loaded_draws - draws)), 1e-13)
        self.assertTrue(parameters.equals(loaded_parameters))

    def test_build_calibration_lookup(self):
        ifos = detector.InterferometerList(["H1", "L1", "H1"])
        ifos.set_strain_data_from_power_spectral_densities(
            duration=4, sampling_frequency=1024
        )
        priors = PriorDict()
        for ifo in ifos:
            ifo.minimum_frequency = 20
            ifo.maximum_frequency = 1024
            priors.update(prior.CalibrationPriorDict.constant_uncertainty_spline(
                amplitude_sigma=0.1,
                phase_sigma=0.1,
                minimum_frequency=ifo.minimum_frequency,
                maximum_frequency=ifo.maximum_frequency,
                n_nodes=10,
                label=ifo.name,
            ))
            ifo.calibration_model = calibration.CubicSpline(
                prefix=f"recalib_{ifo.name}_",
                n_points=10,
                minimum_frequency=ifo.minimum_frequency,
                maximum_frequency=ifo.maximum_frequency,
            )
        draws, parameters = calibration.build_calibration_lookup(
            interferometers=ifos,
            priors=priors,
            number_of_response_curves=10,
        )
        parameters
        for ifo in ifos[:2]:
            self.assertIn(ifo.name, draws)
            self.assertIn(ifo.name, parameters)
            filename = f"{ifo.name}_calibration_file.h5"
            self.assertTrue(os.path.exists(filename))
            os.remove(filename)


if __name__ == "__main__":
    unittest.main()
