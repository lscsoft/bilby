import unittest
from unittest import mock

import lal
import lalsimulation
import pytest
from shutil import rmtree

import numpy as np

import bilby


class TestInterferometer(unittest.TestCase):
    def setUp(self):
        self.name = "name"
        self.power_spectral_density = (
            bilby.gw.detector.PowerSpectralDensity.from_aligo()
        )
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.length = 30
        self.latitude = 1
        self.longitude = 2
        self.elevation = 3
        self.xarm_azimuth = 4
        self.yarm_azimuth = 5
        self.xarm_tilt = 0.0
        self.yarm_tilt = 0.0
        # noinspection PyTypeChecker
        self.ifo = bilby.gw.detector.Interferometer(
            name=self.name,
            power_spectral_density=self.power_spectral_density,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            length=self.length,
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            xarm_azimuth=self.xarm_azimuth,
            yarm_azimuth=self.yarm_azimuth,
            xarm_tilt=self.xarm_tilt,
            yarm_tilt=self.yarm_tilt,
        )
        self.ifo.strain_data.set_from_frequency_domain_strain(
            np.linspace(0, 4096, 4097), sampling_frequency=4096, duration=2
        )
        self.outdir = "outdir"

        self.injection_polarizations = dict()
        np.random.seed(42)
        self.injection_polarizations["plus"] = np.random.random(4097)
        self.injection_polarizations["cross"] = np.random.random(4097)

        self.waveform_generator = mock.MagicMock()
        self.wg_polarizations = dict(
            plus=np.random.random(4097), cross=np.random.random(4097)
        )
        self.waveform_generator.frequency_domain_strain = (
            lambda _: self.wg_polarizations
        )
        self.parameters = dict(
            ra=0.0, dec=0.0, geocent_time=0.0, psi=0.0,
            mass_1=100, mass_2=100
        )

        bilby.core.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

    def tearDown(self):
        del self.name
        del self.power_spectral_density
        del self.minimum_frequency
        del self.maximum_frequency
        del self.length
        del self.latitude
        del self.longitude
        del self.elevation
        del self.xarm_azimuth
        del self.yarm_azimuth
        del self.xarm_tilt
        del self.yarm_tilt
        del self.ifo
        del self.injection_polarizations
        del self.wg_polarizations
        del self.waveform_generator
        del self.parameters
        rmtree(self.outdir)

    def test_name_setting(self):
        self.assertEqual(self.ifo.name, self.name)

    def test_psd_setting(self):
        self.assertEqual(self.ifo.power_spectral_density, self.power_spectral_density)

    def test_min_freq_setting(self):
        self.assertEqual(self.ifo.strain_data.minimum_frequency, self.minimum_frequency)

    def test_max_freq_setting(self):
        self.assertEqual(self.ifo.strain_data.maximum_frequency, self.maximum_frequency)

    def test_get_detector_response_default_behaviour(self):
        self.ifo.antenna_response = mock.MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = mock.MagicMock(return_value=0)
        self.ifo.epoch = 0
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        # self.ifo.frequency_array = np.array([8, 12, 16, 20, 24])
        plus = np.linspace(0, 4096, 4097)
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0),
        )
        self.assertTrue(
            np.array_equal(response, plus * self.ifo.frequency_mask * np.exp(-0j))
        )

    def test_get_detector_response_with_dt(self):
        self.ifo.antenna_response = mock.MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = mock.MagicMock(return_value=0)
        self.ifo.epoch = 1
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        plus = np.linspace(0, 4096, 4097)
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0),
        )
        expected_response = (
            plus
            * self.ifo.frequency_mask
            * np.exp(-1j * 2 * np.pi * self.ifo.frequency_array)
        )
        self.assertTrue(np.allclose(abs(expected_response), abs(response)))

    def test_get_detector_response_multiple_modes(self):
        self.ifo.antenna_response = mock.MagicMock(return_value=1)
        self.ifo.time_delay_from_geocenter = mock.MagicMock(return_value=0)
        self.ifo.epoch = 0
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        plus = np.linspace(0, 4096, 4097)
        cross = np.linspace(0, 4096, 4097)
        response = self.ifo.get_detector_response(
            waveform_polarizations=dict(plus=plus, cross=cross),
            parameters=dict(ra=0, dec=0, geocent_time=0, psi=0),
        )
        self.assertTrue(
            np.array_equal(
                response, (plus + cross) * self.ifo.frequency_mask * np.exp(-0j)
            )
        )

    def test_inject_signal_from_waveform_polarizations_correct_injection(self):
        original_strain = self.ifo.strain_data.frequency_domain_strain
        self.ifo.get_detector_response = lambda x, params: x["plus"] + x["cross"]
        self.ifo.inject_signal_from_waveform_polarizations(
            parameters=self.parameters,
            injection_polarizations=self.injection_polarizations,
        )
        expected = (
            self.injection_polarizations["plus"]
            + self.injection_polarizations["cross"]
            + original_strain
        )
        self.assertTrue(
            np.array_equal(expected, self.ifo.strain_data._frequency_domain_strain)
        )

    def test_inject_signal_from_waveform_polarizations_update_time_domain_strain(self):
        original_td_strain = self.ifo.strain_data.time_domain_strain
        self.ifo.get_detector_response = lambda x, params: x["plus"] + x["cross"]
        self.ifo.inject_signal_from_waveform_polarizations(
            parameters=self.parameters,
            injection_polarizations=self.injection_polarizations,
        )
        self.assertFalse(
            np.array_equal(original_td_strain, self.ifo.strain_data.time_domain_strain)
        )

    def test_inject_signal_from_waveform_polarizations_meta_data(self):
        self.ifo.get_detector_response = lambda x, params: x["plus"] + x["cross"]
        self.ifo.inject_signal_from_waveform_polarizations(
            parameters=self.parameters,
            injection_polarizations=self.injection_polarizations,
        )
        signal_ifo_expected = (
            self.injection_polarizations["plus"] + self.injection_polarizations["cross"]
        )
        self.assertAlmostEqual(
            self.ifo.optimal_snr_squared(signal=signal_ifo_expected).real,
            self.ifo.meta_data["optimal_SNR"] ** 2,
            10,
        )
        self.assertAlmostEqual(
            self.ifo.matched_filter_snr(signal=signal_ifo_expected),
            self.ifo.meta_data["matched_filter_SNR"],
            10,
        )
        self.assertDictEqual(self.parameters, self.ifo.meta_data["parameters"])

    def test_inject_signal_from_waveform_polarizations_incorrect_length(self):
        self.injection_polarizations["plus"] = np.random.random(1000)
        self.injection_polarizations["cross"] = np.random.random(1000)
        self.ifo.get_detector_response = lambda x, params: x["plus"] + x["cross"]
        with self.assertRaises(ValueError):
            self.ifo.inject_signal_from_waveform_polarizations(
                parameters=self.parameters,
                injection_polarizations=self.injection_polarizations,
            )

    @mock.patch.object(bilby.core.utils.logger, "warning")
    def test_inject_signal_outside_segment_logs_warning(self, m):
        self.parameters["geocent_time"] = 24345.0
        self.ifo.get_detector_response = lambda x, params: x["plus"] + x["cross"]
        self.ifo.inject_signal_from_waveform_polarizations(
            parameters=self.parameters,
            injection_polarizations=self.injection_polarizations,
        )
        self.assertTrue(m.called)

    def test_inject_signal_from_waveform_generator_correct_return_value(self):
        self.ifo.get_detector_response = lambda x, params: x["plus"] + x["cross"]
        returned_polarizations = self.ifo.inject_signal_from_waveform_generator(
            parameters=self.parameters, waveform_generator=self.waveform_generator
        )
        self.assertTrue(
            np.array_equal(
                self.wg_polarizations["plus"], returned_polarizations["plus"]
            )
        )
        self.assertTrue(
            np.array_equal(
                self.wg_polarizations["cross"], returned_polarizations["cross"]
            )
        )

    @mock.patch.object(
        bilby.gw.detector.Interferometer, "inject_signal_from_waveform_generator"
    )
    def test_inject_signal_with_waveform_generator_correct_call(self, m):
        self.ifo.get_detector_response = lambda x, params: x["plus"] + x["cross"]
        _ = self.ifo.inject_signal(
            parameters=self.parameters, waveform_generator=self.waveform_generator
        )
        m.assert_called_with(
            parameters=self.parameters, waveform_generator=self.waveform_generator
        )

    def test_inject_signal_from_waveform_generator_correct_injection(self):
        original_strain = self.ifo.strain_data.frequency_domain_strain
        self.ifo.get_detector_response = lambda x, params: x["plus"] + x["cross"]
        injection_polarizations = self.ifo.inject_signal_from_waveform_generator(
            parameters=self.parameters, waveform_generator=self.waveform_generator
        )
        expected = (
            injection_polarizations["plus"]
            + injection_polarizations["cross"]
            + original_strain
        )
        self.assertTrue(
            np.array_equal(expected, self.ifo.strain_data._frequency_domain_strain)
        )

    def test_inject_signal_with_injection_polarizations(self):
        original_strain = self.ifo.strain_data.frequency_domain_strain
        self.ifo.get_detector_response = lambda x, params: x["plus"] + x["cross"]
        self.ifo.inject_signal(
            parameters=self.parameters,
            injection_polarizations=self.injection_polarizations,
        )
        expected = (
            self.injection_polarizations["plus"]
            + self.injection_polarizations["cross"]
            + original_strain
        )
        self.assertTrue(
            np.array_equal(expected, self.ifo.strain_data._frequency_domain_strain)
        )

    @mock.patch.object(
        bilby.gw.detector.Interferometer, "inject_signal_from_waveform_polarizations"
    )
    def test_inject_signal_with_injection_polarizations_and_waveform_generator(self, m):
        self.ifo.get_detector_response = lambda x, params: x["plus"] + x["cross"]
        _ = self.ifo.inject_signal(
            parameters=self.parameters,
            waveform_generator=self.waveform_generator,
            injection_polarizations=self.injection_polarizations,
        )
        m.assert_called_with(
            parameters=self.parameters,
            injection_polarizations=self.injection_polarizations,
        )
        with self.assertRaises(ValueError):
            m.assert_called_with(
                parameters=self.parameters,
                injection_polarizations=self.wg_polarizations,
            )

    def test_inject_signal_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.ifo.inject_signal(injection_polarizations=None, parameters=None)

    def test_optimal_snr_squared(self):
        """
        Merely checks parameters are given in the right order and the frequency
        mask is applied.
        """
        with mock.patch("bilby.gw.utils.noise_weighted_inner_product") as m:
            m.side_effect = lambda a, b, c, d: [a, b, c, d]
            signal = np.ones_like(self.ifo.power_spectral_density_array)
            mask = self.ifo.frequency_mask
            expected = [
                signal * mask,
                signal * mask,
                np.where(mask, self.ifo.power_spectral_density_array, np.inf),
                self.ifo.strain_data.duration,
            ]
            actual = self.ifo.optimal_snr_squared(signal=signal)
            self.assertTrue(np.array_equal(expected[0], actual[0]))
            self.assertTrue(np.array_equal(expected[1], actual[1]))
            self.assertTrue(np.array_equal(expected[2], actual[2]))
            self.assertEqual(expected[3], actual[3])

    def test_template_template_inner_product(self):
        signal_1 = np.ones_like(self.ifo.power_spectral_density_array)
        signal_2 = np.ones_like(self.ifo.power_spectral_density_array) * 2
        signal_1_optimal = self.ifo.optimal_snr_squared(signal=signal_1)
        signal_1_optimal_by_template_template = self.ifo.template_template_inner_product(
            signal_1=signal_1,
            signal_2=signal_1
        )
        self.assertAlmostEqual(signal_1_optimal, signal_1_optimal_by_template_template, delta=signal_1_optimal / 1e14)
        signal_1_signal_2_inner_product = self.ifo.template_template_inner_product(signal_1=signal_1, signal_2=signal_2)
        self.assertAlmostEqual(signal_1_optimal * 2, signal_1_signal_2_inner_product, delta=signal_1_optimal / 1e14)

    def test_repr(self):
        expected = (
            "Interferometer(name='{}', power_spectral_density={}, minimum_frequency={}, "
            "maximum_frequency={}, length={}, latitude={}, longitude={}, elevation={}, xarm_azimuth={}, "
            "yarm_azimuth={}, xarm_tilt={}, yarm_tilt={})".format(
                self.name,
                self.power_spectral_density,
                float(self.minimum_frequency),
                float(self.maximum_frequency),
                float(self.length),
                float(self.latitude),
                float(self.longitude),
                float(self.elevation),
                float(self.xarm_azimuth),
                float(self.yarm_azimuth),
                float(self.xarm_tilt),
                float(self.yarm_tilt),
            )
        )
        self.assertEqual(expected, repr(self.ifo))

    def test_to_and_from_pkl_loading(self):
        self.ifo.to_pickle(outdir="outdir", label="test")
        filename = "outdir/test.pkl"
        recovered_ifo = bilby.gw.detector.Interferometer.from_pickle(filename)
        self.assertEqual(self.ifo, recovered_ifo)

    def test_to_and_from_pkl_wrong_class(self):
        import dill
        with open("./outdir/psd.pkl", "wb") as ff:
            dill.dump(self.ifo.power_spectral_density, ff)
        filename = self.ifo._filename_from_outdir_label_extension(
            outdir="outdir", label="psd", extension="pkl"
        )
        with self.assertRaises(TypeError):
            bilby.gw.detector.Interferometer.from_pickle(filename)


def test_psd_not_impacted_by_window_factor(monkeypatch):
    ifo = bilby.gw.detector.get_empty_interferometer("H1")
    ifo.set_strain_data_from_zero_noise(duration=4, sampling_frequency=256)
    monkeypatch.setattr(ifo.strain_data, "window_factor", np.nan)
    np.testing.assert_array_equal(
        ifo.power_spectral_density.get_power_spectral_density_array(
            frequency_array=ifo.strain_data.frequency_array
        ),
        ifo.power_spectral_density_array
    )


class TestInterferometerEquals(unittest.TestCase):
    def setUp(self):
        self.name = "name"
        self.power_spectral_density_1 = (
            bilby.gw.detector.PowerSpectralDensity.from_aligo()
        )
        self.power_spectral_density_2 = (
            bilby.gw.detector.PowerSpectralDensity.from_aligo()
        )
        self.minimum_frequency = 10
        self.maximum_frequency = 20
        self.length = 30
        self.latitude = 1
        self.longitude = 2
        self.elevation = 3
        self.xarm_azimuth = 4
        self.yarm_azimuth = 5
        self.xarm_tilt = 0.0
        self.yarm_tilt = 0.0
        # noinspection PyTypeChecker
        self.duration = 1
        self.sampling_frequency = 200
        self.frequency_array = bilby.utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.strain = self.frequency_array
        self.ifo_1 = bilby.gw.detector.Interferometer(
            name=self.name,
            power_spectral_density=self.power_spectral_density_1,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            length=self.length,
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            xarm_azimuth=self.xarm_azimuth,
            yarm_azimuth=self.yarm_azimuth,
            xarm_tilt=self.xarm_tilt,
            yarm_tilt=self.yarm_tilt,
        )
        self.ifo_2 = bilby.gw.detector.Interferometer(
            name=self.name,
            power_spectral_density=self.power_spectral_density_2,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            length=self.length,
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            xarm_azimuth=self.xarm_azimuth,
            yarm_azimuth=self.yarm_azimuth,
            xarm_tilt=self.xarm_tilt,
            yarm_tilt=self.yarm_tilt,
        )
        self.ifo_1.set_strain_data_from_frequency_domain_strain(
            frequency_array=self.frequency_array, frequency_domain_strain=self.strain
        )
        self.ifo_2.set_strain_data_from_frequency_domain_strain(
            frequency_array=self.frequency_array, frequency_domain_strain=self.strain
        )

    def tearDown(self):
        del self.name
        del self.power_spectral_density_1
        del self.power_spectral_density_2
        del self.minimum_frequency
        del self.maximum_frequency
        del self.length
        del self.latitude
        del self.longitude
        del self.elevation
        del self.xarm_azimuth
        del self.yarm_azimuth
        del self.xarm_tilt
        del self.yarm_tilt
        del self.ifo_1
        del self.ifo_2
        del self.sampling_frequency
        del self.duration
        del self.frequency_array
        del self.strain

    def test_eq_true(self):
        self.assertEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_psd(self):
        self.ifo_1.power_spectral_density.psd_array[0] = 1234
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_minimum_frequency(self):
        self.ifo_1.minimum_frequency -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_maximum_frequency(self):
        self.ifo_1.minimum_frequency -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_length(self):
        self.ifo_1.length -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_latitude(self):
        self.ifo_1.latitude -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_longitude(self):
        self.ifo_1.longitude -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_elevation(self):
        self.ifo_1.elevation -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_xarm_azimuth(self):
        self.ifo_1.xarm_azimuth -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_xarmtilt(self):
        self.ifo_1.xarm_tilt -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_yarm_azimuth(self):
        self.ifo_1.yarm_azimuth -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_yarm_tilt(self):
        self.ifo_1.yarm_tilt -= 1
        self.assertNotEqual(self.ifo_1, self.ifo_2)

    def test_eq_false_different_ifo_strain_data(self):
        self.strain = bilby.utils.create_frequency_series(
            sampling_frequency=self.sampling_frequency / 2, duration=self.duration * 2
        )
        self.ifo_1.set_strain_data_from_frequency_domain_strain(
            frequency_array=self.frequency_array, frequency_domain_strain=self.strain
        )
        self.assertNotEqual(self.ifo_1, self.ifo_2)


class TestInterferometerAntennaPatternAgainstLAL(unittest.TestCase):
    def setUp(self):
        self.name = "name"
        self.ifo_names = ['H1', 'L1', 'V1', 'K1', 'GEO600', 'ET']
        self.lal_prefixes = {'H1': 'H1', 'L1': 'L1', 'V1': 'V1', 'K1': 'K1', 'GEO600': 'G1', 'ET': 'E1'}
        self.polarizations = ['plus', 'cross', 'breathing', 'longitudinal', 'x', 'y']
        self.ifos = bilby.gw.detector.InterferometerList(self.ifo_names)
        self.gpstime = 1305303144
        self.trial = 100

    def tearDown(self):
        del self.name
        del self.ifo_names
        del self.lal_prefixes
        del self.polarizations
        del self.ifos
        del self.gpstime
        del self.trial

    def test_antenna_pattern_vs_lal(self):
        gmst = lal.GreenwichMeanSiderealTime(self.gpstime)
        f_bilby = np.zeros((self.trial, 6))
        f_lal = np.zeros((self.trial, 6))

        for n, ifo_name in enumerate(self.ifo_names):
            response = lalsimulation.DetectorPrefixToLALDetector(self.lal_prefixes[ifo_name]).response
            ifo = self.ifos[n]
            for i in range(self.trial):
                ra = 2. * np.pi * np.random.uniform()
                dec = np.pi * np.random.uniform() - np.pi / 2.
                psi = np.pi * np.random.uniform()
                f_lal[i] = lal.ComputeDetAMResponseExtraModes(response, ra, dec, psi, gmst)
                for m, pol in enumerate(self.polarizations):
                    f_bilby[i, m] = ifo.antenna_response(ra, dec, self.gpstime, psi, pol)

            std = np.std(f_bilby - f_lal, axis=0)
            for m, pol in enumerate(self.polarizations):
                with self.subTest(':'.join((ifo_name, pol))):
                    self.assertAlmostEqual(std[m], 0.0, places=7)

    def test_time_delay_vs_lal(self):
        delays = np.zeros(self.trial)

        for n, ifo_name in enumerate(self.ifo_names):
            ifo = self.ifos[n]
            det = lal.cached_detector_by_prefix[self.lal_prefixes[ifo_name]]
            for i in range(self.trial):
                gpstime = np.random.uniform(1205303144, 1405303144)
                ra = 2. * np.pi * np.random.uniform()
                dec = np.pi * np.random.uniform() - np.pi / 2.
                delays[i] = (
                    lal.TimeDelayFromEarthCenter(det.location, ra, dec, gpstime)
                    - ifo.time_delay_from_geocenter(ra, dec, gpstime)
                )

            std = max(abs(delays))
            with self.subTest(ifo_name):
                self.assertAlmostEqual(std, 0.0, places=10)


@pytest.mark.flaky(reruns=3, only_rerun=["AssertionError"])
class TestInterferometerWhitenedStrain(unittest.TestCase):
    def setUp(self):
        self.duration = 64
        self.sampling_frequency = 4096
        self.ifo = bilby.gw.detector.get_empty_interferometer('H1')
        self.ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=self.sampling_frequency, duration=self.duration)
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments={
                "waveform_approximant": "IMRPhenomXP"
            })

        self.parameters = {
            'mass_1': 10,
            'mass_2': 10,
            'a_1': 0,
            'a_2': 0,
            'tilt_1': 0,
            'tilt_2': 0,
            'phi_12': 0,
            'phi_jl': 0,
            'theta_jn': 0,
            'luminosity_distance': 40,
            'phase': 0,
            'ra': 0,
            'dec': 0,
            'geocent_time': 62,
            'psi': 0
        }

    def tearDown(self):
        del self.ifo
        del self.waveform_generator
        del self.parameters
        del self.duration
        del self.sampling_frequency

    def _check_frequency_series_whiteness(self, frequency_series):
        std_real = np.std(frequency_series.real)
        std_imag = np.std(frequency_series.imag)
        self.assertAlmostEqual(std_real, 1, places=2)
        self.assertAlmostEqual(std_imag, 1, places=2)

    def _check_time_series_whiteness(self, time_series):
        std = np.std(time_series)
        self.assertAlmostEqual(std, 1, places=2)

    def test_frequency_domain_whitened_strain(self):
        mask = self.ifo.frequency_mask
        white = self.ifo.whitened_frequency_domain_strain[mask]
        self._check_frequency_series_whiteness(white)

    def test_time_domain_whitened_strain(self):
        whitened_td = self.ifo.whitened_time_domain_strain
        self._check_time_series_whiteness(whitened_td)

    def test_frequency_domain_noise_and_signal_whitening(self):
        # Inject some (loud) signal
        self.ifo.inject_signal(waveform_generator=self.waveform_generator, parameters=self.parameters)
        # Make the template separately
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(parameters=self.parameters)
        signal_ifo = self.ifo.get_detector_response(
            waveform_polarizations=waveform_polarizations,
            parameters=self.parameters
        )
        # Whiten the template
        whitened_signal_ifo = self.ifo.whiten_frequency_series(signal_ifo)
        mask = self.ifo.frequency_mask
        white = self.ifo.whitened_frequency_domain_strain[mask] - whitened_signal_ifo[mask]
        self._check_frequency_series_whiteness(white)

    def test_time_domain_noise_and_signal_whitening(self):
        # Inject some (loud) signal
        self.ifo.inject_signal(waveform_generator=self.waveform_generator, parameters=self.parameters)
        # Make the template separately
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(parameters=self.parameters)
        signal_ifo = self.ifo.get_detector_response(
            waveform_polarizations=waveform_polarizations,
            parameters=self.parameters
        )
        # Whiten the template in FD
        whitened_signal_ifo_fd = self.ifo.whiten_frequency_series(signal_ifo)
        # Get whitened template in TD
        whitened_signal_ifo_td = self.ifo.get_whitened_time_series_from_whitened_frequency_series(
            whitened_signal_ifo_fd
        )
        whitened_td = self.ifo.whitened_time_domain_strain - whitened_signal_ifo_td
        self._check_time_series_whiteness(whitened_td)
