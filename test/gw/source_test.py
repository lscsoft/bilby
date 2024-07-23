import unittest

import bilby
import lal
import lalsimulation

import numpy as np
from copy import copy


class TestLalBBH(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=30.0,
            mass_2=30.0,
            luminosity_distance=400.0,
            a_1=0.4,
            tilt_1=0.2,
            phi_12=1.0,
            a_2=0.8,
            tilt_2=2.7,
            phi_jl=2.9,
            theta_jn=0.3,
            phase=0.0,
        )
        self.waveform_kwargs = dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=50.0,
            minimum_frequency=20.0,
            catch_waveform_errors=True,
        )
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)
        self.bad_parameters = copy(self.parameters)
        self.bad_parameters["mass_1"] = -30.0

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array
        del self.bad_parameters

    def test_lal_bbh_works_runs_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs)
        self.assertIsInstance(
            bilby.gw.source.lal_binary_black_hole(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_waveform_error_catching(self):
        self.bad_parameters.update(self.waveform_kwargs)
        self.assertIsNone(
            bilby.gw.source.lal_binary_black_hole(
                self.frequency_array, **self.bad_parameters
            )
        )

    def test_waveform_error_raising(self):
        raise_error_parameters = copy(self.bad_parameters)
        raise_error_parameters.update(self.waveform_kwargs)
        raise_error_parameters["catch_waveform_errors"] = False
        with self.assertRaises(Exception):
            bilby.gw.source.lal_binary_black_hole(
                self.frequency_array, **raise_error_parameters
            )

    def test_lal_bbh_works_without_waveform_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_binary_black_hole(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    # Removed due to issue with SimInspiralFD - see https://git.ligo.org/lscsoft/lalsuite/issues/153
    # def test_lal_bbh_works_with_time_domain_approximant(self):
    #     self.waveform_kwargs['waveform_approximant'] = 'SEOBNRv3'
    #     self.parameters.update(self.waveform_kwargs)
    #     self.assertIsInstance(
    #         bilby.gw.source.lal_binary_black_hole(
    #             self.frequency_array, **self.parameters), dict)

    def test_lal_bbh_xpprecession_version(self):
        self.parameters.update(self.waveform_kwargs)
        self.parameters["waveform_approximant"] = "IMRPhenomXP"

        # Test that we can modify the XP precession version
        out_v223 = bilby.gw.source.lal_binary_black_hole(
            self.frequency_array, PhenomXPrecVersion=223, **self.parameters
        )
        out_v102 = bilby.gw.source.lal_binary_black_hole(
            self.frequency_array, PhenomXPrecVersion=102, **self.parameters
        )
        self.assertFalse(np.all(out_v223["plus"] == out_v102["plus"]))


class TestGWSignalBBH(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=30.0,
            mass_2=30.0,
            luminosity_distance=400.0,
            a_1=0.4,
            tilt_1=0.2,
            phi_12=1.0,
            a_2=0.8,
            tilt_2=2.7,
            phi_jl=2.9,
            theta_jn=0.3,
            phase=0.0,
        )
        self.waveform_kwargs = dict(
            waveform_approximant="IMRPhenomXPHM",
            reference_frequency=50.0,
            minimum_frequency=20.0,
            catch_waveform_errors=True,
        )
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)
        self.bad_parameters = copy(self.parameters)
        self.bad_parameters["mass_1"] = -30.0

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array
        del self.bad_parameters

    def test_gwsignal_bbh_works_runs_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs)
        self.assertIsInstance(
            bilby.gw.source.gwsignal_binary_black_hole(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_waveform_error_catching(self):
        self.bad_parameters.update(self.waveform_kwargs)
        self.assertIsNone(
            bilby.gw.source.gwsignal_binary_black_hole(
                self.frequency_array, **self.bad_parameters
            )
        )

    def test_waveform_error_raising(self):
        raise_error_parameters = copy(self.bad_parameters)
        raise_error_parameters.update(self.waveform_kwargs)
        raise_error_parameters["catch_waveform_errors"] = False
        with self.assertRaises(Exception):
            bilby.gw.source.gwsignal_binary_black_hole(
                self.frequency_array, **raise_error_parameters
            )
    # def test_gwsignal_bbh_works_without_waveform_parameters(self):
    #    self.assertIsInstance(
    #        bilby.gw.source.gwsignal_binary_black_hole(
    #            self.frequency_array, **self.parameters
    #        ),
    #        dict,
    #    )

    def test_gwsignal_lal_bbh_consistency(self):
        self.parameters.update(self.waveform_kwargs)
        hpc_gwsignal = bilby.gw.source.gwsignal_binary_black_hole(
            self.frequency_array, **self.parameters
        )
        hpc_lal = bilby.gw.source.lal_binary_black_hole(
            self.frequency_array, **self.parameters
        )
        self.assertTrue(
            np.allclose(hpc_gwsignal["plus"], hpc_lal["plus"], atol=0, rtol=1e-7)
        )
        self.assertTrue(
            np.allclose(hpc_gwsignal["cross"], hpc_lal["cross"], atol=0, rtol=1e-7)
        )


class TestLalBNS(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=1.4,
            mass_2=1.4,
            luminosity_distance=400.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.2,
            tilt_2=1.7,
            phi_jl=0.2,
            phi_12=0.9,
            theta_jn=1.7,
            phase=0.0,
            lambda_1=100.0,
            lambda_2=100.0,
        )
        self.waveform_kwargs = dict(
            waveform_approximant="IMRPhenomPv2_NRTidal",
            reference_frequency=50.0,
            minimum_frequency=20.0,
        )
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array

    def test_lal_bns_runs_with_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs)
        self.assertIsInstance(
            bilby.gw.source.lal_binary_neutron_star(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_lal_bns_works_without_waveform_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_binary_neutron_star(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_fails_without_tidal_parameters(self):
        self.parameters.pop("lambda_1")
        self.parameters.pop("lambda_2")
        self.parameters.update(self.waveform_kwargs)
        with self.assertRaises(TypeError):
            bilby.gw.source.lal_binary_neutron_star(
                self.frequency_array, **self.parameters
            )


class TestEccentricLalBBH(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=30.0,
            mass_2=30.0,
            luminosity_distance=400.0,
            theta_jn=0.0,
            phase=0.0,
            eccentricity=0.1,
        )
        self.waveform_kwargs = dict(
            waveform_approximant="EccentricFD",
            reference_frequency=10.0,
            minimum_frequency=10.0,
        )
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array

    def test_lal_ebbh_works_runs_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs)
        self.assertIsInstance(
            bilby.gw.source.lal_eccentric_binary_black_hole_no_spins(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_lal_ebbh_works_without_waveform_parameters(self):
        self.assertIsInstance(
            bilby.gw.source.lal_eccentric_binary_black_hole_no_spins(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_fails_without_eccentricity(self):
        self.parameters.pop("eccentricity")
        self.parameters.update(self.waveform_kwargs)
        with self.assertRaises(TypeError):
            bilby.gw.source.lal_eccentric_binary_black_hole_no_spins(
                self.frequency_array, **self.parameters
            )


class TestROQBBH(unittest.TestCase):
    def setUp(self):
        roq_dir = "/roq_basis"

        fnodes_linear_file = "{}/fnodes_linear.npy".format(roq_dir)
        fnodes_linear = np.load(fnodes_linear_file).T
        fnodes_quadratic_file = "{}/fnodes_quadratic.npy".format(roq_dir)
        fnodes_quadratic = np.load(fnodes_quadratic_file).T

        self.parameters = dict(
            mass_1=30.0,
            mass_2=30.0,
            luminosity_distance=400.0,
            a_1=0.0,
            tilt_1=0.0,
            phi_12=0.0,
            a_2=0.0,
            tilt_2=0.0,
            phi_jl=0.0,
            theta_jn=0.0,
            phase=0.0,
        )
        self.waveform_kwargs = dict(
            frequency_nodes_linear=fnodes_linear,
            frequency_nodes_quadratic=fnodes_quadratic,
            reference_frequency=50.0,
            waveform_approximant="IMRPhenomPv2",
        )
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array

    def test_roq_runs_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs)
        self.assertIsInstance(
            bilby.gw.source.binary_black_hole_roq(self.frequency_array, **self.parameters), dict
        )

    def test_roq_fails_without_frequency_nodes(self):
        self.parameters.update(self.waveform_kwargs)
        del self.parameters["frequency_nodes_linear"]
        del self.parameters["frequency_nodes_quadratic"]
        with self.assertRaises(KeyError):
            bilby.gw.source.binary_black_hole_roq(self.frequency_array, **self.parameters)


class TestBBHfreqseq(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=30.0,
            mass_2=30.0,
            luminosity_distance=400.0,
            a_1=0.4,
            tilt_1=0.,
            phi_12=0.,
            a_2=0.8,
            tilt_2=0.,
            phi_jl=0.,
            theta_jn=0.3,
            phase=0.0
        )
        self.minimum_frequency = 20.0
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 8)
        self.full_frequencies_to_sequence = self.frequency_array >= self.minimum_frequency
        self.frequencies = self.frequency_array[self.full_frequencies_to_sequence]
        self.waveform_kwargs = dict(
            waveform_approximant="IMRPhenomHM",
            reference_frequency=50.0,
            catch_waveform_errors=True,
        )
        self.bad_parameters = copy(self.parameters)
        self.bad_parameters["mass_1"] = -30.0

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array
        del self.bad_parameters
        del self.minimum_frequency

    def test_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs)
        self.assertIsInstance(
            bilby.gw.source.binary_black_hole_frequency_sequence(
                self.frequency_array, frequencies=self.frequencies, **self.parameters
            ),
            dict
        )

    def test_waveform_error_catching(self):
        self.bad_parameters.update(self.waveform_kwargs)
        self.assertIsNone(
            bilby.gw.source.binary_black_hole_frequency_sequence(
                self.frequency_array, frequencies=self.frequencies, **self.bad_parameters
            )
        )

    def test_waveform_error_raising(self):
        raise_error_parameters = copy(self.bad_parameters)
        raise_error_parameters.update(self.waveform_kwargs)
        raise_error_parameters["catch_waveform_errors"] = False
        with self.assertRaises(Exception):
            bilby.gw.source.binary_black_hole_frequency_sequence(
                self.frequency_array, frequencies=self.frequencies, **raise_error_parameters
            )

    def test_match_LalBBH(self):
        self.parameters.update(self.waveform_kwargs)
        freqseq = bilby.gw.source.binary_black_hole_frequency_sequence(
            self.frequency_array, frequencies=self.frequencies, **self.parameters
        )
        lalbbh = bilby.gw.source.lal_binary_black_hole(
            self.frequency_array, minimum_frequency=self.minimum_frequency, **self.parameters
        )
        self.assertEqual(freqseq.keys(), lalbbh.keys())
        for mode in freqseq:
            diff = np.sum(np.abs(freqseq[mode] - lalbbh[mode][self.full_frequencies_to_sequence])**2.)
            norm = np.sum(np.abs(freqseq[mode])**2.)
            self.assertLess(diff / norm, 1e-15)

    def test_match_LalBBH_specify_modes(self):
        parameters = copy(self.parameters)
        parameters.update(self.waveform_kwargs)
        parameters['mode_array'] = [[2, 2]]
        freqseq = bilby.gw.source.binary_black_hole_frequency_sequence(
            self.frequency_array, frequencies=self.frequencies, **parameters
        )
        lalbbh = bilby.gw.source.lal_binary_black_hole(
            self.frequency_array, minimum_frequency=self.minimum_frequency, **parameters
        )
        self.assertEqual(freqseq.keys(), lalbbh.keys())
        for mode in freqseq:
            diff = np.sum(np.abs(freqseq[mode] - lalbbh[mode][self.full_frequencies_to_sequence])**2.)
            norm = np.sum(np.abs(freqseq[mode])**2.)
            self.assertLess(diff / norm, 1e-15)

    def test_match_LalBBH_nonGR(self):
        parameters = copy(self.parameters)
        parameters.update(self.waveform_kwargs)
        wf_dict = lal.CreateDict()
        lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi0(wf_dict, 1.)
        parameters['lal_waveform_dictionary'] = wf_dict
        freqseq = bilby.gw.source.binary_black_hole_frequency_sequence(
            self.frequency_array, frequencies=self.frequencies, **parameters
        )
        lalbbh = bilby.gw.source.lal_binary_black_hole(
            self.frequency_array, minimum_frequency=self.minimum_frequency, **parameters
        )
        self.assertEqual(freqseq.keys(), lalbbh.keys())
        for mode in freqseq:
            diff = np.sum(np.abs(freqseq[mode] - lalbbh[mode][self.full_frequencies_to_sequence])**2.)
            norm = np.sum(np.abs(freqseq[mode])**2.)
            self.assertLess(diff / norm, 1e-15)


class TestBNSfreqseq(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=1.4,
            mass_2=1.4,
            luminosity_distance=400.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.2,
            tilt_2=1.7,
            phi_jl=0.2,
            phi_12=0.9,
            theta_jn=1.7,
            phase=0.0,
            lambda_1=1000.0,
            lambda_2=1000.0
        )
        self.minimum_frequency = 50.0
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 16)
        self.full_frequencies_to_sequence = self.frequency_array >= self.minimum_frequency
        self.frequencies = self.frequency_array[self.full_frequencies_to_sequence]
        self.waveform_kwargs = dict(
            waveform_approximant="IMRPhenomPv2_NRTidal",
            reference_frequency=50.0,
        )

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs
        del self.frequency_array
        del self.minimum_frequency

    def test_with_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs)
        self.assertIsInstance(
            bilby.gw.source.binary_neutron_star_frequency_sequence(
                self.frequency_array, frequencies=self.frequencies, **self.parameters
            ),
            dict
        )

    def test_fails_without_tidal_parameters(self):
        self.parameters.pop("lambda_1")
        self.parameters.pop("lambda_2")
        self.parameters.update(self.waveform_kwargs)
        with self.assertRaises(TypeError):
            bilby.gw.source.binary_neutron_star_frequency_sequence(
                self.frequency_array, frequencies=self.frequencies, **self.parameters
            )

    def test_match_LalBNS(self):
        self.parameters.update(self.waveform_kwargs)
        freqseq = bilby.gw.source.binary_neutron_star_frequency_sequence(
            self.frequency_array, frequencies=self.frequencies, **self.parameters
        )
        lalbns = bilby.gw.source.lal_binary_neutron_star(
            self.frequency_array, minimum_frequency=self.minimum_frequency, **self.parameters
        )
        self.assertEqual(freqseq.keys(), lalbns.keys())
        for mode in freqseq:
            diff = np.sum(np.abs(freqseq[mode] - lalbns[mode][self.full_frequencies_to_sequence])**2.)
            norm = np.sum(np.abs(freqseq[mode])**2.)
            self.assertLess(diff / norm, 1e-5)


class TestRelbinBBH(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=30.0,
            mass_2=30.0,
            luminosity_distance=400.0,
            a_1=0.4,
            tilt_1=0.2,
            phi_12=1.0,
            a_2=0.8,
            tilt_2=2.7,
            phi_jl=2.9,
            theta_jn=0.3,
            phase=0.0,
        )
        self.waveform_kwargs_fiducial = dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=50.0,
            minimum_frequency=20.0,
            catch_waveform_errors=True,
            fiducial=True,
        )
        self.waveform_kwargs_binned = dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=50.0,
            minimum_frequency=20.0,
            catch_waveform_errors=True,
            fiducial=False,
            frequency_bin_edges=np.arange(20, 1500, 50)
        )
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)
        self.bad_parameters = copy(self.parameters)
        self.bad_parameters["mass_1"] = -30.0

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs_fiducial
        del self.waveform_kwargs_binned
        del self.frequency_array
        del self.bad_parameters

    def test_relbin_fiducial_bbh_works_runs_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs_fiducial)
        self.assertIsInstance(
            bilby.gw.source.lal_binary_black_hole_relative_binning(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_relbin_binned_bbh_works_runs_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs_binned)
        self.assertIsInstance(
            bilby.gw.source.lal_binary_black_hole_relative_binning(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_waveform_error_catching_fiducial(self):
        self.bad_parameters.update(self.waveform_kwargs_fiducial)
        self.assertIsNone(
            bilby.gw.source.lal_binary_black_hole_relative_binning(
                self.frequency_array, **self.bad_parameters
            )
        )

    def test_waveform_error_catching_binned(self):
        self.bad_parameters.update(self.waveform_kwargs_binned)
        self.assertIsNone(
            bilby.gw.source.lal_binary_black_hole_relative_binning(
                self.frequency_array, **self.bad_parameters
            )
        )

    def test_waveform_error_raising_fiducial(self):
        raise_error_parameters = copy(self.bad_parameters)
        raise_error_parameters.update(self.waveform_kwargs_fiducial)
        raise_error_parameters["catch_waveform_errors"] = False
        with self.assertRaises(Exception):
            bilby.gw.source.lal_binary_black_hole_relative_binning(
                self.frequency_array, **raise_error_parameters
            )

    def test_waveform_error_raising_binned(self):
        raise_error_parameters = copy(self.bad_parameters)
        raise_error_parameters.update(self.waveform_kwargs_binned)
        raise_error_parameters["catch_waveform_errors"] = False
        with self.assertRaises(Exception):
            bilby.gw.source.lal_binary_black_hole_relative_binning(
                self.frequency_array, **raise_error_parameters
            )

    def test_relbin_bbh_fails_without_fiducial_option(self):
        with self.assertRaises(TypeError):
            bilby.gw.source.lal_binary_black_hole_relative_binning(
                self.frequency_array, **self.parameters
            )

    def test_relbin_bbh_xpprecession_version(self):
        self.parameters.update(self.waveform_kwargs_fiducial)
        self.parameters["waveform_approximant"] = "IMRPhenomXP"

        # Test that we can modify the XP precession version
        out_v223 = bilby.gw.source.lal_binary_black_hole_relative_binning(
            self.frequency_array, PhenomXPrecVersion=223, **self.parameters
        )
        out_v102 = bilby.gw.source.lal_binary_black_hole_relative_binning(
            self.frequency_array, PhenomXPrecVersion=102, **self.parameters
        )
        self.assertFalse(np.all(out_v223["plus"] == out_v102["plus"]))


class TestRelbinBNS(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            mass_1=1.4,
            mass_2=1.4,
            luminosity_distance=400.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.2,
            tilt_2=1.7,
            phi_jl=0.2,
            phi_12=0.9,
            theta_jn=1.7,
            phase=0.0,
            lambda_1=100.0,
            lambda_2=100.0,
        )
        self.waveform_kwargs_fiducial = dict(
            waveform_approximant="IMRPhenomPv2_NRTidal",
            reference_frequency=50.0,
            minimum_frequency=20.0,
            fiducial=True,
        )
        self.waveform_kwargs_binned = dict(
            waveform_approximant="IMRPhenomPv2_NRTidal",
            reference_frequency=50.0,
            minimum_frequency=20.0,
            fiducial=False,
            frequency_bin_edges=np.arange(20, 1500, 50),
        )
        self.frequency_array = bilby.core.utils.create_frequency_series(2048, 4)

    def tearDown(self):
        del self.parameters
        del self.waveform_kwargs_fiducial
        del self.waveform_kwargs_binned
        del self.frequency_array

    def test_relbin_fiducial_bns_runs_with_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs_fiducial)
        self.assertIsInstance(
            bilby.gw.source.lal_binary_neutron_star_relative_binning(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_relbin_binned_bns_runs_with_valid_parameters(self):
        self.parameters.update(self.waveform_kwargs_binned)
        self.assertIsInstance(
            bilby.gw.source.lal_binary_neutron_star_relative_binning(
                self.frequency_array, **self.parameters
            ),
            dict,
        )

    def test_relbin_bns_fails_without_fiducial_option(self):
        with self.assertRaises(TypeError):
            bilby.gw.source.lal_binary_neutron_star_relative_binning(
                self.frequency_array, **self.parameters
            )

    def test_fiducial_fails_without_tidal_parameters(self):
        self.parameters.pop("lambda_1")
        self.parameters.pop("lambda_2")
        self.parameters.update(self.waveform_kwargs_fiducial)
        with self.assertRaises(TypeError):
            bilby.gw.source.lal_binary_neutron_star_relative_binning(
                self.frequency_array, **self.parameters
            )

    def test_binned_fails_without_tidal_parameters(self):
        self.parameters.pop("lambda_1")
        self.parameters.pop("lambda_2")
        self.parameters.update(self.waveform_kwargs_binned)
        with self.assertRaises(TypeError):
            bilby.gw.source.lal_binary_neutron_star_relative_binning(
                self.frequency_array, **self.parameters
            )


if __name__ == "__main__":
    unittest.main()
