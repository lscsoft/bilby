import unittest
import bilby
import lal
import lalsimulation as lalsim
import numpy as np
import pytest
from bilby.gw.utils import overlap


pytestmark = pytest.mark.gw


class TestWaveformDirectAgainstLALSIM(unittest.TestCase):
    def setUp(self):
        self.BBH_precessing_injection_parameters = dict(
            mass_1=36.0,
            mass_2=32.0,
            a_1=0.2,
            a_2=0.4,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=0.0,
            phi_jl=0.0,
            luminosity_distance=4000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3 + np.pi / 2.0,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=0.2108,
        )

        self.BNS_precessing_injection_parameters = dict(
            mass_1=36.0,
            mass_2=32.0,
            a_1=0.2,
            a_2=0.4,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=0.0,
            phi_jl=0.0,
            luminosity_distance=4000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3 + np.pi / 2.0,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=0.2108,
            lambda_1=1000,
            lambda_2=1500,
        )

    def test_IMRPhenomPv2(self):
        waveform_approximant = "IMRPhenomPv2"
        self.run_for_approximant(waveform_approximant, source="bbh")

    def test_IMRPhenomD(self):
        waveform_approximant = "IMRPhenomD"
        self.run_for_approximant(waveform_approximant, source="bbh")

    def test_IMRPhenomPv2_NRTidal(self):
        waveform_approximant = "IMRPhenomPv2_NRTidal"
        self.run_for_approximant(waveform_approximant, source="bns")

    def test_IMRPhenomD_NRTidal(self):
        waveform_approximant = "IMRPhenomD_NRTidal"
        self.run_for_approximant(waveform_approximant, source="bns")

    def test_TaylorF2(self):
        waveform_approximant = "TaylorF2"
        self.run_for_approximant(waveform_approximant, source="bns")

    def run_for_approximant(self, waveform_approximant, source):

        if source == "bbh":
            injection_parameters = self.BBH_precessing_injection_parameters
            frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole
        elif source == "bns":
            injection_parameters = self.BNS_precessing_injection_parameters
            frequency_domain_source_model = bilby.gw.source.lal_binary_neutron_star
        else:
            raise ValueError("Source can only be 'bbh' or 'bns', but was '{}'".format(source))

        # create a waveform generator for bilby
        duration = 4.0
        sampling_frequency = 2048.0
        reference_frequency = 20.0
        minimum_frequency = 20.0
        # Fixed arguments passed into the source model

        waveform_arguments = dict(
            waveform_approximant=waveform_approximant,
            reference_frequency=reference_frequency,
            minimum_frequency=minimum_frequency,
        )

        (
            iota,
            spin_1x,
            spin_1y,
            spin_1z,
            spin_2x,
            spin_2y,
            spin_2z,
        ) = bilby.gw.conversion.bilby_to_lalsimulation_spins(
            theta_jn=injection_parameters["theta_jn"],
            phi_jl=injection_parameters["phi_jl"],
            tilt_1=injection_parameters["tilt_1"],
            tilt_2=injection_parameters["tilt_2"],
            phi_12=injection_parameters["phi_12"],
            a_1=injection_parameters["a_1"],
            a_2=injection_parameters["a_2"],
            mass_1=injection_parameters["mass_1"],
            mass_2=injection_parameters["mass_2"],
            reference_frequency=reference_frequency,
            phase=injection_parameters["phase"],
        )

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=frequency_domain_source_model,
            waveform_arguments=waveform_arguments,
        )

        bilby_strain = waveform_generator.frequency_domain_strain(
            parameters=injection_parameters
        )

        # LALSIM Waveform

        lambda_1 = injection_parameters.get("lambda_1", None)
        lambda_2 = injection_parameters.get("lambda_2", None)

        get_lalsim_waveform = lalsim_FD_waveform(
            injection_parameters["mass_1"],
            injection_parameters["mass_2"],
            spin_1x,
            spin_1y,
            spin_1z,
            spin_2x,
            spin_2y,
            spin_2z,
            iota,
            injection_parameters["phase"],
            duration,
            injection_parameters["luminosity_distance"],
            (waveform_generator.frequency_array)[-1],
            lambda_1,
            lambda_2,
            **waveform_arguments
        )

        h_plus = get_lalsim_waveform["plus"]
        h_cross = get_lalsim_waveform["cross"]

        if waveform_approximant == "TaylorF2":
            upper_freq = ISCO(
                injection_parameters["mass_1"], injection_parameters["mass_2"]
            )

        else:
            upper_freq = waveform_generator.frequency_array[-1]

        # Frequency resolution
        delta_f = 1.0 / duration
        # length of PSD
        f_len = int((2 * sampling_frequency) / delta_f)

        # PSD aLIGO
        psd_aLIGO = generate_PSD(
            psd_name="aLIGOZeroDetHighPower", length=f_len, delta_f=delta_f
        )

        norm_hp_bilby = normalize_strain(
            bilby_strain["plus"],
            psd=psd_aLIGO.data.data,
            delta_f=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
        )

        norm_hc_bilby = normalize_strain(
            bilby_strain["cross"],
            psd=psd_aLIGO.data.data,
            delta_f=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
        )

        norm_hp_lalsim = normalize_strain(
            h_plus,
            psd=psd_aLIGO.data.data,
            delta_f=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
        )

        norm_hc_lalsim = normalize_strain(
            h_cross,
            psd=psd_aLIGO.data.data,
            delta_f=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
        )

        # Match/Overpal between polarizations of lalsim and Bilby
        match_Hplus = overlap(
            bilby_strain["plus"],
            h_plus,
            power_spectral_density=psd_aLIGO.data.data,
            delta_frequency=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
            norm_a=norm_hp_bilby,
            norm_b=norm_hp_lalsim,
        )

        match_Hcross = overlap(
            bilby_strain["cross"],
            h_cross,
            power_spectral_density=psd_aLIGO.data.data,
            delta_frequency=delta_f,
            lower_cut_off=minimum_frequency,
            upper_cut_off=upper_freq,
            norm_a=norm_hc_bilby,
            norm_b=norm_hc_lalsim,
        )

        self.assertAlmostEqual(match_Hplus, 1, places=4)
        self.assertAlmostEqual(match_Hcross, 1, places=4)


def ISCO(m1, m2):
    return 1.0 / (6.0 * np.sqrt(6.0) * np.pi * (m1 + m2) * lal.MTSUN_SI)


def lalsim_FD_waveform(
    m1,
    m2,
    s1x,
    s1y,
    s1z,
    s2x,
    s2y,
    s2z,
    theta_jn,
    phase,
    duration,
    dL,
    fmax,
    lambda_1=None,
    lambda_2=None,
    **kwarg
):
    mass1 = m1 * lal.MSUN_SI
    mass2 = m2 * lal.MSUN_SI
    spin_1x = s1x
    spin_1y = s1y
    spin_1z = s1z
    spin_2x = s2x
    spin_2y = s2y
    spin_2z = s2z
    iota = theta_jn
    phaseC = phase  # Phase is hard coded to be zero

    eccentricity = 0
    longitude_ascending_nodes = 0
    mean_per_ano = 0

    waveform_arg = dict(minimum_freq=20.0, reference_frequency=20)
    waveform_arg.update(kwarg)
    dL = dL * lal.PC_SI * 1e6  # MPC --> Km
    approximant = lalsim.GetApproximantFromString(waveform_arg["waveform_approximant"])
    flow = waveform_arg["minimum_freq"]
    delta_freq = 1.0 / duration
    maximum_frequency = fmax  # 1024.0 # ISCO(m1, m2)
    fref = waveform_arg["reference_frequency"]
    waveform_dictionary = lal.CreateDict()

    if lambda_1 is not None:
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(
            waveform_dictionary, float(lambda_1)
        )
    if lambda_2 is not None:
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(
            waveform_dictionary, float(lambda_2)
        )

    hplus, hcross = lalsim.SimInspiralChooseFDWaveform(
        mass1,
        mass2,
        spin_1x,
        spin_1y,
        spin_1z,
        spin_2x,
        spin_2y,
        spin_2z,
        dL,
        iota,
        phaseC,
        longitude_ascending_nodes,
        eccentricity,
        mean_per_ano,
        delta_freq,
        flow,
        maximum_frequency,
        fref,
        waveform_dictionary,
        approximant,
    )

    h_plus = hplus.data.data[:]
    h_cross = hcross.data.data[:]

    return {"plus": h_plus, "cross": h_cross}


# Function for PSD list
def get_lalsim_psd_list():
    PSD_prefix = "SimNoisePSD"
    PSD_suffix = "Ptr"
    blacklist = [
        "FromFile",
        "MirrorTherm",
        "Quantum",
        "Seismic",
        "Shot",
        "SuspTherm",
        "TAMA",
        "GEO",
        "GEOHF",
        "aLIGOThermal",
    ]
    psd_list = []
    # Avoid the string 'SimNoisePSD'
    for name in lalsim.__dict__:
        if (
            name != PSD_prefix
            and name.startswith(PSD_prefix)
            and not name.endswith(PSD_suffix)
        ):
            # if name in blacklist:
            name = name[len(PSD_prefix) :]
            if (
                name not in blacklist
                and not name.startswith("iLIGO")
                and not name.startswith("eLIGO")
            ):
                psd_list.append(name)
    return sorted(psd_list)


# Function to generate PSDs
def generate_PSD(psd_name="aLIGOZeroDetHighPower", length=None, delta_f=None):
    psd_list = get_lalsim_psd_list()

    if psd_name in psd_list:
        # print (psd_name)
        # Function for PSD
        func = lalsim.__dict__["SimNoisePSD" + psd_name + "Ptr"]
        # Generate a lal frequency series
        PSDseries = lal.CreateREAL8FrequencySeries(
            "", lal.LIGOTimeGPS(0), 0, delta_f, lal.DimensionlessUnit, length
        )
        # func(PSDseries)
        lalsim.SimNoisePSD(PSDseries, 0, func)
    return PSDseries


# Normalizing a waveform
def normalize_strain(
    signal, psd=None, delta_f=None, lower_cut_off=None, upper_cut_off=None
):
    low_index = int(lower_cut_off / delta_f)
    up_index = int(upper_cut_off / delta_f)
    integrand = np.conj(signal) * signal
    integrand = integrand[low_index:up_index] / psd[low_index:up_index]
    integral = sum(4 * delta_f * integrand)
    return np.sqrt(integral).real


if __name__ == "__main__":
    unittest.main()
