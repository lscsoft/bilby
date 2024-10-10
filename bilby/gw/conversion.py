"""
A collection of functions to convert between parameters describing
gravitational-wave sources.
"""

import os
import sys
import multiprocessing
import pickle

import numpy as np
from pandas import DataFrame, Series
from scipy.stats import norm

from .utils import (lalsim_SimNeutronStarEOS4ParamSDGammaCheck,
                    lalsim_SimNeutronStarEOS4ParameterSpectralDecomposition,
                    lalsim_SimNeutronStarEOS4ParamSDViableFamilyCheck,
                    lalsim_SimNeutronStarEOS3PieceDynamicPolytrope,
                    lalsim_SimNeutronStarEOS3PieceCausalAnalytic,
                    lalsim_SimNeutronStarEOS3PDViableFamilyCheck,
                    lalsim_CreateSimNeutronStarFamily,
                    lalsim_SimNeutronStarEOSMaxPseudoEnthalpy,
                    lalsim_SimNeutronStarEOSSpeedOfSoundGeometerized,
                    lalsim_SimNeutronStarFamMinimumMass,
                    lalsim_SimNeutronStarMaximumMass,
                    lalsim_SimNeutronStarRadius,
                    lalsim_SimNeutronStarLoveNumberK2)

from ..core.likelihood import MarginalizedLikelihoodReconstructionError
from ..core.utils import logger, solar_mass, gravitational_constant, speed_of_light, command_line_args, safe_file_dump
from ..core.prior import DeltaFunction
from .utils import lalsim_SimInspiralTransformPrecessingNewInitialConditions
from .eos.eos import IntegrateTOV
from .cosmology import get_cosmology, z_at_value


def redshift_to_luminosity_distance(redshift, cosmology=None):
    cosmology = get_cosmology(cosmology)
    return cosmology.luminosity_distance(redshift).value


def redshift_to_comoving_distance(redshift, cosmology=None):
    cosmology = get_cosmology(cosmology)
    return cosmology.comoving_distance(redshift).value


def luminosity_distance_to_redshift(distance, cosmology=None):
    from astropy import units
    cosmology = get_cosmology(cosmology)
    if isinstance(distance, Series):
        distance = distance.values
    return z_at_value(cosmology.luminosity_distance, distance * units.Mpc)


def comoving_distance_to_redshift(distance, cosmology=None):
    from astropy import units
    cosmology = get_cosmology(cosmology)
    if isinstance(distance, Series):
        distance = distance.values
    return z_at_value(cosmology.comoving_distance, distance * units.Mpc)


def comoving_distance_to_luminosity_distance(distance, cosmology=None):
    cosmology = get_cosmology(cosmology)
    redshift = comoving_distance_to_redshift(distance, cosmology)
    return redshift_to_luminosity_distance(redshift, cosmology)


def luminosity_distance_to_comoving_distance(distance, cosmology=None):
    cosmology = get_cosmology(cosmology)
    redshift = luminosity_distance_to_redshift(distance, cosmology)
    return redshift_to_comoving_distance(redshift, cosmology)


_cosmology_docstring = """
Convert from {input} to {output}

Parameters
----------
{input}: float
    The {input} to convert.
cosmology: astropy.cosmology.Cosmology
    The cosmology to use for the transformation.
    See :code:`bilby.gw.cosmology.get_cosmology` for details of how to
    specify this.

Returns
-------
float
    The {output} corresponding to the provided {input}.
"""

for _func in [
    comoving_distance_to_luminosity_distance,
    comoving_distance_to_redshift,
    luminosity_distance_to_comoving_distance,
    luminosity_distance_to_redshift,
    redshift_to_comoving_distance,
    redshift_to_luminosity_distance,
]:
    input, output = _func.__name__.split("_to_")
    _func.__doc__ = _cosmology_docstring.format(input=input, output=output)


def bilby_to_lalsimulation_spins(
    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
    reference_frequency, phase
):
    """
    Convert from Bilby spin parameters to lalsimulation ones.

    All parameters are defined at the reference frequency and in SI units.

    Parameters
    ==========
    theta_jn: float
        Inclination angle
    phi_jl: float
        Spin phase angle
    tilt_1: float
        Primary object tilt
    tilt_2: float
        Secondary object tilt
    phi_12: float
        Relative spin azimuthal angle
    a_1: float
        Primary dimensionless spin magnitude
    a_2: float
        Secondary dimensionless spin magnitude
    mass_1: float
        Primary mass in SI units
    mass_2: float
        Secondary mass in SI units
    reference_frequency: float
    phase: float
        Orbital phase

    Returns
    =======
    iota: float
        Transformed inclination
    spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z: float
        Cartesian spin components
    """
    if (a_1 == 0 or tilt_1 in [0, np.pi]) and (a_2 == 0 or tilt_2 in [0, np.pi]):
        spin_1x = 0
        spin_1y = 0
        spin_1z = a_1 * np.cos(tilt_1)
        spin_2x = 0
        spin_2y = 0
        spin_2z = a_2 * np.cos(tilt_2)
        iota = theta_jn
    else:
        from numbers import Number
        args = (
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1,
            mass_2, reference_frequency, phase
        )
        float_inputs = all([isinstance(arg, Number) for arg in args])
        if float_inputs:
            func = lalsim_SimInspiralTransformPrecessingNewInitialConditions
        else:
            func = transform_precessing_spins
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = func(*args)
    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z


@np.vectorize
def transform_precessing_spins(*args):
    """
    Vectorized wrapper for
    lalsimulation.SimInspiralTransformPrecessingNewInitialConditions

    For detailed documentation see
    :code:`bilby.gw.conversion.bilby_to_lalsimulation_spins`.
    This will be removed from the public API in a future release.
    """
    return lalsim_SimInspiralTransformPrecessingNewInitialConditions(*args)


def convert_to_lal_binary_black_hole_parameters(parameters):
    """
    Convert parameters we have into parameters we need.

    This is defined by the parameters of bilby.source.lal_binary_black_hole()


    Mass: mass_1, mass_2
    Spin: a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl
    Extrinsic: luminosity_distance, theta_jn, phase, ra, dec, geocent_time, psi

    This involves popping a lot of things from parameters.
    The keys in added_keys should be popped after evaluating the waveform.

    Parameters
    ==========
    parameters: dict
        dictionary of parameter values to convert into the required parameters

    Returns
    =======
    converted_parameters: dict
        dict of the required parameters
    added_keys: list
        keys which are added to parameters during function call
    """

    converted_parameters = parameters.copy()
    original_keys = list(converted_parameters.keys())
    if 'luminosity_distance' not in original_keys:
        if 'redshift' in converted_parameters.keys():
            converted_parameters['luminosity_distance'] = \
                redshift_to_luminosity_distance(parameters['redshift'])
        elif 'comoving_distance' in converted_parameters.keys():
            converted_parameters['luminosity_distance'] = \
                comoving_distance_to_luminosity_distance(
                    parameters['comoving_distance'])

    for key in original_keys:
        if key[-7:] == '_source':
            if 'redshift' not in converted_parameters.keys():
                converted_parameters['redshift'] =\
                    luminosity_distance_to_redshift(
                        parameters['luminosity_distance'])
            converted_parameters[key[:-7]] = converted_parameters[key] * (
                1 + converted_parameters['redshift'])

    # we do not require the component masses be added if no mass parameters are present
    converted_parameters = generate_component_masses(converted_parameters, require_add=False)

    for idx in ['1', '2']:
        key = 'chi_{}'.format(idx)
        if key in original_keys:
            if "chi_{}_in_plane".format(idx) in original_keys:
                converted_parameters["a_{}".format(idx)] = (
                    converted_parameters[f"chi_{idx}"] ** 2
                    + converted_parameters[f"chi_{idx}_in_plane"] ** 2
                ) ** 0.5
                converted_parameters[f"cos_tilt_{idx}"] = (
                    converted_parameters[f"chi_{idx}"]
                    / converted_parameters[f"a_{idx}"]
                )
            elif "a_{}".format(idx) not in original_keys:
                converted_parameters['a_{}'.format(idx)] = abs(
                    converted_parameters[key])
                converted_parameters['cos_tilt_{}'.format(idx)] = \
                    np.sign(converted_parameters[key])
            else:
                with np.errstate(invalid="raise"):
                    try:
                        converted_parameters[f"cos_tilt_{idx}"] = (
                            converted_parameters[key] / converted_parameters[f"a_{idx}"]
                        )
                    except (FloatingPointError, ZeroDivisionError):
                        logger.debug(
                            "Error in conversion to spherical spin tilt. "
                            "This is often due to the spin parameters being zero. "
                            f"Setting cos_tilt_{idx} = 1."
                        )
                        converted_parameters[f"cos_tilt_{idx}"] = 1.0

    for key in ["phi_jl", "phi_12"]:
        if key not in converted_parameters:
            converted_parameters[key] = 0.0

    for angle in ['tilt_1', 'tilt_2', 'theta_jn']:
        cos_angle = str('cos_' + angle)
        if cos_angle in converted_parameters.keys():
            with np.errstate(invalid="ignore"):
                converted_parameters[angle] = np.arccos(converted_parameters[cos_angle])

    if "delta_phase" in original_keys:
        with np.errstate(invalid="ignore"):
            converted_parameters["phase"] = np.mod(
                converted_parameters["delta_phase"]
                - np.sign(np.cos(converted_parameters["theta_jn"]))
                * converted_parameters["psi"],
                2 * np.pi)
    added_keys = [key for key in converted_parameters.keys()
                  if key not in original_keys]

    return converted_parameters, added_keys


def convert_to_lal_binary_neutron_star_parameters(parameters):
    """
    Convert parameters we have into parameters we need.

    This is defined by the parameters of bilby.source.lal_binary_black_hole()


    Mass: mass_1, mass_2
    Spin: a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl
    Extrinsic: luminosity_distance, theta_jn, phase, ra, dec, geocent_time, psi
    Tidal: lambda_1, lamda_2, lambda_tilde, delta_lambda_tilde

    This involves popping a lot of things from parameters.
    The keys in added_keys should be popped after evaluating the waveform.
    For details on tidal parameters see https://arxiv.org/pdf/1402.5156.pdf.

    Parameters
    ==========
    parameters: dict
        dictionary of parameter values to convert into the required parameters

    Returns
    =======
    converted_parameters: dict
        dict of the required parameters
    added_keys: list
        keys which are added to parameters during function call
    """
    converted_parameters = parameters.copy()
    original_keys = list(converted_parameters.keys())
    converted_parameters, added_keys =\
        convert_to_lal_binary_black_hole_parameters(converted_parameters)

    if not any([key in converted_parameters for key in
                ['lambda_1', 'lambda_2',
                 'lambda_tilde', 'delta_lambda_tilde', 'lambda_symmetric',
                 'eos_polytrope_gamma_0', 'eos_spectral_pca_gamma_0', 'eos_v1']]):
        converted_parameters['lambda_1'] = 0
        converted_parameters['lambda_2'] = 0
        added_keys = added_keys + ['lambda_1', 'lambda_2']
        return converted_parameters, added_keys

    if 'delta_lambda_tilde' in converted_parameters.keys():
        converted_parameters['lambda_1'], converted_parameters['lambda_2'] =\
            lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(
                converted_parameters['lambda_tilde'],
                parameters['delta_lambda_tilde'], converted_parameters['mass_1'],
                converted_parameters['mass_2'])
    elif 'lambda_tilde' in converted_parameters.keys():
        converted_parameters['lambda_1'], converted_parameters['lambda_2'] =\
            lambda_tilde_to_lambda_1_lambda_2(
                converted_parameters['lambda_tilde'],
                converted_parameters['mass_1'], converted_parameters['mass_2'])
    if 'lambda_2' not in converted_parameters.keys() and 'lambda_1' in converted_parameters.keys():
        converted_parameters['lambda_2'] =\
            converted_parameters['lambda_1']\
            * converted_parameters['mass_1']**5\
            / converted_parameters['mass_2']**5
    elif 'lambda_2' in converted_parameters.keys() and converted_parameters['lambda_2'] is None:
        converted_parameters['lambda_2'] =\
            converted_parameters['lambda_1']\
            * converted_parameters['mass_1']**5\
            / converted_parameters['mass_2']**5
    elif 'eos_spectral_pca_gamma_0' in converted_parameters.keys():  # FIXME: This is a clunky way to do this
        converted_parameters = generate_source_frame_parameters(converted_parameters)
        float_eos_params = {}
        max_len = 1
        eos_keys = ['eos_spectral_pca_gamma_0',
                    'eos_spectral_pca_gamma_1',
                    'eos_spectral_pca_gamma_2',
                    'eos_spectral_pca_gamma_3',
                    'mass_1_source', 'mass_2_source']
        for key in eos_keys:
            try:
                if (len(converted_parameters[key]) > max_len):
                    max_len = len(converted_parameters[key])
            except TypeError:
                float_eos_params[key] = converted_parameters[key]
        if len(float_eos_params) == len(eos_keys):  # case where all eos params are floats (pinned)
            g0, g1, g2, g3 = spectral_pca_to_spectral(
                converted_parameters['eos_spectral_pca_gamma_0'],
                converted_parameters['eos_spectral_pca_gamma_1'],
                converted_parameters['eos_spectral_pca_gamma_2'],
                converted_parameters['eos_spectral_pca_gamma_3'])
            converted_parameters['lambda_1'], converted_parameters['lambda_2'], converted_parameters['eos_check'] = \
                spectral_params_to_lambda_1_lambda_2(
                    g0, g1, g2, g3, converted_parameters['mass_1_source'], converted_parameters['mass_2_source'])
        elif len(float_eos_params) < len(eos_keys):  # case where some or none of the eos params are floats (pinned)
            for key in float_eos_params.keys():
                converted_parameters[key] = np.ones(max_len) * converted_parameters[key]
            g0pca = converted_parameters['eos_spectral_pca_gamma_0']
            g1pca = converted_parameters['eos_spectral_pca_gamma_1']
            g2pca = converted_parameters['eos_spectral_pca_gamma_2']
            g3pca = converted_parameters['eos_spectral_pca_gamma_3']
            m1s = converted_parameters['mass_1_source']
            m2s = converted_parameters['mass_2_source']
            all_lambda_1 = np.empty(0)
            all_lambda_2 = np.empty(0)
            all_eos_check = np.empty(0, dtype=bool)
            for (g_0pca, g_1pca, g_2pca, g_3pca, m1_s, m2_s) in zip(g0pca, g1pca, g2pca, g3pca, m1s, m2s):
                g_0, g_1, g_2, g_3 = spectral_pca_to_spectral(g_0pca, g_1pca, g_2pca, g_3pca)
                lambda_1, lambda_2, eos_check = \
                    spectral_params_to_lambda_1_lambda_2(g_0, g_1, g_2, g_3, m1_s, m2_s)
                all_lambda_1 = np.append(all_lambda_1, lambda_1)
                all_lambda_2 = np.append(all_lambda_2, lambda_2)
                all_eos_check = np.append(all_eos_check, eos_check)
            converted_parameters['lambda_1'] = all_lambda_1
            converted_parameters['lambda_2'] = all_lambda_2
            converted_parameters['eos_check'] = all_eos_check
            for key in float_eos_params.keys():
                converted_parameters[key] = float_eos_params[key]
    elif 'eos_polytrope_gamma_0' and 'eos_polytrope_log10_pressure_1' in converted_parameters.keys():
        converted_parameters = generate_source_frame_parameters(converted_parameters)
        float_eos_params = {}
        max_len = 1
        eos_keys = ['eos_polytrope_gamma_0',
                    'eos_polytrope_gamma_1',
                    'eos_polytrope_gamma_2',
                    'eos_polytrope_log10_pressure_1',
                    'eos_polytrope_log10_pressure_2',
                    'mass_1_source', 'mass_2_source']
        for key in eos_keys:
            try:
                if (len(converted_parameters[key]) > max_len):
                    max_len = len(converted_parameters[key])
            except TypeError:
                float_eos_params[key] = converted_parameters[key]
        if len(float_eos_params) == len(eos_keys):  # case where all eos params are floats (pinned)
            converted_parameters['lambda_1'], converted_parameters['lambda_2'], converted_parameters['eos_check'] = \
                polytrope_or_causal_params_to_lambda_1_lambda_2(
                    converted_parameters['eos_polytrope_gamma_0'],
                    converted_parameters['eos_polytrope_log10_pressure_1'],
                    converted_parameters['eos_polytrope_gamma_1'],
                    converted_parameters['eos_polytrope_log10_pressure_2'],
                    converted_parameters['eos_polytrope_gamma_2'],
                    converted_parameters['mass_1_source'],
                    converted_parameters['mass_2_source'],
                    causal=0)
        elif len(float_eos_params) < len(eos_keys):  # case where some or none are floats (pinned)
            for key in float_eos_params.keys():
                converted_parameters[key] = np.ones(max_len) * converted_parameters[key]
            pg0 = converted_parameters['eos_polytrope_gamma_0']
            pg1 = converted_parameters['eos_polytrope_gamma_1']
            pg2 = converted_parameters['eos_polytrope_gamma_2']
            logp1 = converted_parameters['eos_polytrope_log10_pressure_1']
            logp2 = converted_parameters['eos_polytrope_log10_pressure_2']
            m1s = converted_parameters['mass_1_source']
            m2s = converted_parameters['mass_2_source']
            all_lambda_1 = np.empty(0)
            all_lambda_2 = np.empty(0)
            all_eos_check = np.empty(0, dtype=bool)
            for (pg_0, pg_1, pg_2, logp_1, logp_2, m1_s, m2_s) in zip(pg0, pg1, pg2, logp1, logp2, m1s, m2s):
                lambda_1, lambda_2, eos_check = \
                    polytrope_or_causal_params_to_lambda_1_lambda_2(
                        pg_0, logp_1, pg_1, logp_2, pg_2, m1_s, m2_s, causal=0)
                all_lambda_1 = np.append(all_lambda_1, lambda_1)
                all_lambda_2 = np.append(all_lambda_2, lambda_2)
                all_eos_check = np.append(all_eos_check, eos_check)
            converted_parameters['lambda_1'] = all_lambda_1
            converted_parameters['lambda_2'] = all_lambda_2
            converted_parameters['eos_check'] = all_eos_check
            for key in float_eos_params.keys():
                converted_parameters[key] = float_eos_params[key]
    elif 'eos_polytrope_gamma_0' and 'eos_polytrope_scaled_pressure_ratio' in converted_parameters.keys():
        converted_parameters = generate_source_frame_parameters(converted_parameters)
        float_eos_params = {}
        max_len = 1
        eos_keys = ['eos_polytrope_gamma_0',
                    'eos_polytrope_gamma_1',
                    'eos_polytrope_gamma_2',
                    'eos_polytrope_scaled_pressure_ratio',
                    'eos_polytrope_scaled_pressure_2',
                    'mass_1_source', 'mass_2_source']
        for key in eos_keys:
            try:
                if (len(converted_parameters[key]) > max_len):
                    max_len = len(converted_parameters[key])
            except TypeError:
                float_eos_params[key] = converted_parameters[key]
        if len(float_eos_params) == len(eos_keys):  # case where all eos params are floats (pinned)
            logp1, logp2 = log_pressure_reparameterization_conversion(
                converted_parameters['eos_polytrope_scaled_pressure_ratio'],
                converted_parameters['eos_polytrope_scaled_pressure_2'])
            converted_parameters['lambda_1'], converted_parameters['lambda_2'], converted_parameters['eos_check'] = \
                polytrope_or_causal_params_to_lambda_1_lambda_2(
                    converted_parameters['eos_polytrope_gamma_0'],
                    logp1,
                    converted_parameters['eos_polytrope_gamma_1'],
                    logp2,
                    converted_parameters['eos_polytrope_gamma_2'],
                    converted_parameters['mass_1_source'],
                    converted_parameters['mass_2_source'],
                    causal=0)
        elif len(float_eos_params) < len(eos_keys):  # case where some or none are floats (pinned)
            for key in float_eos_params.keys():
                converted_parameters[key] = np.ones(max_len) * converted_parameters[key]
            pg0 = converted_parameters['eos_polytrope_gamma_0']
            pg1 = converted_parameters['eos_polytrope_gamma_1']
            pg2 = converted_parameters['eos_polytrope_gamma_2']
            scaledratio = converted_parameters['eos_polytrope_scaled_pressure_ratio']
            scaled_p2 = converted_parameters['eos_polytrope_scaled_pressure_2']
            m1s = converted_parameters['mass_1_source']
            m2s = converted_parameters['mass_2_source']
            all_lambda_1 = np.empty(0)
            all_lambda_2 = np.empty(0)
            all_eos_check = np.empty(0, dtype=bool)
            for (pg_0, pg_1, pg_2, scaled_ratio, scaled_p_2, m1_s,
                    m2_s) in zip(pg0, pg1, pg2, scaledratio, scaled_p2, m1s, m2s):
                logp_1, logp_2 = log_pressure_reparameterization_conversion(scaled_ratio, scaled_p_2)
                lambda_1, lambda_2, eos_check = \
                    polytrope_or_causal_params_to_lambda_1_lambda_2(
                        pg_0, logp_1, pg_1, logp_2, pg_2, m1_s, m2_s, causal=0)
                all_lambda_1 = np.append(all_lambda_1, lambda_1)
                all_lambda_2 = np.append(all_lambda_2, lambda_2)
                all_eos_check = np.append(all_eos_check, eos_check)
            converted_parameters['lambda_1'] = all_lambda_1
            converted_parameters['lambda_2'] = all_lambda_2
            converted_parameters['eos_check'] = all_eos_check
            for key in float_eos_params.keys():
                converted_parameters[key] = float_eos_params[key]
    elif 'eos_v1' in converted_parameters.keys():
        converted_parameters = generate_source_frame_parameters(converted_parameters)
        float_eos_params = {}
        max_len = 1
        eos_keys = ['eos_v1',
                    'eos_v2',
                    'eos_v3',
                    'eos_log10_pressure1_cgs',
                    'eos_log10_pressure2_cgs',
                    'mass_1_source', 'mass_2_source']
        for key in eos_keys:
            try:
                if (len(converted_parameters[key]) > max_len):
                    max_len = len(converted_parameters[key])
            except TypeError:
                float_eos_params[key] = converted_parameters[key]
        if len(float_eos_params) == len(eos_keys):  # case where all eos params are floats (pinned)
            converted_parameters['lambda_1'], converted_parameters['lambda_2'], converted_parameters['eos_check'] = \
                polytrope_or_causal_params_to_lambda_1_lambda_2(
                    converted_parameters['eos_v1'],
                    converted_parameters['eos_log10_pressure1_cgs'],
                    converted_parameters['eos_v2'],
                    converted_parameters['eos_log10_pressure2_cgs'],
                    converted_parameters['eos_v3'],
                    converted_parameters['mass_1_source'],
                    converted_parameters['mass_2_source'],
                    causal=1)
        elif len(float_eos_params) < len(eos_keys):  # case where some or none are floats (pinned)
            for key in float_eos_params.keys():
                converted_parameters[key] = np.ones(max_len) * converted_parameters[key]
            v1 = converted_parameters['eos_v1']
            v2 = converted_parameters['eos_v2']
            v3 = converted_parameters['eos_v3']
            logp1 = converted_parameters['eos_log10_pressure1_cgs']
            logp2 = converted_parameters['eos_log10_pressure2_cgs']
            m1s = converted_parameters['mass_1_source']
            m2s = converted_parameters['mass_2_source']
            all_lambda_1 = np.empty(0)
            all_lambda_2 = np.empty(0)
            all_eos_check = np.empty(0, dtype=bool)
            for (v_1, v_2, v_3, logp_1, logp_2, m1_s, m2_s) in zip(v1, v2, v3, logp1, logp2, m1s, m2s):
                lambda_1, lambda_2, eos_check = \
                    polytrope_or_causal_params_to_lambda_1_lambda_2(
                        v_1, logp_1, v_2, logp_2, v_3, m1_s, m2_s, causal=1)
                all_lambda_1 = np.append(all_lambda_1, lambda_1)
                all_lambda_2 = np.append(all_lambda_2, lambda_2)
                all_eos_check = np.append(all_eos_check, eos_check)
            converted_parameters['lambda_1'] = all_lambda_1
            converted_parameters['lambda_2'] = all_lambda_2
            converted_parameters['eos_check'] = all_eos_check
            for key in float_eos_params.keys():
                converted_parameters[key] = float_eos_params[key]
    elif 'lambda_symmetric' in converted_parameters.keys():
        if 'lambda_antisymmetric' in converted_parameters.keys():
            converted_parameters['lambda_1'], converted_parameters['lambda_2'] =\
                lambda_symmetric_lambda_antisymmetric_to_lambda_1_lambda_2(
                    converted_parameters['lambda_symmetric'],
                    converted_parameters['lambda_antisymmetric'])
        elif 'mass_ratio' in converted_parameters.keys():
            if 'binary_love_uniform' in converted_parameters.keys():
                converted_parameters['lambda_1'], converted_parameters['lambda_2'] =\
                    binary_love_lambda_symmetric_to_lambda_1_lambda_2_manual_marginalisation(
                        converted_parameters['binary_love_uniform'],
                        converted_parameters['lambda_symmetric'],
                        converted_parameters['mass_ratio'])
            else:
                converted_parameters['lambda_1'], converted_parameters['lambda_2'] =\
                    binary_love_lambda_symmetric_to_lambda_1_lambda_2_automatic_marginalisation(
                        converted_parameters['lambda_symmetric'],
                        converted_parameters['mass_ratio'])

    added_keys = [key for key in converted_parameters.keys()
                  if key not in original_keys]

    return converted_parameters, added_keys


def log_pressure_reparameterization_conversion(scaled_pressure_ratio, scaled_pressure_2):
    '''
    Converts the reparameterization joining pressures from
        (scaled_pressure_ratio,scaled_pressure_2) to (log10_pressure_1,log10_pressure_2).
    This reparameterization with a triangular prior (with mode = max)  on scaled_pressure_2
        and a uniform prior on scaled_pressure_ratio
        mimics identical uniform priors on log10_pressure_1 and log10_pressure_2
        where samples with log10_pressure_2 > log10_pressure_1 are rejected.
    This reparameterization allows for a faster initialization.
    A minimum log10_pressure of 33 (in cgs units) is chosen to be slightly higher than the low-density crust EOS
        that is stitched to the dynamic polytrope EOS model in LALSimulation.

    Parameters
    ----------
    scaled_pressure_ratio, scaled_pressure_2: float
        reparameterizations of the dividing pressures

    Returns
    -------
    log10_pressure_1, log10_pressure_2: float
        joining pressures in the original parameterization

    '''
    minimum_pressure = 33.
    log10_pressure_1 = (scaled_pressure_ratio * scaled_pressure_2) + minimum_pressure
    log10_pressure_2 = minimum_pressure + scaled_pressure_2

    return log10_pressure_1, log10_pressure_2


def spectral_pca_to_spectral(gamma_pca_0, gamma_pca_1, gamma_pca_2, gamma_pca_3):
    '''
    Change of basis on parameter space
        from an efficient space to sample in (sample space)
        to the space used in spectral eos model (model space).
    Uses principle component analysis (PCA) to sample spectral gamma parameters more efficiently.
    See arxiv:2001.01747 for the PCA conversion transformation matrix, mean, and standard deviation.
    (Note that this transformation matrix is the inverse of the one referenced
        in order to perform the inverse transformation.)

    Parameters
    ----------
    gamma_pca_0, gamma_pca_1, gamma_pca_2, gamma_pca_3: float
        Spectral gamma PCA parameters to be converted to spectral gamma parameters.

    Returns
    -------
    converted_gamma_parameters:  np.array()
        array of gamma_0, gamma_1, gamma_2, gamma_3 in model space

    '''
    sampled_pca_gammas = np.array([gamma_pca_0, gamma_pca_1, gamma_pca_2, gamma_pca_3])
    transformation_matrix = np.array(
        [
            [0.43801, -0.76705, 0.45143, 0.12646],
            [-0.53573, 0.17169, 0.67968, 0.47070],
            [0.52660, 0.31255, -0.19454, 0.76626],
            [-0.49379, -0.53336, -0.54444, 0.41868]
        ]
    )

    model_space_mean = np.array([0.89421, 0.33878, -0.07894, 0.00393])
    model_space_standard_deviation = np.array([0.35700, 0.25769, 0.05452, 0.00312])
    converted_gamma_parameters = \
        model_space_mean + model_space_standard_deviation * np.dot(transformation_matrix, sampled_pca_gammas)

    return converted_gamma_parameters


def spectral_params_to_lambda_1_lambda_2(gamma_0, gamma_1, gamma_2, gamma_3, mass_1_source, mass_2_source):
    '''
    Converts from the 4 spectral decomposition parameters and the source masses
    to the tidal deformability parameters.

    Parameters
    ----------
    gamma_0, gamma_1, gamma_2, gamma_3: float
        sampled spectral decomposition parameters
    mass_1_source, mass_2_source: float
        sampled component mass parameters converted to source frame in solar masses

    Returns
    -------
    lambda_1, lambda_2: float
        component tidal deformability parameters
    eos_check: bool
        whether or not the equation of state is viable /
            if eos_check = False, lambdas are 0 and the sample is rejected.

    '''
    eos_check = True
    if lalsim_SimNeutronStarEOS4ParamSDGammaCheck(gamma_0, gamma_1, gamma_2, gamma_3) != 0:
        lambda_1 = 0.0
        lambda_2 = 0.0
        eos_check = False
    else:
        eos = lalsim_SimNeutronStarEOS4ParameterSpectralDecomposition(gamma_0, gamma_1, gamma_2, gamma_3)
        if lalsim_SimNeutronStarEOS4ParamSDViableFamilyCheck(gamma_0, gamma_1, gamma_2, gamma_3) != 0:
            lambda_1 = 0.0
            lambda_2 = 0.0
            eos_check = False
        else:
            lambda_1, lambda_2, eos_check = neutron_star_family_physical_check(eos, mass_1_source, mass_2_source)

    return lambda_1, lambda_2, eos_check


def polytrope_or_causal_params_to_lambda_1_lambda_2(
        param1, log10_pressure1_cgs, param2, log10_pressure2_cgs, param3, mass_1_source, mass_2_source, causal):
    """
    Converts parameters from sampled dynamic piecewise polytrope parameters
        to component tidal deformablity parameters.
    Enforces log10_pressure1_cgs < log10_pressure2_cgs.
    Checks number of points in the equation of state for viability.
    Note that subtracting 1 from the log10 pressure in cgs converts it to
        log10 pressure in si units.

    Parameters
    ----------
    param1, param2, param3: float
        either the sampled adiabatic indices in piecewise polytrope model
        or the sampled causal model params v1, v2, v3
    log10_pressure1_cgs, log10_pressure2_cgs: float
        dividing pressures in piecewise polytrope model or causal model
    mass_1_source, mass_2_source: float
        source frame component mass parameters in Msuns
    causal: bool
        whether or not to use causal polytrope model
        1 - causal; 0 - not causal

    Returns
    -------
    lambda_1: float
        tidal deformability parameter associated with mass 1
    lambda_2: float
        tidal deformability parameter associated with mass 2
    eos_check: bool
        whether eos is valid or not

    """
    eos_check = True
    if log10_pressure1_cgs >= log10_pressure2_cgs:
        lambda_1 = 0.0
        lambda_2 = 0.0
        eos_check = False
    else:
        if causal == 0:
            eos = lalsim_SimNeutronStarEOS3PieceDynamicPolytrope(
                param1, log10_pressure1_cgs - 1., param2, log10_pressure2_cgs - 1., param3)
        else:
            eos = lalsim_SimNeutronStarEOS3PieceCausalAnalytic(
                param1, log10_pressure1_cgs - 1., param2, log10_pressure2_cgs - 1., param3)
        if lalsim_SimNeutronStarEOS3PDViableFamilyCheck(
                param1, log10_pressure1_cgs - 1., param2, log10_pressure2_cgs - 1., param3, causal) != 0:
            lambda_1 = 0.0
            lambda_2 = 0.0
            eos_check = False
        else:
            lambda_1, lambda_2, eos_check = neutron_star_family_physical_check(eos, mass_1_source, mass_2_source)

    return lambda_1, lambda_2, eos_check


def neutron_star_family_physical_check(eos, mass_1_source, mass_2_source):
    """
    Takes in a lalsim eos object. Performs causal and max/min mass eos checks.
    Calculates component lambdas if eos object passes causality.
    Returns lambda = 0 if not.

    Parameters
    ----------
    eos: lalsim swig-wrapped eos object
        the neutron star equation of state
    mass_1_source, mass_2_source: float
        source frame component masses 1 and 2 in solar masses

    Returns
    -------
    lambda_1, lambda_2: float
        component tidal deformability parameters
    eos_check: bool
        whether or not the equation of state is physically allowed

    """
    eos_check = True
    family = lalsim_CreateSimNeutronStarFamily(eos)
    max_pseudo_enthalpy = lalsim_SimNeutronStarEOSMaxPseudoEnthalpy(eos)
    max_speed_of_sound = lalsim_SimNeutronStarEOSSpeedOfSoundGeometerized(max_pseudo_enthalpy, eos)
    min_mass = lalsim_SimNeutronStarFamMinimumMass(family) / solar_mass
    max_mass = lalsim_SimNeutronStarMaximumMass(family) / solar_mass
    if max_speed_of_sound <= 1.1 and min_mass <= mass_1_source <= max_mass and min_mass <= mass_2_source <= max_mass:
        lambda_1 = lambda_from_mass_and_family(mass_1_source, family)
        lambda_2 = lambda_from_mass_and_family(mass_2_source, family)
    else:
        lambda_1 = 0.0
        lambda_2 = 0.0
        eos_check = False

    return lambda_1, lambda_2, eos_check


def lambda_from_mass_and_family(mass_i, family):
    """
    Convert from equation of state model parameters to
    component tidal parameters.

    Parameters
    ----------
    family: lalsim family object
        EOS family of type lalsimulation.SimNeutronStarFamily.
    mass_i: Component mass of neutron star in solar masses.

    Returns
    -------
    lambda_1: float
        component tidal deformability parameter

    """
    radius = lalsim_SimNeutronStarRadius(mass_i * solar_mass, family)
    love_number_k2 = lalsim_SimNeutronStarLoveNumberK2(mass_i * solar_mass, family)
    mass_geometrized = mass_i * solar_mass * gravitational_constant / speed_of_light ** 2.
    compactness = mass_geometrized / radius
    lambda_i = (2. / 3.) * love_number_k2 / compactness ** 5.

    return lambda_i


def eos_family_physical_check(eos):
    """
    Function that determines if the EoS family contains
    sufficient number of points before maximum mass is reached.

    e_min is chosen to be sufficiently small so that the entire
    EoS is captured when converting to mass-radius space.

    Returns True if family is valid, False if not.
    """
    e_min = 1.6e-10
    e_central = eos.e_pdat[-1, 1]
    loge_min = np.log(e_min)
    loge_central = np.log(e_central)
    logedat = np.linspace(loge_min, loge_central, num=eos.npts)
    edat = np.exp(logedat)

    # Generate m, r, and k2 lists
    mdat = []
    rdat = []
    k2dat = []
    for i in range(8):
        tov_solver = IntegrateTOV(eos, edat[i])
        m, r, k2 = tov_solver.integrate_TOV()
        mdat.append(m)
        rdat.append(r)
        k2dat.append(k2)

        # Check if maximum mass has been found
        if i > 0 and mdat[i] <= mdat[i - 1]:
            break

    if len(mdat) < 8:
        return False
    else:
        return True


def total_mass_and_mass_ratio_to_component_masses(mass_ratio, total_mass):
    """
    Convert total mass and mass ratio of a binary to its component masses.

    Parameters
    ==========
    mass_ratio: float
        Mass ratio (mass_2/mass_1) of the binary
    total_mass: float
        Total mass of the binary

    Returns
    =======
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object
    """

    mass_1 = total_mass / (1 + mass_ratio)
    mass_2 = mass_1 * mass_ratio
    return mass_1, mass_2


def chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio):
    """
    Convert total mass and mass ratio of a binary to its component masses.

    Parameters
    ==========
    chirp_mass: float
        Chirp mass of the binary
    mass_ratio: float
        Mass ratio (mass_2/mass_1) of the binary

    Returns
    =======
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object
    """
    total_mass = chirp_mass_and_mass_ratio_to_total_mass(chirp_mass=chirp_mass,
                                                         mass_ratio=mass_ratio)
    mass_1, mass_2 = (
        total_mass_and_mass_ratio_to_component_masses(
            total_mass=total_mass, mass_ratio=mass_ratio)
    )
    return mass_1, mass_2


def symmetric_mass_ratio_to_mass_ratio(symmetric_mass_ratio):
    """
    Convert the symmetric mass ratio to the normal mass ratio.

    Parameters
    ==========
    symmetric_mass_ratio: float
        Symmetric mass ratio of the binary

    Returns
    =======
    mass_ratio: float
        Mass ratio of the binary
    """

    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return temp - (temp ** 2 - 1) ** 0.5


def chirp_mass_and_total_mass_to_symmetric_mass_ratio(chirp_mass, total_mass):
    """
    Convert chirp mass and total mass of a binary to its symmetric mass ratio.

    Parameters
    ==========
    chirp_mass: float
        Chirp mass of the binary
    total_mass: float
        Total mass of the binary

    Returns
    =======
    symmetric_mass_ratio: float
        Symmetric mass ratio of the binary
    """

    return (chirp_mass / total_mass) ** (5 / 3)


def chirp_mass_and_primary_mass_to_mass_ratio(chirp_mass, mass_1):
    """
    Convert chirp mass and mass ratio of a binary to its total mass.

    Rearranging the relation for chirp mass (as a function of mass_1 and
    mass_2) and q = mass_2 / mass_1, it can be shown that

        (chirp_mass/mass_1)^5 = q^3 / (1 + q)

    Solving for q, we find the relation expressed in python below for q.

    Parameters
    ==========
    chirp_mass: float
        Chirp mass of the binary
    mass_1: float
        The primary mass

    Returns
    =======
    mass_ratio: float
        Mass ratio (mass_2/mass_1) of the binary
    """
    a = (chirp_mass / mass_1) ** 5
    t0 = np.cbrt(9 * a + np.sqrt(3) * np.sqrt(27 * a ** 2 - 4 * a ** 3))
    t1 = np.cbrt(2) * 3 ** (2 / 3)
    t2 = np.cbrt(2 / 3) * a
    return t2 / t0 + t0 / t1


def chirp_mass_and_mass_ratio_to_total_mass(chirp_mass, mass_ratio):
    """
    Convert chirp mass and mass ratio of a binary to its total mass.

    Parameters
    ==========
    chirp_mass: float
        Chirp mass of the binary
    mass_ratio: float
        Mass ratio (mass_2/mass_1) of the binary

    Returns
    =======
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object
    """

    with np.errstate(invalid="ignore"):
        return chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio ** 0.6


def component_masses_to_chirp_mass(mass_1, mass_2):
    """
    Convert the component masses of a binary to its chirp mass.

    Parameters
    ==========
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object

    Returns
    =======
    chirp_mass: float
        Chirp mass of the binary
    """

    return (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2


def component_masses_to_total_mass(mass_1, mass_2):
    """
    Convert the component masses of a binary to its total mass.

    Parameters
    ==========
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object

    Returns
    =======
    total_mass: float
        Total mass of the binary
    """

    return mass_1 + mass_2


def component_masses_to_symmetric_mass_ratio(mass_1, mass_2):
    """
    Convert the component masses of a binary to its symmetric mass ratio.

    Parameters
    ==========
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object

    Returns
    =======
    symmetric_mass_ratio: float
        Symmetric mass ratio of the binary
    """

    return np.minimum((mass_1 * mass_2) / (mass_1 + mass_2) ** 2, 1 / 4)


def component_masses_to_mass_ratio(mass_1, mass_2):
    """
    Convert the component masses of a binary to its chirp mass.

    Parameters
    ==========
    mass_1: float
        Mass of the heavier object
    mass_2: float
        Mass of the lighter object

    Returns
    =======
    mass_ratio: float
        Mass ratio of the binary
    """

    return mass_2 / mass_1


def mass_1_and_chirp_mass_to_mass_ratio(mass_1, chirp_mass):
    """
    Calculate mass ratio from mass_1 and chirp_mass.

    This involves solving mc = m1 * q**(3/5) / (1 + q)**(1/5).

    Parameters
    ==========
    mass_1: float
        Mass of the heavier object
    chirp_mass: float
        Mass of the lighter object

    Returns
    =======
    mass_ratio: float
        Mass ratio of the binary
    """
    temp = (chirp_mass / mass_1) ** 5
    mass_ratio = (2 / 3 / (3 ** 0.5 * (27 * temp ** 2 - 4 * temp ** 3) ** 0.5 +
                           9 * temp)) ** (1 / 3) * temp + \
                 ((3 ** 0.5 * (27 * temp ** 2 - 4 * temp ** 3) ** 0.5 +
                   9 * temp) / (2 * 3 ** 2)) ** (1 / 3)
    return mass_ratio


def mass_2_and_chirp_mass_to_mass_ratio(mass_2, chirp_mass):
    """
    Calculate mass ratio from mass_1 and chirp_mass.

    This involves solving mc = m2 * (1/q)**(3/5) / (1 + (1/q))**(1/5).

    Parameters
    ==========
    mass_2: float
        Mass of the lighter object
    chirp_mass: float
        Chirp mass of the binary

    Returns
    =======
    mass_ratio: float
        Mass ratio of the binary
    """
    # Passing mass_2, the expression from the function above
    # returns 1/q (because chirp mass is invariant under
    # mass_1 <-> mass_2)
    return 1 / mass_1_and_chirp_mass_to_mass_ratio(mass_2, chirp_mass)


def lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2):
    """
    Convert from individual tidal parameters to domainant tidal term.

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Parameters
    ==========
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.
    mass_1: float
        Mass of more massive neutron star.
    mass_2: float
        Mass of less massive neutron star.

    Returns
    =======
    lambda_tilde: float
        Dominant tidal term.

    """
    eta = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
    lambda_plus = lambda_1 + lambda_2
    lambda_minus = lambda_1 - lambda_2
    lambda_tilde = 8 / 13 * (
        (1 + 7 * eta - 31 * eta**2) * lambda_plus +
        (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2) * lambda_minus)

    return lambda_tilde


def lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2):
    """
    Convert from individual tidal parameters to second domainant tidal term.

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Parameters
    ==========
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.
    mass_1: float
        Mass of more massive neutron star.
    mass_2: float
        Mass of less massive neutron star.

    Returns
    =======
    delta_lambda_tilde: float
        Second dominant tidal term.
    """
    eta = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
    lambda_plus = lambda_1 + lambda_2
    lambda_minus = lambda_1 - lambda_2
    delta_lambda_tilde = 1 / 2 * (
        (1 - 4 * eta) ** 0.5 * (1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2) *
        lambda_plus + (1 - 15910 / 1319 * eta + 32850 / 1319 * eta ** 2 +
                       3380 / 1319 * eta ** 3) * lambda_minus)
    return delta_lambda_tilde


def lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(
        lambda_tilde, delta_lambda_tilde, mass_1, mass_2):
    """
    Convert from dominant tidal terms to individual tidal parameters.

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Parameters
    ==========
    lambda_tilde: float
        Dominant tidal term.
    delta_lambda_tilde: float
        Secondary tidal term.
    mass_1: float
        Mass of more massive neutron star.
    mass_2: float
        Mass of less massive neutron star.

    Returns
    =======
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.

    """
    eta = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
    coefficient_1 = (1 + 7 * eta - 31 * eta**2)
    coefficient_2 = (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2)
    coefficient_3 = (1 - 4 * eta)**0.5 *\
                    (1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2)
    coefficient_4 = (1 - 15910 / 1319 * eta + 32850 / 1319 * eta**2 +
                     3380 / 1319 * eta**3)
    lambda_1 =\
        (13 * lambda_tilde / 8 * (coefficient_3 - coefficient_4) -
         2 * delta_lambda_tilde * (coefficient_1 - coefficient_2))\
        / ((coefficient_1 + coefficient_2) * (coefficient_3 - coefficient_4) -
           (coefficient_1 - coefficient_2) * (coefficient_3 + coefficient_4))
    lambda_2 =\
        (13 * lambda_tilde / 8 * (coefficient_3 + coefficient_4) -
         2 * delta_lambda_tilde * (coefficient_1 + coefficient_2)) \
        / ((coefficient_1 - coefficient_2) * (coefficient_3 + coefficient_4) -
           (coefficient_1 + coefficient_2) * (coefficient_3 - coefficient_4))

    return lambda_1, lambda_2


def lambda_tilde_to_lambda_1_lambda_2(
        lambda_tilde, mass_1, mass_2):
    """
    Convert from dominant tidal term to individual tidal parameters
    assuming lambda_1 * mass_1**5 = lambda_2 * mass_2**5.

    See, e.g., Wade et al., https://arxiv.org/pdf/1402.5156.pdf.

    Parameters
    ==========
    lambda_tilde: float
        Dominant tidal term.
    mass_1: float
        Mass of more massive neutron star.
    mass_2: float
        Mass of less massive neutron star.

    Returns
    =======
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.
    """
    eta = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
    q = mass_2 / mass_1
    lambda_1 = 13 / 8 * lambda_tilde / (
        (1 + 7 * eta - 31 * eta**2) * (1 + q**-5) +
        (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2) * (1 - q**-5))
    lambda_2 = lambda_1 / q**5
    return lambda_1, lambda_2


def lambda_1_lambda_2_to_lambda_symmetric(lambda_1, lambda_2):
    """
    Convert from individual tidal parameters to symmetric tidal term.

    See, e.g., Yagi & Yunes, https://arxiv.org/abs/1512.02639

    Parameters
    ==========
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.

    Returns
    ======
    lambda_symmetric: float
        Symmetric tidal parameter.
    """
    lambda_symmetric = (lambda_2 + lambda_1) / 2.
    return lambda_symmetric


def lambda_1_lambda_2_to_lambda_antisymmetric(lambda_1, lambda_2):
    """
    Convert from individual tidal parameters to antisymmetric tidal term.

    See, e.g., Yagi & Yunes, https://arxiv.org/abs/1512.02639

    Parameters
    ==========
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.

    Returns
    ======
    lambda_antisymmetric: float
        Antisymmetric tidal parameter.
    """
    lambda_antisymmetric = (lambda_2 - lambda_1) / 2.
    return lambda_antisymmetric


def lambda_symmetric_lambda_antisymmetric_to_lambda_1(lambda_symmetric, lambda_antisymmetric):
    """
    Convert from symmetric and  antisymmetric tidal terms to lambda_1

    See, e.g., Yagi & Yunes, https://arxiv.org/abs/1512.02639

    Parameters
    ==========
    lambda_symmetric: float
        Symmetric tidal parameter.
    lambda_antisymmetric: float
        Antisymmetric tidal parameter.

    Returns
    ======
    lambda_1: float
        Tidal parameter of more massive neutron star.
    """
    lambda_1 = lambda_symmetric - lambda_antisymmetric
    return lambda_1


def lambda_symmetric_lambda_antisymmetric_to_lambda_2(lambda_symmetric, lambda_antisymmetric):
    """
    Convert from symmetric and antisymmetric tidal terms to lambda_2

    See, e.g., Yagi & Yunes, https://arxiv.org/abs/1512.02639

    Parameters
    ==========
    lambda_symmetric: float
        Symmetric tidal parameter.
    lambda_antisymmetric: float
        Antisymmetric tidal parameter.

    Returns
    ======
    lambda_2: float
        Tidal parameter of less massive neutron star.
    """
    lambda_2 = lambda_symmetric + lambda_antisymmetric
    return lambda_2


def lambda_symmetric_lambda_antisymmetric_to_lambda_1_lambda_2(lambda_symmetric, lambda_antisymmetric):
    """
    Convert from symmetric and antisymmetric tidal terms to lambda_1 and lambda_2

    See, e.g., Yagi & Yunes, https://arxiv.org/abs/1512.02639

    Parameters
    ==========
    lambda_symmetric: float
        Symmetric tidal parameter.
    lambda_antisymmetric: float
        Antisymmetric tidal parameter.

    Returns
    ======
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.
    """
    lambda_1 = lambda_symmetric_lambda_antisymmetric_to_lambda_1(lambda_symmetric, lambda_antisymmetric)
    lambda_2 = lambda_symmetric_lambda_antisymmetric_to_lambda_2(lambda_symmetric, lambda_antisymmetric)
    return lambda_1, lambda_2


def binary_love_fit_lambda_symmetric_mass_ratio_to_lambda_antisymmetric(lambda_symmetric, mass_ratio):

    """
    Convert from symmetric tidal terms and mass ratio to antisymmetric tidal terms
    using BinaryLove relations.
    This function does only the fit itself, and doesn't account for marginalisation
    over the uncertanties in the fit

    See Yagi & Yunes, https://arxiv.org/abs/1512.02639
    and Chatziioannou, Haster, Zimmerman, https://arxiv.org/abs/1804.03221

    This function will use the implementation from the CHZ paper, which
    marginalises over the uncertainties in the BinaryLove fits.
    This is also implemented in LALInference through the function
    LALInferenceBinaryLove.

    Parameters
    ==========
    lambda_symmetric: float
        Symmetric tidal parameter.
    mass_ratio: float
        Mass ratio (mass_2/mass_1, with mass_2 < mass_1) of the binary

    Returns
    ======
    lambda_antisymmetric: float
        Antisymmetric tidal parameter.
    """
    lambda_symmetric_m1o5 = np.power(lambda_symmetric, -1. / 5.)
    lambda_symmetric_m2o5 = lambda_symmetric_m1o5 * lambda_symmetric_m1o5
    lambda_symmetric_m3o5 = lambda_symmetric_m2o5 * lambda_symmetric_m1o5

    q = mass_ratio
    q2 = np.square(mass_ratio)

    # Eqn.2 from CHZ, incorporating the dependence on mass ratio

    n_polytropic = 0.743  # average polytropic index for the EoSs included in the fit
    q_for_Fnofq = np.power(q, 10. / (3. - n_polytropic))
    Fnofq = (1. - q_for_Fnofq) / (1. + q_for_Fnofq)

    # b_ij and c_ij coefficients are given in Table I of CHZ

    b11 = -27.7408
    b12 = 8.42358
    b21 = 122.686
    b22 = -19.7551
    b31 = -175.496
    b32 = 133.708

    c11 = -25.5593
    c12 = 5.58527
    c21 = 92.0337
    c22 = 26.8586
    c31 = -70.247
    c32 = -56.3076

    # Eqn 1 from CHZ, giving the lambda_antisymmetric_fitOnly
    # not yet accounting for the uncertainty in the fit

    numerator = 1.0 + \
        (b11 * q * lambda_symmetric_m1o5) + (b12 * q2 * lambda_symmetric_m1o5) + \
        (b21 * q * lambda_symmetric_m2o5) + (b22 * q2 * lambda_symmetric_m2o5) + \
        (b31 * q * lambda_symmetric_m3o5) + (b32 * q2 * lambda_symmetric_m3o5)

    denominator = 1.0 + \
        (c11 * q * lambda_symmetric_m1o5) + (c12 * q2 * lambda_symmetric_m1o5) + \
        (c21 * q * lambda_symmetric_m2o5) + (c22 * q2 * lambda_symmetric_m2o5) + \
        (c31 * q * lambda_symmetric_m3o5) + (c32 * q2 * lambda_symmetric_m3o5)

    lambda_antisymmetric_fitOnly = Fnofq * lambda_symmetric * numerator / denominator

    return lambda_antisymmetric_fitOnly


def binary_love_lambda_symmetric_to_lambda_1_lambda_2_manual_marginalisation(binary_love_uniform,
                                                                             lambda_symmetric, mass_ratio):
    """
    Convert from symmetric tidal terms to lambda_1 and lambda_2
    using BinaryLove relations

    See Yagi & Yunes, https://arxiv.org/abs/1512.02639
    and Chatziioannou, Haster, Zimmerman, https://arxiv.org/abs/1804.03221

    This function will use the implementation from the CHZ paper, which
    marginalises over the uncertainties in the BinaryLove fits.
    This is also implemented in LALInference through the function
    LALInferenceBinaryLove.

    Parameters
    ==========
    binary_love_uniform: float (defined in the range [0,1])
        Uniformly distributed variable used in BinaryLove uncertainty marginalisation
    lambda_symmetric: float
        Symmetric tidal parameter.
    mass_ratio: float
        Mass ratio (mass_2/mass_1, with mass_2 < mass_1) of the binary

    Returns
    ======
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.
    """
    lambda_antisymmetric_fitOnly = binary_love_fit_lambda_symmetric_mass_ratio_to_lambda_antisymmetric(lambda_symmetric,
                                                                                                       mass_ratio)

    lambda_symmetric_sqrt = np.sqrt(lambda_symmetric)

    q = mass_ratio
    q2 = np.square(mass_ratio)

    # mu_i and sigma_i coefficients are given in Table II of CHZ

    mu_1 = 137.1252739
    mu_2 = -32.8026613
    mu_3 = 0.5168637
    mu_4 = -11.2765281
    mu_5 = 14.9499544
    mu_6 = - 4.6638851

    sigma_1 = -0.0000739
    sigma_2 = 0.0103778
    sigma_3 = 0.4581717
    sigma_4 = -0.8341913
    sigma_5 = -201.4323962
    sigma_6 = 273.9268276
    sigma_7 = -71.2342246

    # Eqn 6 from CHZ, correction on fit for lambdaA caused by
    # uncertainty in the mean of the lambdaS residual fit,
    # using coefficients mu_1, mu_2 and mu_3 from Table II of CHZ

    lambda_antisymmetric_lambda_symmetric_meanCorr = \
        (mu_1 / (lambda_symmetric * lambda_symmetric)) + \
        (mu_2 / lambda_symmetric) + mu_3

    # Eqn 8 from CHZ, correction on fit for lambdaA caused by
    # uncertainty in the standard deviation of lambdaS residual fit,
    # using coefficients sigma_1, sigma_2, sigma_3 and  sigma_4 from Table II

    lambda_antisymmetric_lambda_symmetric_stdCorr = \
        (sigma_1 * lambda_symmetric * lambda_symmetric_sqrt) + \
        (sigma_2 * lambda_symmetric) + \
        (sigma_3 * lambda_symmetric_sqrt) + sigma_4

    # Eqn 7, correction on fit for lambdaA caused by
    # uncertainty in the mean of the q residual fit,
    # using coefficients mu_4, mu_5 and mu_6 from Table II

    lambda_antisymmetric_mass_ratio_meanCorr = \
        (mu_4 * q2) + (mu_5 * q) + mu_6

    # Eqn 9 from CHZ, correction on fit for lambdaA caused by
    # uncertainty in the standard deviation of the q residual fit,
    # using coefficients sigma_5, sigma_6 and sigma_7 from Table II

    lambda_antisymmetric_mass_ratio_stdCorr = \
        (sigma_5 * q2) + (sigma_6 * q) + sigma_7

    # Eqn 4 from CHZ, averaging the corrections from the
    # mean of the residual fits

    lambda_antisymmetric_meanCorr = \
        (lambda_antisymmetric_lambda_symmetric_meanCorr +
            lambda_antisymmetric_mass_ratio_meanCorr) / 2.

    # Eqn 5 from CHZ, averaging the corrections from the
    # standard deviations of the residual fits

    lambda_antisymmetric_stdCorr = \
        np.sqrt(np.square(lambda_antisymmetric_lambda_symmetric_stdCorr) +
                np.square(lambda_antisymmetric_mass_ratio_stdCorr))

    # Draw a correction on the fit from a
    # Gaussian distribution with width lambda_antisymmetric_stdCorr
    # this is done by sampling a percent point function  (inverse cdf)
    # through a U{0,1} variable called binary_love_uniform

    lambda_antisymmetric_scatter = norm.ppf(binary_love_uniform, loc=0.,
                                            scale=lambda_antisymmetric_stdCorr)

    # Add the correction of the residual mean
    # and the Gaussian scatter to the lambda_antisymmetric_fitOnly value

    lambda_antisymmetric = lambda_antisymmetric_fitOnly + \
        (lambda_antisymmetric_meanCorr + lambda_antisymmetric_scatter)

    lambda_1 = lambda_symmetric_lambda_antisymmetric_to_lambda_1(lambda_symmetric, lambda_antisymmetric)
    lambda_2 = lambda_symmetric_lambda_antisymmetric_to_lambda_2(lambda_symmetric, lambda_antisymmetric)

    # The BinaryLove model is only physically valid where
    # lambda_2 > lambda_1 as it assumes mass_1 > mass_2
    # It also assumes both lambda1 and lambda2 to be positive
    # This is an explicit feature of the "raw" model fit,
    # but since this implementation also incorporates
    # marginalisation over the fit uncertainty, there can
    # be instances where those assumptions are randomly broken.
    # For those cases, set lambda_1 and lambda_2 to negative
    # values, which in turn will cause (effectively all) the
    # waveform models in LALSimulation to fail, thus setting
    # logL = -infinity

    # if np.greater(lambda_1, lambda_2) or np.less(lambda_1, 0) or np.less(lambda_2, 0):
    #    lambda_1 = -np.inf
    #    lambda_2 = -np.inf

    # For now set this through an explicit constraint prior instead

    return lambda_1, lambda_2


def binary_love_lambda_symmetric_to_lambda_1_lambda_2_automatic_marginalisation(lambda_symmetric, mass_ratio):
    """
    Convert from symmetric tidal terms to lambda_1 and lambda_2
    using BinaryLove relations

    See Yagi & Yunes, https://arxiv.org/abs/1512.02639
    and Chatziioannou, Haster, Zimmerman, https://arxiv.org/abs/1804.03221

    This function will use the implementation from the CHZ paper, which
    marginalises over the uncertainties in the BinaryLove fits.
    This is also implemented in LALInference through the function
    LALInferenceBinaryLove.

    This function should be used when the BinaryLove marginalisation wasn't
    explicitly active in the sampling stage. It will draw a random number in U[0,1]
    here instead.

    Parameters
    ==========
    lambda_symmetric: float
        Symmetric tidal parameter.
    mass_ratio: float
        Mass ratio (mass_2/mass_1, with mass_2 < mass_1) of the binary

    Returns
    ======
    lambda_1: float
        Tidal parameter of more massive neutron star.
    lambda_2: float
        Tidal parameter of less massive neutron star.
    """
    from ..core.utils.random import rng

    binary_love_uniform = rng.uniform(0, 1, len(lambda_symmetric))

    lambda_1, lambda_2 = binary_love_lambda_symmetric_to_lambda_1_lambda_2_manual_marginalisation(
        binary_love_uniform, lambda_symmetric, mass_ratio)

    return lambda_1, lambda_2


def _generate_all_cbc_parameters(sample, defaults, base_conversion,
                                 likelihood=None, priors=None, npool=1):
    """Generate all cbc parameters, helper function for BBH/BNS"""
    output_sample = sample.copy()
    waveform_defaults = defaults
    for key in waveform_defaults:
        try:
            output_sample[key] = \
                likelihood.waveform_generator.waveform_arguments[key]
        except (KeyError, AttributeError):
            default = waveform_defaults[key]
            output_sample[key] = default
            logger.debug('Assuming {} = {}'.format(key, default))

    output_sample = fill_from_fixed_priors(output_sample, priors)
    output_sample, _ = base_conversion(output_sample)
    if likelihood is not None:
        compute_per_detector_log_likelihoods(
            samples=output_sample, likelihood=likelihood, npool=npool)

        marginalized_parameters = getattr(likelihood, "_marginalized_parameters", list())
        if len(marginalized_parameters) > 0:
            try:
                generate_posterior_samples_from_marginalized_likelihood(
                    samples=output_sample, likelihood=likelihood, npool=npool)
            except MarginalizedLikelihoodReconstructionError as e:
                logger.warning(
                    "Marginalised parameter reconstruction failed with message "
                    "{}. Some parameters may not have the intended "
                    "interpretation.".format(e)
                )
        if priors is not None:
            misnamed_marginalizations = dict(
                luminosity_distance="distance",
                geocent_time="time",
                recalib_index="calibration",
            )
            for par in marginalized_parameters:
                name = misnamed_marginalizations.get(par, par)
                if (
                    getattr(likelihood, f'{name}_marginalization', False)
                    and par in likelihood.priors
                ):
                    priors[par] = likelihood.priors[par]

        if (
            not getattr(likelihood, "reference_frame", "sky") == "sky"
            or not getattr(likelihood, "time_reference", "geocenter") == "geocenter"
        ):
            try:
                generate_sky_frame_parameters(
                    samples=output_sample, likelihood=likelihood
                )
            except TypeError:
                logger.info(
                    "Failed to generate sky frame parameters for type {}"
                    .format(type(output_sample))
                )
        compute_snrs(output_sample, likelihood, npool=npool)
    for key, func in zip(["mass", "spin", "source frame"], [
            generate_mass_parameters, generate_spin_parameters,
            generate_source_frame_parameters]):
        try:
            output_sample = func(output_sample)
        except KeyError as e:
            logger.info(
                "Generation of {} parameters failed with message {}".format(
                    key, e))
    return output_sample


def generate_all_bbh_parameters(sample, likelihood=None, priors=None, npool=1):
    """
    From either a single sample or a set of samples fill in all missing
    BBH parameters, in place.

    Parameters
    ==========
    sample: dict or pandas.DataFrame
        Samples to fill in with extra parameters, this may be either an
        injection or posterior samples.
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient, optional
        GravitationalWaveTransient used for sampling, used for waveform and
        likelihood.interferometers.
    priors: dict, optional
        Dictionary of prior objects, used to fill in non-sampled parameters.
    """
    waveform_defaults = {
        'reference_frequency': 50.0, 'waveform_approximant': 'IMRPhenomPv2',
        'minimum_frequency': 20.0}
    output_sample = _generate_all_cbc_parameters(
        sample, defaults=waveform_defaults,
        base_conversion=convert_to_lal_binary_black_hole_parameters,
        likelihood=likelihood, priors=priors, npool=npool)
    return output_sample


def generate_all_bns_parameters(sample, likelihood=None, priors=None, npool=1):
    """
    From either a single sample or a set of samples fill in all missing
    BNS parameters, in place.

    Since we assume BNS waveforms are aligned, component spins won't be
    calculated.

    Parameters
    ==========
    sample: dict or pandas.DataFrame
        Samples to fill in with extra parameters, this may be either an
        injection or posterior samples.
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient, optional
        GravitationalWaveTransient used for sampling, used for waveform and
        likelihood.interferometers.
    priors: dict, optional
        Dictionary of prior objects, used to fill in non-sampled parameters.
    npool: int, (default=1)
        If given, perform generation (where possible) using a multiprocessing pool

    """
    waveform_defaults = {
        'reference_frequency': 50.0, 'waveform_approximant': 'TaylorF2',
        'minimum_frequency': 20.0}
    output_sample = _generate_all_cbc_parameters(
        sample, defaults=waveform_defaults,
        base_conversion=convert_to_lal_binary_neutron_star_parameters,
        likelihood=likelihood, priors=priors, npool=npool)
    try:
        output_sample = generate_tidal_parameters(output_sample)
    except KeyError as e:
        logger.debug(
            "Generation of tidal parameters failed with message {}".format(e))
    return output_sample


def generate_specific_parameters(sample, parameters):
    """
    Generate a specific subset of parameters that can be generated.

    Parameters
    ----------
    sample: dict
        The input sample to be converted.
    parameters: list
        The list of parameters to return.

    Returns
    -------
    output_sample: dict
        The converted parameters

    Notes
    -----
    This is _not_ an optimized function. Under the hood, it generates all
    possible parameters and then downselects.

    If the passed :code:`parameters` do not include the input parameters,
    those will not be returned.
    """
    updated_sample = generate_all_bns_parameters(sample=sample.copy())
    output_sample = sample.__class__()
    for key in parameters:
        if key in updated_sample:
            output_sample[key] = updated_sample[key]
        else:
            raise KeyError("{} not in converted sample.".format(key))
    return output_sample


def fill_from_fixed_priors(sample, priors):
    """Add parameters with delta function prior to the data frame/dictionary.

    Parameters
    ==========
    sample: dict
        A dictionary or data frame
    priors: dict
        A dictionary of priors

    Returns
    =======
    dict:
    """
    output_sample = sample.copy()
    if priors is not None:
        for name in priors:
            if isinstance(priors[name], DeltaFunction):
                output_sample[name] = priors[name].peak
    return output_sample


def generate_component_masses(sample, require_add=False, source=False):
    """"
    Add the component masses to the dataframe/dictionary
    We add:
        mass_1, mass_2
    Or if source=True
        mass_1_source, mass_2_source
    We also add any other masses which may be necessary for
    intermediate steps, i.e. typically the  total mass is necessary, along
    with the mass ratio, so these will usually be added to the dictionary

    If `require_add` is True, then having an incomplete set of mass
    parameters (so that the component mass parameters cannot be added)
    will throw an error, otherwise it will quietly add nothing to the
    dictionary.

    Parameters
    =========
    sample : dict
        The input dictionary with at least one
        component with overall mass scaling (i.e.
        chirp_mass, mass_1, mass_2, total_mass) and
        then any other mass parameter.
    source : bool, default False
        If True, then perform the conversions for source mass parameters
        i.e. mass_1_source instead of mass_1

    Returns
    dict : the updated dictionary
    """
    def check_and_return_quietly(require_add, sample):
        if require_add:
            raise KeyError("Insufficient mass parameters in input dictionary")
        else:
            return sample
    output_sample = sample.copy()

    if source:
        mass_1_key = "mass_1_source"
        mass_2_key = "mass_2_source"
        total_mass_key = "total_mass_source"
        chirp_mass_key = "chirp_mass_source"
    else:
        mass_1_key = "mass_1"
        mass_2_key = "mass_2"
        total_mass_key = "total_mass"
        chirp_mass_key = "chirp_mass"

    if mass_1_key in sample.keys():
        if mass_2_key in sample.keys():
            return output_sample
        if total_mass_key in sample.keys():
            output_sample[mass_2_key] = output_sample[total_mass_key] - (
                output_sample[mass_1_key]
            )
            return output_sample

        elif "mass_ratio" in sample.keys():
            pass
        elif "symmetric_mass_ratio" in sample.keys():
            output_sample["mass_ratio"] = (
                symmetric_mass_ratio_to_mass_ratio(
                    output_sample["symmetric_mass_ratio"])
            )
        elif chirp_mass_key in sample.keys():
            output_sample["mass_ratio"] = (
                mass_1_and_chirp_mass_to_mass_ratio(
                    mass_1=output_sample[mass_1_key],
                    chirp_mass=output_sample[chirp_mass_key])
            )
        else:
            return check_and_return_quietly(require_add, sample)

        output_sample[mass_2_key] = (
            output_sample["mass_ratio"] * output_sample[mass_1_key]
        )

        return output_sample

    elif mass_2_key in sample.keys():
        # mass_1 is not in the dict
        if total_mass_key in sample.keys():
            output_sample[mass_1_key] = (
                output_sample[total_mass_key] - output_sample[mass_2_key]
            )
            return output_sample
        elif "mass_ratio" in sample.keys():
            pass
        elif "symmetric_mass_ratio" in sample.keys():
            output_sample["mass_ratio"] = (
                symmetric_mass_ratio_to_mass_ratio(
                    output_sample["symmetric_mass_ratio"])
            )
        elif chirp_mass_key in sample.keys():
            output_sample["mass_ratio"] = (
                mass_2_and_chirp_mass_to_mass_ratio(
                    mass_2=output_sample[mass_2_key],
                    chirp_mass=output_sample[chirp_mass_key])
            )
        else:
            check_and_return_quietly(require_add, sample)

        output_sample[mass_1_key] = 1 / output_sample["mass_ratio"] * (
            output_sample[mass_2_key]
        )

        return output_sample

    # Only if neither mass_1 or mass_2 is in the input sample
    if total_mass_key in sample.keys():
        if "mass_ratio" in sample.keys():
            pass  # We have everything we need already
        elif "symmetric_mass_ratio" in sample.keys():
            output_sample["mass_ratio"] = (
                symmetric_mass_ratio_to_mass_ratio(
                    output_sample["symmetric_mass_ratio"])
            )
        elif chirp_mass_key in sample.keys():
            output_sample["symmetric_mass_ratio"] = (
                chirp_mass_and_total_mass_to_symmetric_mass_ratio(
                    chirp_mass=output_sample[chirp_mass_key],
                    total_mass=output_sample[total_mass_key])
            )
            output_sample["mass_ratio"] = (
                symmetric_mass_ratio_to_mass_ratio(
                    output_sample["symmetric_mass_ratio"])
            )
        else:
            return check_and_return_quietly(require_add, sample)

    elif chirp_mass_key in sample.keys():
        if "mass_ratio" in sample.keys():
            pass
        elif "symmetric_mass_ratio" in sample.keys():
            output_sample["mass_ratio"] = (
                symmetric_mass_ratio_to_mass_ratio(
                    sample["symmetric_mass_ratio"])
            )
        else:
            return check_and_return_quietly(require_add, sample)

        output_sample[total_mass_key] = (
            chirp_mass_and_mass_ratio_to_total_mass(
                chirp_mass=output_sample[chirp_mass_key],
                mass_ratio=output_sample["mass_ratio"])
        )

    # We haven't matched any of the criteria
    if total_mass_key not in output_sample.keys() or (
            "mass_ratio" not in output_sample.keys()):
        return check_and_return_quietly(require_add, sample)
    mass_1, mass_2 = (
        total_mass_and_mass_ratio_to_component_masses(
            total_mass=output_sample[total_mass_key],
            mass_ratio=output_sample["mass_ratio"])
    )
    output_sample[mass_1_key] = mass_1
    output_sample[mass_2_key] = mass_2

    return output_sample


def generate_mass_parameters(sample, source=False):
    """
    Add the known mass parameters to the data frame/dictionary.  We do
    not recompute keys already present in the dictionary

    We add, potentially:
        chirp_mass, total_mass, symmetric_mass_ratio, mass_ratio, mass_1, mass_2
    Or if source=True:
        chirp_mass_source, total_mass_source, symmetric_mass_ratio, mass_ratio, mass_1_source, mass_2_source

    Parameters
    ==========
    sample : dict
        The input dictionary with two "spanning" mass parameters
        e.g. (mass_1, mass_2), or (chirp_mass, mass_ratio), but not e.g. only
        (mass_ratio, symmetric_mass_ratio)
    source : bool, default False
        If True, then perform the conversions for source mass parameters
        i.e. mass_1_source instead of mass_1

    Returns
    =======
    dict: The updated dictionary

    """
    # Only add the parameters if they're not already present
    intermediate_sample = generate_component_masses(sample, source=source)
    output_sample = intermediate_sample.copy()

    if source:
        mass_1_key = 'mass_1_source'
        mass_2_key = 'mass_2_source'
        total_mass_key = 'total_mass_source'
        chirp_mass_key = 'chirp_mass_source'
    else:
        mass_1_key = 'mass_1'
        mass_2_key = 'mass_2'
        total_mass_key = 'total_mass'
        chirp_mass_key = 'chirp_mass'

    if chirp_mass_key not in output_sample.keys():
        output_sample[chirp_mass_key] = (
            component_masses_to_chirp_mass(output_sample[mass_1_key],
                                           output_sample[mass_2_key])
        )
    if total_mass_key not in output_sample.keys():
        output_sample[total_mass_key] = (
            component_masses_to_total_mass(output_sample[mass_1_key],
                                           output_sample[mass_2_key])
        )
    if 'symmetric_mass_ratio' not in output_sample.keys():
        output_sample['symmetric_mass_ratio'] = (
            component_masses_to_symmetric_mass_ratio(output_sample[mass_1_key],
                                                     output_sample[mass_2_key])
        )
    if 'mass_ratio' not in output_sample.keys():
        output_sample['mass_ratio'] = (
            component_masses_to_mass_ratio(output_sample[mass_1_key],
                                           output_sample[mass_2_key])
        )

    return output_sample


def generate_spin_parameters(sample):
    """
    Add all spin parameters to the data frame/dictionary.

    We add:
        cartesian spin components, chi_eff, chi_p cos tilt 1, cos tilt 2

    Parameters
    ==========
    sample : dict, pandas.DataFrame
        The input dictionary with some spin parameters

    Returns
    =======
    dict: The updated dictionary

    """
    output_sample = sample.copy()

    output_sample = generate_component_spins(output_sample)

    output_sample['chi_eff'] = (output_sample['spin_1z'] +
                                output_sample['spin_2z'] *
                                output_sample['mass_ratio']) /\
                               (1 + output_sample['mass_ratio'])

    output_sample['chi_1_in_plane'] = np.sqrt(
        output_sample['spin_1x'] ** 2 + output_sample['spin_1y'] ** 2
    )
    output_sample['chi_2_in_plane'] = np.sqrt(
        output_sample['spin_2x'] ** 2 + output_sample['spin_2y'] ** 2
    )

    output_sample['chi_p'] = np.maximum(
        output_sample['chi_1_in_plane'],
        (4 * output_sample['mass_ratio'] + 3) /
        (3 * output_sample['mass_ratio'] + 4) * output_sample['mass_ratio'] *
        output_sample['chi_2_in_plane'])

    try:
        output_sample['cos_tilt_1'] = np.cos(output_sample['tilt_1'])
        output_sample['cos_tilt_2'] = np.cos(output_sample['tilt_2'])
    except KeyError:
        pass

    return output_sample


def generate_component_spins(sample):
    """
    Add the component spins to the data frame/dictionary.

    This function uses a lalsimulation function to transform the spins.

    Parameters
    ==========
    sample: A dictionary with the necessary spin conversion parameters:
    'theta_jn', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', 'a_1', 'a_2', 'mass_1',
    'mass_2', 'reference_frequency', 'phase'

    Returns
    =======
    dict: The updated dictionary

    """
    output_sample = sample.copy()
    spin_conversion_parameters =\
        ['theta_jn', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', 'a_1', 'a_2',
         'mass_1', 'mass_2', 'reference_frequency', 'phase']
    if all(key in output_sample.keys() for key in spin_conversion_parameters):
        (
            output_sample['iota'], output_sample['spin_1x'],
            output_sample['spin_1y'], output_sample['spin_1z'],
            output_sample['spin_2x'], output_sample['spin_2y'],
            output_sample['spin_2z']
        ) = np.vectorize(bilby_to_lalsimulation_spins)(
            output_sample['theta_jn'], output_sample['phi_jl'],
            output_sample['tilt_1'], output_sample['tilt_2'],
            output_sample['phi_12'], output_sample['a_1'], output_sample['a_2'],
            output_sample['mass_1'] * solar_mass,
            output_sample['mass_2'] * solar_mass,
            output_sample['reference_frequency'], output_sample['phase']
        )

        output_sample['phi_1'] =\
            np.fmod(2 * np.pi + np.arctan2(
                output_sample['spin_1y'], output_sample['spin_1x']), 2 * np.pi)
        output_sample['phi_2'] =\
            np.fmod(2 * np.pi + np.arctan2(
                output_sample['spin_2y'], output_sample['spin_2x']), 2 * np.pi)

    elif 'chi_1' in output_sample and 'chi_2' in output_sample:
        output_sample['spin_1x'] = 0
        output_sample['spin_1y'] = 0
        output_sample['spin_1z'] = output_sample['chi_1']
        output_sample['spin_2x'] = 0
        output_sample['spin_2y'] = 0
        output_sample['spin_2z'] = output_sample['chi_2']
    else:
        logger.debug("Component spin extraction failed.")

    return output_sample


def generate_tidal_parameters(sample):
    """
    Generate all tidal parameters

    lambda_tilde, delta_lambda_tilde

    Parameters
    ==========
    sample: dict, pandas.DataFrame
        Should contain lambda_1, lambda_2

    Returns
    =======
    output_sample: dict, pandas.DataFrame
        Updated sample
    """
    output_sample = sample.copy()

    output_sample['lambda_tilde'] =\
        lambda_1_lambda_2_to_lambda_tilde(
            output_sample['lambda_1'], output_sample['lambda_2'],
            output_sample['mass_1'], output_sample['mass_2'])
    output_sample['delta_lambda_tilde'] = \
        lambda_1_lambda_2_to_delta_lambda_tilde(
            output_sample['lambda_1'], output_sample['lambda_2'],
            output_sample['mass_1'], output_sample['mass_2'])

    return output_sample


def generate_source_frame_parameters(sample):
    """
    Generate source frame masses along with redshifts and comoving distance.

    Parameters
    ==========
    sample: dict, pandas.DataFrame

    Returns
    =======
    output_sample: dict, pandas.DataFrame
    """
    output_sample = sample.copy()

    output_sample['redshift'] =\
        luminosity_distance_to_redshift(output_sample['luminosity_distance'])
    output_sample['comoving_distance'] =\
        redshift_to_comoving_distance(output_sample['redshift'])

    for key in ['mass_1', 'mass_2', 'chirp_mass', 'total_mass']:
        if key in output_sample:
            output_sample['{}_source'.format(key)] =\
                output_sample[key] / (1 + output_sample['redshift'])

    return output_sample


def compute_snrs(sample, likelihood, npool=1):
    """
    Compute the optimal and matched filter snrs of all posterior samples
    and print it out.

    Parameters
    ==========
    sample: dict or array_like

    likelihood: bilby.gw.likelihood.GravitationalWaveTransient
        Likelihood function to be applied on the posterior

    """
    if likelihood is not None:
        if isinstance(sample, dict):
            likelihood.parameters.update(sample)
            signal_polarizations = likelihood.waveform_generator.frequency_domain_strain(likelihood.parameters.copy())
            for ifo in likelihood.interferometers:
                per_detector_snr = likelihood.calculate_snrs(signal_polarizations, ifo)
                sample['{}_matched_filter_snr'.format(ifo.name)] =\
                    per_detector_snr.complex_matched_filter_snr
                sample['{}_optimal_snr'.format(ifo.name)] = \
                    per_detector_snr.optimal_snr_squared.real ** 0.5
        else:
            from tqdm.auto import tqdm
            logger.info('Computing SNRs for every sample.')

            fill_args = [(ii, row) for ii, row in sample.iterrows()]
            if npool > 1:
                from ..core.sampler.base_sampler import _initialize_global_variables
                pool = multiprocessing.Pool(
                    processes=npool,
                    initializer=_initialize_global_variables,
                    initargs=(likelihood, None, None, False),
                )
                logger.info(
                    "Using a pool with size {} for nsamples={}".format(npool, len(sample))
                )
                new_samples = pool.map(_compute_snrs, tqdm(fill_args, file=sys.stdout))
                pool.close()
                pool.join()
            else:
                from ..core.sampler.base_sampler import _sampling_convenience_dump
                _sampling_convenience_dump.likelihood = likelihood
                new_samples = [_compute_snrs(xx) for xx in tqdm(fill_args, file=sys.stdout)]

            for ii, ifo in enumerate(likelihood.interferometers):
                snr_updates = dict()
                for key in new_samples[0][ii].snrs_as_sample.keys():
                    snr_updates[f"{ifo.name}_{key}"] = list()
                for new_sample in new_samples:
                    snr_update = new_sample[ii].snrs_as_sample
                    for key, val in snr_update.items():
                        snr_updates[f"{ifo.name}_{key}"].append(val)
                for k, v in snr_updates.items():
                    sample[k] = v
    else:
        logger.debug('Not computing SNRs.')


def _compute_snrs(args):
    """A wrapper of computing the SNRs to enable multiprocessing"""
    from ..core.sampler.base_sampler import _sampling_convenience_dump
    likelihood = _sampling_convenience_dump.likelihood
    ii, sample = args
    sample = dict(sample).copy()
    likelihood.parameters.update(sample)
    signal_polarizations = likelihood.waveform_generator.frequency_domain_strain(
        likelihood.parameters.copy()
    )
    snrs = list()
    for ifo in likelihood.interferometers:
        snrs.append(likelihood.calculate_snrs(signal_polarizations, ifo, return_array=False))
    return snrs


def compute_per_detector_log_likelihoods(samples, likelihood, npool=1, block=10):
    """
    Calculate the log likelihoods in each detector.

    Parameters
    ==========
    samples: DataFrame
        Posterior from run with a marginalised likelihood.
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient
        Likelihood used during sampling.
    npool: int, (default=1)
        If given, perform generation (where possible) using a multiprocessing pool
    block: int, (default=10)
        Size of the blocks to use in multiprocessing

    Returns
    =======
    sample: DataFrame
        Returns the posterior with new samples.
    """
    if likelihood is not None:
        if not callable(likelihood.compute_per_detector_log_likelihood):
            logger.debug('Not computing per-detector log likelihoods.')
            return samples

        if isinstance(samples, dict):
            likelihood.parameters.update(samples)
            samples = likelihood.compute_per_detector_log_likelihood()
            return samples

        elif not isinstance(samples, DataFrame):
            raise ValueError("Unable to handle input samples of type {}".format(type(samples)))
        from tqdm.auto import tqdm

        logger.info('Computing per-detector log likelihoods.')

        # Initialize cache dict
        cached_samples_dict = dict()

        # Store samples to convert for checking
        cached_samples_dict["_samples"] = samples

        # Set up the multiprocessing
        if npool > 1:
            from ..core.sampler.base_sampler import _initialize_global_variables
            pool = multiprocessing.Pool(
                processes=npool,
                initializer=_initialize_global_variables,
                initargs=(likelihood, None, None, False),
            )
            logger.info(
                "Using a pool with size {} for nsamples={}"
                .format(npool, len(samples))
            )
        else:
            from ..core.sampler.base_sampler import _sampling_convenience_dump
            _sampling_convenience_dump.likelihood = likelihood
            pool = None

        fill_args = [(ii, row) for ii, row in samples.iterrows()]
        ii = 0
        pbar = tqdm(total=len(samples), file=sys.stdout)
        while ii < len(samples):
            if ii in cached_samples_dict:
                ii += block
                pbar.update(block)
                continue

            if pool is not None:
                subset_samples = pool.map(_compute_per_detector_log_likelihoods,
                                          fill_args[ii: ii + block])
            else:
                subset_samples = [list(_compute_per_detector_log_likelihoods(xx))
                                  for xx in fill_args[ii: ii + block]]

            cached_samples_dict[ii] = subset_samples

            ii += block
            pbar.update(len(subset_samples))
        pbar.close()

        if pool is not None:
            pool.close()
            pool.join()

        new_samples = np.concatenate(
            [np.array(val) for key, val in cached_samples_dict.items() if key != "_samples"]
        )

        for ii, key in \
                enumerate([f'{ifo.name}_log_likelihood' for ifo in likelihood.interferometers]):
            samples[key] = new_samples[:, ii]

        return samples

    else:
        logger.debug('Not computing per-detector log likelihoods.')


def _compute_per_detector_log_likelihoods(args):
    """A wrapper of computing the per-detector log likelihoods to enable multiprocessing"""
    from ..core.sampler.base_sampler import _sampling_convenience_dump
    likelihood = _sampling_convenience_dump.likelihood
    ii, sample = args
    sample = dict(sample).copy()
    likelihood.parameters.update(dict(sample).copy())
    new_sample = likelihood.compute_per_detector_log_likelihood()
    return tuple((new_sample[key] for key in
                  [f'{ifo.name}_log_likelihood' for ifo in likelihood.interferometers]))


def generate_posterior_samples_from_marginalized_likelihood(
        samples, likelihood, npool=1, block=10, use_cache=True):
    """
    Reconstruct the distance posterior from a run which used a likelihood which
    explicitly marginalised over time/distance/phase.

    See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

    Parameters
    ==========
    samples: DataFrame
        Posterior from run with a marginalised likelihood.
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient
        Likelihood used during sampling.
    npool: int, (default=1)
        If given, perform generation (where possible) using a multiprocessing pool
    block: int, (default=10)
        Size of the blocks to use in multiprocessing
    use_cache: bool, (default=True)
        If true, cache the generation so that reconstuction can begin from the
        cache on restart.

    Returns
    =======
    sample: DataFrame
        Returns the posterior with new samples.
    """
    from ..core.utils.random import generate_seeds

    marginalized_parameters = getattr(likelihood, "_marginalized_parameters", list())
    if len(marginalized_parameters) == 0:
        return samples

    # pass through a dictionary
    if isinstance(samples, dict):
        return samples
    elif not isinstance(samples, DataFrame):
        raise ValueError("Unable to handle input samples of type {}".format(type(samples)))
    from tqdm.auto import tqdm

    logger.info('Reconstructing marginalised parameters.')

    try:
        cache_filename = f"{likelihood.outdir}/.{likelihood.label}_generate_posterior_cache.pickle"
    except AttributeError:
        logger.warning("Likelihood has no outdir and label attribute: caching disabled")
        use_cache = False

    if use_cache and os.path.exists(cache_filename) and not command_line_args.clean:
        try:
            with open(cache_filename, "rb") as f:
                cached_samples_dict = pickle.load(f)
        except EOFError:
            logger.warning("Cache file is empty")
            cached_samples_dict = None

        # Check the samples are identical between the cache and current
        if (cached_samples_dict is not None) and (cached_samples_dict["_samples"].equals(samples)):
            # Calculate reconstruction percentage and print a log message
            nsamples_converted = np.sum(
                [len(val) for key, val in cached_samples_dict.items() if key != "_samples"]
            )
            perc = 100 * nsamples_converted / len(cached_samples_dict["_samples"])
            logger.info(f'Using cached reconstruction with {perc:0.1f}% converted.')
        else:
            logger.info("Cached samples dict out of date, ignoring")
            cached_samples_dict = dict(_samples=samples)

    else:
        # Initialize cache dict
        cached_samples_dict = dict()

        # Store samples to convert for checking
        cached_samples_dict["_samples"] = samples

    # Set up the multiprocessing
    if npool > 1:
        from ..core.sampler.base_sampler import _initialize_global_variables
        pool = multiprocessing.Pool(
            processes=npool,
            initializer=_initialize_global_variables,
            initargs=(likelihood, None, None, False),
        )
        logger.info(
            "Using a pool with size {} for nsamples={}"
            .format(npool, len(samples))
        )
    else:
        from ..core.sampler.base_sampler import _sampling_convenience_dump
        _sampling_convenience_dump.likelihood = likelihood
        pool = None

    seeds = generate_seeds(len(samples))
    fill_args = [(ii, row, seed) for (ii, row), seed in zip(samples.iterrows(), seeds)]
    ii = 0
    pbar = tqdm(total=len(samples), file=sys.stdout)
    while ii < len(samples):
        if ii in cached_samples_dict:
            ii += block
            pbar.update(block)
            continue

        if pool is not None:
            subset_samples = pool.map(fill_sample, fill_args[ii: ii + block])
        else:
            subset_samples = [list(fill_sample(xx)) for xx in fill_args[ii: ii + block]]

        cached_samples_dict[ii] = subset_samples

        if use_cache:
            safe_file_dump(cached_samples_dict, cache_filename, "pickle")

        ii += block
        pbar.update(len(subset_samples))
    pbar.close()

    if pool is not None:
        pool.close()
        pool.join()

    new_samples = np.concatenate(
        [np.array(val) for key, val in cached_samples_dict.items() if key != "_samples"]
    )

    for ii, key in enumerate(marginalized_parameters):
        samples[key] = new_samples[:, ii]

    return samples


def generate_sky_frame_parameters(samples, likelihood):
    if isinstance(samples, dict):
        likelihood.parameters.update(samples)
        samples.update(likelihood.get_sky_frame_parameters())
        return
    elif not isinstance(samples, DataFrame):
        raise ValueError
    from tqdm.auto import tqdm

    logger.info('Generating sky frame parameters.')
    new_samples = list()
    for ii in tqdm(range(len(samples)), file=sys.stdout):
        sample = dict(samples.iloc[ii]).copy()
        likelihood.parameters.update(sample)
        new_samples.append(likelihood.get_sky_frame_parameters())
    new_samples = DataFrame(new_samples)
    for key in new_samples:
        samples[key] = new_samples[key]


def fill_sample(args):
    from ..core.sampler.base_sampler import _sampling_convenience_dump
    from ..core.utils.random import seed

    ii, sample, rseed = args
    seed(rseed)
    likelihood = _sampling_convenience_dump.likelihood
    marginalized_parameters = getattr(likelihood, "_marginalized_parameters", list())
    sample = dict(sample).copy()
    likelihood.parameters.update(dict(sample).copy())
    new_sample = likelihood.generate_posterior_sample_from_marginalized_likelihood()
    return tuple((new_sample[key] for key in marginalized_parameters))


def identity_map_conversion(parameters):
    """An identity map conversion function that makes no changes to the parameters,
    but returns the correct signature expected by other conversion functions
    (e.g. convert_to_lal_binary_black_hole_parameters)"""
    return parameters, []


def identity_map_generation(sample, likelihood=None, priors=None, npool=1):
    """An identity map generation function that handles marginalizations, SNRs, etc. correctly,
    but does not attempt e.g. conversions in mass or spins

    Parameters
    ==========
    sample: dict or pandas.DataFrame
        Samples to fill in with extra parameters, this may be either an
        injection or posterior samples.
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient, optional
        GravitationalWaveTransient used for sampling, used for waveform and
        likelihood.interferometers.
    priors: dict, optional
        Dictionary of prior objects, used to fill in non-sampled parameters.

    Returns
    =======

    """
    output_sample = sample.copy()

    output_sample = fill_from_fixed_priors(output_sample, priors)

    if likelihood is not None:
        compute_per_detector_log_likelihoods(
            samples=output_sample, likelihood=likelihood, npool=npool)

        marginalized_parameters = getattr(likelihood, "_marginalized_parameters", list())
        if len(marginalized_parameters) > 0:
            try:
                generate_posterior_samples_from_marginalized_likelihood(
                    samples=output_sample, likelihood=likelihood, npool=npool)
            except MarginalizedLikelihoodReconstructionError as e:
                logger.warning(
                    "Marginalised parameter reconstruction failed with message "
                    "{}. Some parameters may not have the intended "
                    "interpretation.".format(e)
                )

        if ("ra" in output_sample.keys() and "dec" in output_sample.keys() and "psi" in output_sample.keys()):
            compute_snrs(output_sample, likelihood, npool=npool)
        else:
            logger.info(
                "Skipping SNR computation since samples have insufficient sky location information"
            )

    return output_sample
