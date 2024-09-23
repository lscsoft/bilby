import numpy as np

from ..core import utils
from ..core.utils import logger
from .conversion import bilby_to_lalsimulation_spins
from .utils import (lalsim_GetApproximantFromString,
                    lalsim_SimInspiralFD,
                    lalsim_SimInspiralChooseFDWaveform,
                    lalsim_SimInspiralChooseFDWaveformSequence)

UNUSED_KWARGS_MESSAGE = """There are unused waveform kwargs. This is deprecated behavior and will
result in an error in future releases. Make sure all of the waveform kwargs are correctly
spelled.

Unused waveform_kwargs: {waveform_kwargs}
"""


def gwsignal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
                               phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):
    """
    A binary black hole waveform model using GWsignal

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Angle between the total binary angular momentum and the line of sight
    phase: float
        The phase at coalescence
    kwargs: dict
        Optional keyword arguments
        Supported arguments:

        - waveform_approximant
        - reference_frequency
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2]])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes

    Notes
    =====
    This function is a temporary wrapper to the interface that will
    likely be significantly changed or removed in a future release.
    This version is only intended to be used with `SEOBNRv5HM` and `SEOBNRv5PHM` and
    does not have full functionality for other waveform models.
    """

    from lalsimulation.gwsignal import GenerateFDWaveform
    from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator
    import astropy.units as u

    waveform_kwargs = dict(
        waveform_approximant="SEOBNRv5PHM",
        reference_frequency=50.0,
        minimum_frequency=20.0,
        maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False,
        mode_array=None,
        pn_amplitude_order=0,
    )
    waveform_kwargs.update(kwargs)

    waveform_approximant = waveform_kwargs['waveform_approximant']
    if waveform_approximant not in ["SEOBNRv5HM", "SEOBNRv5PHM"]:
        if waveform_approximant == "IMRPhenomXPHM":
            logger.warning("The new waveform interface is unreviewed for this model" +
                           "and it is only intended for testing.")
        else:
            raise ValueError("The new waveform interface is unreviewed for this model.")
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    maximum_frequency = waveform_kwargs['maximum_frequency']
    catch_waveform_errors = waveform_kwargs['catch_waveform_errors']
    mode_array = waveform_kwargs['mode_array']
    pn_amplitude_order = waveform_kwargs['pn_amplitude_order']

    if pn_amplitude_order != 0:
        # This is to mimic the behaviour in
        # https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L5542
        if pn_amplitude_order == -1:
            if waveform_approximant in ["SpinTaylorT4", "SpinTaylorT5"]:
                pn_amplitude_order = 3  # Equivalent to MAX_PRECESSING_AMP_PN_ORDER in LALSimulation
            else:
                pn_amplitude_order = 6  # Equivalent to MAX_NONPRECESSING_AMP_PN_ORDER in LALSimulation
        start_frequency = minimum_frequency * 2. / (pn_amplitude_order + 2)
    else:
        start_frequency = minimum_frequency

    # Call GWsignal generator
    wf_gen = gwsignal_get_waveform_generator(waveform_approximant)

    delta_frequency = frequency_array[1] - frequency_array[0]

    frequency_bounds = ((frequency_array >= minimum_frequency) *
                        (frequency_array <= maximum_frequency))

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1 * utils.solar_mass, mass_2=mass_2 * utils.solar_mass,
        reference_frequency=reference_frequency, phase=phase)

    eccentricity = 0.0
    longitude_ascending_nodes = 0.0
    mean_per_ano = 0.0

    # Check if conditioning is needed
    condition = 0
    if wf_gen.metadata["implemented_domain"] == 'time':
        condition = 1

    # Create dict for gwsignal generator
    gwsignal_dict = {'mass1' : mass_1 * u.solMass,
                     'mass2' : mass_2 * u.solMass,
                     'spin1x' : spin_1x * u.dimensionless_unscaled,
                     'spin1y' : spin_1y * u.dimensionless_unscaled,
                     'spin1z' : spin_1z * u.dimensionless_unscaled,
                     'spin2x' : spin_2x * u.dimensionless_unscaled,
                     'spin2y' : spin_2y * u.dimensionless_unscaled,
                     'spin2z' : spin_2z * u.dimensionless_unscaled,
                     'deltaF' : delta_frequency * u.Hz,
                     'f22_start' : start_frequency * u.Hz,
                     'f_max': maximum_frequency * u.Hz,
                     'f22_ref': reference_frequency * u.Hz,
                     'phi_ref' : phase * u.rad,
                     'distance' : luminosity_distance * u.Mpc,
                     'inclination' : iota * u.rad,
                     'eccentricity' : eccentricity * u.dimensionless_unscaled,
                     'longAscNodes' : longitude_ascending_nodes * u.rad,
                     'meanPerAno' : mean_per_ano * u.rad,
                     # 'ModeArray': mode_array,
                     'condition': condition
                     }

    if mode_array is not None:
        gwsignal_dict.update(ModeArray=mode_array)

    # Pass extra waveform arguments to gwsignal
    extra_args = waveform_kwargs.copy()

    for key in [
            "waveform_approximant",
            "reference_frequency",
            "minimum_frequency",
            "maximum_frequency",
            "catch_waveform_errors",
            "mode_array",
            "pn_spin_order",
            "pn_amplitude_order",
            "pn_tidal_order",
            "pn_phase_order",
            "numerical_relativity_file",
    ]:
        if key in extra_args.keys():
            del extra_args[key]

    gwsignal_dict.update(extra_args)

    try:
        hpc = GenerateFDWaveform(gwsignal_dict, wf_gen)
    except Exception as e:
        if not catch_waveform_errors:
            raise
        else:
            EDOM = (
                "Internal function call failed: Input domain error" in e.args[0]
            ) or "Input domain error" in e.args[
                0
            ]
            if EDOM:
                failed_parameters = dict(mass_1=mass_1, mass_2=mass_2,
                                         spin_1=(spin_1x, spin_1y, spin_1z),
                                         spin_2=(spin_2x, spin_2y, spin_2z),
                                         luminosity_distance=luminosity_distance,
                                         iota=iota, phase=phase,
                                         eccentricity=eccentricity,
                                         start_frequency=minimum_frequency)
                logger.warning("Evaluating the waveform failed with error: {}\n".format(e) +
                               "The parameters were {}\n".format(failed_parameters) +
                               "Likelihood will be set to -inf.")
                return None
            else:
                raise

    hplus = hpc.hp
    hcross = hpc.hc

    h_plus = np.zeros_like(frequency_array, dtype=complex)
    h_cross = np.zeros_like(frequency_array, dtype=complex)

    if len(hplus) > len(frequency_array):
        logger.debug("GWsignal waveform longer than bilby's `frequency_array`" +
                     "({} vs {}), ".format(len(hplus), len(frequency_array)) +
                     "probably because padded with zeros up to the next power of two length." +
                     " Truncating GWsignal array.")
        h_plus = hplus[:len(h_plus)]
        h_cross = hcross[:len(h_cross)]
    else:
        h_plus[:len(hplus)] = hplus
        h_cross[:len(hcross)] = hcross

    h_plus *= frequency_bounds
    h_cross *= frequency_bounds

    if condition:
        dt = 1 / hplus.df.value + hplus.epoch.value
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array[frequency_bounds])
        h_plus[frequency_bounds] *= time_shift
        h_cross[frequency_bounds] *= time_shift

    return dict(plus=h_plus, cross=h_cross)


def lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):
    """ A Binary Black Hole waveform model using lalsimulation

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Angle between the total binary angular momentum and the line of sight
    phase: float
        The phase at reference frequency or peak amplitude (depends on waveform)
    kwargs: dict
        Optional keyword arguments
        Supported arguments:

        - waveform_approximant
        - reference_frequency
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.
        - lal_waveform_dictionary:
          A dictionary (lal.Dict) of arguments passed to the lalsimulation
          waveform generator. The arguments are specific to the waveform used.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)


def lal_binary_neutron_star(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, lambda_1, lambda_2,
        **kwargs):
    """ A Binary Neutron Star waveform model using lalsimulation

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at reference frequency or peak amplitude (depends on waveform)
    lambda_1: float
        Dimensionless tidal deformability of mass_1
    lambda_2: float
        Dimensionless tidal deformability of mass_2
    kwargs: dict
        Optional keyword arguments
        Supported arguments:

        - waveform_approximant
        - reference_frequency
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2_NRTidal', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, lambda_1=lambda_1, lambda_2=lambda_2, **waveform_kwargs)


def lal_eccentric_binary_black_hole_no_spins(
        frequency_array, mass_1, mass_2, eccentricity, luminosity_distance,
        theta_jn, phase, **kwargs):
    """ Eccentric binary black hole waveform model using lalsimulation (EccentricFD)

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    eccentricity: float
        The orbital eccentricity of the system
    luminosity_distance: float
        The luminosity distance in megaparsec
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at reference frequency or peak amplitude (depends on waveform)
    kwargs: dict
        Optional keyword arguments
        Supported arguments:

        - waveform_approximant
        - reference_frequency
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='EccentricFD', reference_frequency=10.0,
        minimum_frequency=10.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        eccentricity=eccentricity, **waveform_kwargs)


def set_waveform_dictionary(waveform_kwargs, lambda_1=0, lambda_2=0):
    """
    Add keyword arguments to the :code:`LALDict` object.

    Parameters
    ==========
    waveform_kwargs: dict
        A dictionary of waveform kwargs. This is modified in place to remove used arguments.
    lambda_1: float
        Dimensionless tidal deformability of the primary object.
    lambda_2: float
        Dimensionless tidal deformability of the primary object.

    Returns
    =======
    waveform_dictionary: lal.LALDict
        The lal waveform dictionary. This is either taken from the waveform_kwargs or created
        internally.
    """
    import lalsimulation as lalsim
    from lal import CreateDict
    waveform_dictionary = waveform_kwargs.pop('lal_waveform_dictionary', CreateDict())
    waveform_kwargs["TidalLambda1"] = lambda_1
    waveform_kwargs["TidalLambda2"] = lambda_2
    waveform_kwargs["NumRelData"] = waveform_kwargs.pop("numerical_relativity_data", None)

    for key in [
        "pn_spin_order", "pn_tidal_order", "pn_phase_order", "pn_amplitude_order"
    ]:
        waveform_kwargs[key[:2].upper() + key[3:].title().replace('_', '')] = waveform_kwargs.pop(key)

    for key in list(waveform_kwargs.keys()).copy():
        func = getattr(lalsim, f"SimInspiralWaveformParamsInsert{key}", None)
        if func is None:
            continue
        value = waveform_kwargs.pop(key)
        if func is not None and value is not None:
            func(waveform_dictionary, value)

    mode_array = waveform_kwargs.pop("mode_array", None)
    if mode_array is not None:
        mode_array_lal = lalsim.SimInspiralCreateModeArray()
        for mode in mode_array:
            lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, mode[0], mode[1])
        lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_dictionary, mode_array_lal)
    return waveform_dictionary


def _base_lal_cbc_fd_waveform(
        frequency_array, mass_1, mass_2, luminosity_distance, theta_jn, phase,
        a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
        lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, **waveform_kwargs):
    """ Generate a cbc waveform model using lalsimulation

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total and orbital angular momenta
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at reference frequency or peak amplitude (depends on waveform)
    eccentricity: float
        Binary eccentricity
    lambda_1: float
        Tidal deformability of the more massive object
    lambda_2: float
        Tidal deformability of the less massive object
    kwargs: dict
        Optional keyword arguments

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    import lalsimulation as lalsim

    waveform_approximant = waveform_kwargs.pop('waveform_approximant')
    reference_frequency = waveform_kwargs.pop('reference_frequency')
    minimum_frequency = waveform_kwargs.pop('minimum_frequency')
    maximum_frequency = waveform_kwargs.pop('maximum_frequency')
    catch_waveform_errors = waveform_kwargs.pop('catch_waveform_errors')
    pn_amplitude_order = waveform_kwargs['pn_amplitude_order']

    waveform_dictionary = set_waveform_dictionary(waveform_kwargs, lambda_1, lambda_2)
    approximant = lalsim_GetApproximantFromString(waveform_approximant)

    if pn_amplitude_order != 0:
        start_frequency = lalsim.SimInspiralfLow2fStart(
            float(minimum_frequency), int(pn_amplitude_order), approximant
        )
    else:
        start_frequency = minimum_frequency

    delta_frequency = frequency_array[1] - frequency_array[0]

    frequency_bounds = ((frequency_array >= minimum_frequency) *
                        (frequency_array <= maximum_frequency))

    luminosity_distance = luminosity_distance * 1e6 * utils.parsec
    mass_1 = mass_1 * utils.solar_mass
    mass_2 = mass_2 * utils.solar_mass

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1, mass_2=mass_2,
        reference_frequency=reference_frequency, phase=phase)

    longitude_ascending_nodes = 0.0
    mean_per_ano = 0.0

    if lalsim.SimInspiralImplementedFDApproximants(approximant):
        wf_func = lalsim_SimInspiralChooseFDWaveform
    else:
        wf_func = lalsim_SimInspiralFD
    try:
        hplus, hcross = wf_func(
            mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
            spin_2z, luminosity_distance, iota, phase,
            longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
            start_frequency, maximum_frequency, reference_frequency,
            waveform_dictionary, approximant)
    except Exception as e:
        if not catch_waveform_errors:
            raise
        else:
            EDOM = (e.args[0] == 'Internal function call failed: Input domain error')
            if EDOM:
                failed_parameters = dict(mass_1=mass_1, mass_2=mass_2,
                                         spin_1=(spin_1x, spin_1y, spin_1z),
                                         spin_2=(spin_2x, spin_2y, spin_2z),
                                         luminosity_distance=luminosity_distance,
                                         iota=iota, phase=phase,
                                         eccentricity=eccentricity,
                                         start_frequency=start_frequency)
                logger.warning("Evaluating the waveform failed with error: {}\n".format(e) +
                               "The parameters were {}\n".format(failed_parameters) +
                               "Likelihood will be set to -inf.")
                return None
            else:
                raise

    h_plus = np.zeros_like(frequency_array, dtype=complex)
    h_cross = np.zeros_like(frequency_array, dtype=complex)

    if len(hplus.data.data) > len(frequency_array):
        logger.debug("LALsim waveform longer than bilby's `frequency_array`" +
                     "({} vs {}), ".format(len(hplus.data.data), len(frequency_array)) +
                     "probably because padded with zeros up to the next power of two length." +
                     " Truncating lalsim array.")
        h_plus = hplus.data.data[:len(h_plus)]
        h_cross = hcross.data.data[:len(h_cross)]
    else:
        h_plus[:len(hplus.data.data)] = hplus.data.data
        h_cross[:len(hcross.data.data)] = hcross.data.data

    h_plus *= frequency_bounds
    h_cross *= frequency_bounds

    if wf_func == lalsim_SimInspiralFD:
        dt = 1 / hplus.deltaF + (hplus.epoch.gpsSeconds + hplus.epoch.gpsNanoSeconds * 1e-9)
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array[frequency_bounds])
        h_plus[frequency_bounds] *= time_shift
        h_cross[frequency_bounds] *= time_shift

    if len(waveform_kwargs) > 0:
        logger.warning(UNUSED_KWARGS_MESSAGE.format(waveform_kwargs=waveform_kwargs))

    return dict(plus=h_plus, cross=h_cross)


def binary_black_hole_roq(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **waveform_arguments):
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=20.0,
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(waveform_arguments)
    return _base_roq_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
        phi_12=phi_12, lambda_1=0.0, lambda_2=0.0, **waveform_kwargs)


def binary_neutron_star_roq(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, lambda_1, lambda_2, theta_jn, phase,
        **waveform_arguments):
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomD_NRTidal', reference_frequency=20.0,
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(waveform_arguments)
    return _base_roq_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
        phi_12=phi_12, lambda_1=lambda_1, lambda_2=lambda_2, **waveform_kwargs)


def lal_binary_black_hole_relative_binning(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, fiducial, **kwargs):
    """ Source model to go with RelativeBinningGravitationalWaveTransient likelihood.

    All parameters are the same as in the usual source models, except `fiducial`

    fiducial: float
        If fiducial=1, waveform evaluated on the full frequency grid is returned.
        If fiducial=0, waveform evaluated at waveform_kwargs["frequency_bin_edges"]
        is returned.
    """

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    if fiducial == 1:
        _ = waveform_kwargs.pop("frequency_bin_edges", None)
        return _base_lal_cbc_fd_waveform(
            frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
            luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
            a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
            phi_12=phi_12, lambda_1=0.0, lambda_2=0.0, **waveform_kwargs)

    else:
        _ = waveform_kwargs.pop("minimum_frequency", None)
        _ = waveform_kwargs.pop("maximum_frequency", None)
        waveform_kwargs["frequencies"] = waveform_kwargs.pop("frequency_bin_edges")
        return _base_waveform_frequency_sequence(
            frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
            luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
            a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
            phi_12=phi_12, lambda_1=0.0, lambda_2=0.0, **waveform_kwargs)


def lal_binary_neutron_star_relative_binning(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, lambda_1, lambda_2, theta_jn, phase,
        fiducial, **kwargs):
    """ Source model to go with RelativeBinningGravitationalWaveTransient likelihood.

    All parameters are the same as in the usual source models, except `fiducial`

    fiducial: float
        If fiducial=1, waveform evaluated on the full frequency grid is returned.
        If fiducial=0, waveform evaluated at waveform_kwargs["frequency_bin_edges"]
        is returned.
    """

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2_NRTidal', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    if fiducial == 1:
        return _base_lal_cbc_fd_waveform(
            frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
            luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
            a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
            phi_jl=phi_jl, lambda_1=lambda_1, lambda_2=lambda_2, **waveform_kwargs)
    else:
        _ = waveform_kwargs.pop("minimum_frequency", None)
        _ = waveform_kwargs.pop("maximum_frequency", None)
        waveform_kwargs["frequencies"] = waveform_kwargs.pop("frequency_bin_edges")
        return _base_waveform_frequency_sequence(
            frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
            luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
            a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
            phi_12=phi_12, lambda_1=lambda_1, lambda_2=lambda_2, **waveform_kwargs)


def _base_roq_waveform(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, lambda_1, lambda_2, phi_jl, theta_jn, phase,
        **waveform_arguments):
    """ Base source model for ROQGravitationalWaveTransient, which evaluates
    waveform values at frequency nodes contained in waveform_arguments. This
    requires that waveform_arguments contain all of 'frequency_nodes',
    'linear_indices', and 'quadratic_indices', or both 'frequency_nodes_linear' and
    'frequency_nodes_quadratic'.

    Parameters
    ==========
    frequency_array: np.array
        This input is ignored for the roq source model
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float

    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float

    theta_jn: float
        Orbital inclination
    phase: float
        The phase at reference frequency or peak amplitude (depends on waveform)

    Waveform arguments
    ===================
    Non-sampled extra data used in the source model calculation
    frequency_nodes_linear: np.array
        frequency nodes for linear likelihood part
    frequency_nodes_quadratic: np.array
        frequency nodes for quadratic likelihood part
    frequency_nodes: np.array
        unique frequency nodes for linear and quadratic likelihood parts
    linear_indices: np.array
        indices to recover frequency nodes for linear part from unique frequency nodes
    quadratic_indices: np.array
        indices to recover frequency nodes for quadratic part from unique frequency nodes
    reference_frequency: float
    approximant: str

    Note: for the frequency_nodes_linear and frequency_nodes_quadratic arguments,
    if using data from https://git.ligo.org/lscsoft/ROQ_data, this should be
    loaded as `np.load(filename).T`.

    Returns
    =======
    waveform_polarizations: dict
        Dict containing plus and cross modes evaluated at the linear and
        quadratic frequency nodes.
    """
    if 'frequency_nodes' not in waveform_arguments:
        size_linear = len(waveform_arguments['frequency_nodes_linear'])
        frequency_nodes_combined = np.hstack(
            (waveform_arguments.pop('frequency_nodes_linear'),
             waveform_arguments.pop('frequency_nodes_quadratic'))
        )
        frequency_nodes_unique, original_indices = np.unique(
            frequency_nodes_combined, return_inverse=True
        )
        linear_indices = original_indices[:size_linear]
        quadratic_indices = original_indices[size_linear:]
        waveform_arguments['frequencies'] = frequency_nodes_unique
    else:
        linear_indices = waveform_arguments.pop("linear_indices")
        quadratic_indices = waveform_arguments.pop("quadratic_indices")
        for key in ["frequency_nodes_linear", "frequency_nodes_quadratic"]:
            _ = waveform_arguments.pop(key, None)
        waveform_arguments['frequencies'] = waveform_arguments.pop('frequency_nodes')
    waveform_polarizations = _base_waveform_frequency_sequence(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
        phi_12=phi_12, lambda_1=lambda_1, lambda_2=lambda_2, **waveform_arguments)

    return {
        'linear': {
            'plus': waveform_polarizations['plus'][linear_indices],
            'cross': waveform_polarizations['cross'][linear_indices]
        },
        'quadratic': {
            'plus': waveform_polarizations['plus'][quadratic_indices],
            'cross': waveform_polarizations['cross'][quadratic_indices]
        }
    }


def binary_black_hole_frequency_sequence(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):
    """ A Binary Black Hole waveform model using lalsimulation. This generates
    a waveform only on specified frequency points. This is useful for
    likelihood requiring waveform values at a subset of all the frequency
    samples. For example, this is used for MBGravitationalWaveTransient.

    Parameters
    ==========
    frequency_array: array_like
        The input is ignored.
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Angle between the total binary angular momentum and the line of sight
    phase: float
        The phase at reference frequency or peak amplitude (depends on waveform)
    kwargs: dict
        Required keyword arguments
        - frequencies:
            ndarray of frequencies at which waveforms are evaluated

        Optional keyword arguments
        - waveform_approximant
        - reference_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_waveform_frequency_sequence(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
        phi_12=phi_12, lambda_1=0.0, lambda_2=0.0, **waveform_kwargs)


def binary_neutron_star_frequency_sequence(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, lambda_1, lambda_2, theta_jn, phase,
        **kwargs):
    """ A Binary Neutron Star waveform model using lalsimulation. This generates
    a waveform only on specified frequency points. This is useful for
    likelihood requiring waveform values at a subset of all the frequency
    samples. For example, this is used for MBGravitationalWaveTransient.

    Parameters
    ==========
    frequency_array: array_like
        The input is ignored.
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    lambda_1: float
        Dimensionless tidal deformability of mass_1
    lambda_2: float
        Dimensionless tidal deformability of mass_2
    theta_jn: float
        Angle between the total binary angular momentum and the line of sight
    phase: float
        The phase at reference frequency or peak amplitude (depends on waveform)
    kwargs: dict
        Required keyword arguments
        - frequencies:
            ndarray of frequencies at which waveforms are evaluated

        Optional keyword arguments
        - waveform_approximant
        - reference_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2_NRTidal', reference_frequency=50.0,
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_waveform_frequency_sequence(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
        phi_12=phi_12, lambda_1=lambda_1, lambda_2=lambda_2, **waveform_kwargs)


def _base_waveform_frequency_sequence(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, lambda_1, lambda_2, phi_jl, theta_jn, phase,
        **waveform_kwargs):
    """ Generate a cbc waveform model on specified frequency samples

    Parameters
    ----------
    frequency_array: np.array
        This input is ignored
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at reference frequency or peak amplitude (depends on waveform)
    waveform_kwargs: dict
        Optional keyword arguments

    Returns
    -------
    waveform_polarizations: dict
        Dict containing plus and cross modes evaluated at the linear and
        quadratic frequency nodes.
    """
    frequencies = waveform_kwargs.pop('frequencies')
    reference_frequency = waveform_kwargs.pop('reference_frequency')
    approximant = waveform_kwargs.pop('waveform_approximant')
    catch_waveform_errors = waveform_kwargs.pop('catch_waveform_errors')

    waveform_dictionary = set_waveform_dictionary(waveform_kwargs, lambda_1, lambda_2)
    approximant = lalsim_GetApproximantFromString(approximant)

    luminosity_distance = luminosity_distance * 1e6 * utils.parsec
    mass_1 = mass_1 * utils.solar_mass
    mass_2 = mass_2 * utils.solar_mass

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1, mass_2=mass_2,
        reference_frequency=reference_frequency, phase=phase)

    try:
        h_plus, h_cross = lalsim_SimInspiralChooseFDWaveformSequence(
            phase, mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
            spin_2z, reference_frequency, luminosity_distance, iota,
            waveform_dictionary, approximant, frequencies)
    except Exception as e:
        if not catch_waveform_errors:
            raise
        else:
            EDOM = (e.args[0] == 'Internal function call failed: Input domain error')
            if EDOM:
                failed_parameters = dict(mass_1=mass_1, mass_2=mass_2,
                                         spin_1=(spin_1x, spin_1y, spin_1z),
                                         spin_2=(spin_2x, spin_2y, spin_2z),
                                         luminosity_distance=luminosity_distance,
                                         iota=iota, phase=phase)
                logger.warning("Evaluating the waveform failed with error: {}\n".format(e) +
                               "The parameters were {}\n".format(failed_parameters) +
                               "Likelihood will be set to -inf.")
                return None
            else:
                raise

    if len(waveform_kwargs) > 0:
        logger.warning(UNUSED_KWARGS_MESSAGE.format(waveform_kwargs=waveform_kwargs))

    return dict(plus=h_plus.data.data, cross=h_cross.data.data)


def sinegaussian(frequency_array, hrss, Q, frequency, **kwargs):
    r"""
    A frequency-domain sine-Gaussian burst source model.

    .. math::

        \tau &= \frac{Q}{\sqrt{2}\pi f_{0}} \\
        \alpha &= \frac{Q}{4\sqrt{\pi} f_{0}} \\
        h_{+} &=
            \frac{h_{\rm rss}\sqrt{\pi}\tau}{2\sqrt{\alpha (1 + e^{-Q^2})}}
            \left(
                e^{-\pi^2 \tau^2 (f + f_{0})^2}
                + e^{-\pi^2 \tau^2 (f - f_{0})^2}
            \right) \\
        h_{\times} &=
            \frac{i h_{\rm rss}\sqrt{\pi}\tau}{2\sqrt{\alpha (1 - e^{-Q^2})}}
            \left(
                e^{-\pi^2 \tau^2 (f + f_{0})^2}
                - e^{-\pi^2 \tau^2 (f - f_{0})^2}
            \right)

    Parameters
    ----------
    frequency_array: array-like
        The frequencies at which to compute the model.
    hrss: float
        The amplitude of the signal.
    Q: float
        The quality factor of the burst, determines the decay time.
    frequency: float
        The peak frequency of the burst.
    kwargs: dict
        UNUSED

    Returns
    -------
    dict:
        Dictionary containing the plus and cross components of the strain.
    """
    tau = Q / (np.sqrt(2.0) * np.pi * frequency)
    temp = Q / (4.0 * np.sqrt(np.pi) * frequency)
    fm = frequency_array - frequency
    fp = frequency_array + frequency

    h_plus = ((hrss / np.sqrt(temp * (1 + np.exp(-Q**2)))) *
              ((np.sqrt(np.pi) * tau) / 2.0) *
              (np.exp(-fm**2 * np.pi**2 * tau**2) +
              np.exp(-fp**2 * np.pi**2 * tau**2)))

    h_cross = (-1j * (hrss / np.sqrt(temp * (1 - np.exp(-Q**2)))) *
               ((np.sqrt(np.pi) * tau) / 2.0) *
               (np.exp(-fm**2 * np.pi**2 * tau**2) -
               np.exp(-fp**2 * np.pi**2 * tau**2)))

    return {'plus': h_plus, 'cross': h_cross}


def supernova(frequency_array, luminosity_distance, **kwargs):
    """
    A source model that reads a simulation from a text file.

    This was originally intended for use with supernova simulations, but can
    be applied to any source class.

    Parameters
    ----------
    frequency_array: array-like
        Unused
    file_path: str
        Path to the file containing the NR simulation. The format of this file
        should be readable by :code:`numpy.loadtxt` and have four columns
        containing the real and imaginary components of the plus and cross
        polarizations.
    luminosity_distance: float
        The distance to the source in kpc, this scales the amplitude of the
        signal. The simulation is assumed to be at 10kpc.
    kwargs:
        extra keyword arguments, this should include the :code:`file_path`

    Returns
    -------
    dict:
        A dictionary containing the plus and cross components of the signal.
    """

    file_path = kwargs["file_path"]
    data = np.genfromtxt(file_path)

    # waveform in file at 10kpc
    scaling = 1e-3 * (10.0 / luminosity_distance)

    h_plus = scaling * (data[:, 0] + 1j * data[:, 1])
    h_cross = scaling * (data[:, 2] + 1j * data[:, 3])
    return {'plus': h_plus, 'cross': h_cross}


def supernova_pca_model(
        frequency_array, pc_coeff1, pc_coeff2, pc_coeff3, pc_coeff4, pc_coeff5, luminosity_distance, **kwargs
):
    r"""
    Signal model based on a five-component principal component decomposition
    of a model.

    While this was initially intended for modelling supernova signal, it is
    applicable to any situation using such a principal component decomposition.

    .. math::

        h_{A} = \frac{10^{-22}}{d_{L}} \sum_{i=1}^{5} c_{i} h_{i}

    Parameters
    ----------
    frequency_array: UNUSED
    pc_coeff1: float
        The first principal component coefficient.
    pc_coeff2: float
        The second principal component coefficient.
    pc_coeff3: float
        The third principal component coefficient.
    pc_coeff4: float
        The fourth principal component coefficient.
    pc_coeff5: float
        The fifth principal component coefficient.
    luminosity_distance: float
        The distance to the source, the amplitude is scaled such that the
        amplitude at 10 kpc is 1e-23.
    kwargs: dict
        Dictionary containing numpy arrays with the real and imaginary
        components of the principal component decomposition.

    Returns
    -------
    dict:
        The plus and cross polarizations of the signal
    """

    principal_components = kwargs["realPCs"] + 1j * kwargs["imagPCs"]
    coefficients = [pc_coeff1, pc_coeff2, pc_coeff3, pc_coeff4, pc_coeff5]

    strain = np.sum(
        [coeff * principal_components[:, ii] for ii, coeff in enumerate(coefficients)],
        axis=0
    )

    # file at 10kpc
    scaling = 1e-23 * (10.0 / luminosity_distance)

    h_plus = scaling * strain
    h_cross = scaling * strain

    return {'plus': h_plus, 'cross': h_cross}


precession_only = {
    "tilt_1", "tilt_2", "phi_12", "phi_jl", "chi_1_in_plane", "chi_2_in_plane",
}

spin = {
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "chi_1", "chi_2",
    "chi_1_in_plane", "chi_2_in_plane",
}
mass = {
    "chirp_mass", "mass_ratio", "total_mass", "mass_1", "mass_2",
    "symmetric_mass_ratio",
}
primary_spin_and_q = {
    "a_1", "chi_1", "mass_ratio"
}
tidal = {
    "lambda_1", "lambda_2", "lambda_tilde", "delta_lambda_tilde"
}
phase = {
    "phase", "delta_phase",
}
extrinsic = {
    "azimuth", "zenith", "luminosity_distance", "psi", "theta_jn",
    "cos_theta_jn", "geocent_time", "time_jitter", "ra", "dec",
    "H1_time", "L1_time", "V1_time",
}
sky = {
    "azimuth", "zenith", "ra", "dec",
}
distance_inclination = {
    "luminosity_distance", "redshift", "theta_jn", "cos_theta_jn",
}
measured_spin = {
    "chi_1", "chi_2", "a_1", "a_2", "chi_1_in_plane"
}

PARAMETER_SETS = dict(
    spin=spin, mass=mass, phase=phase, extrinsic=extrinsic,
    tidal=tidal, primary_spin_and_q=primary_spin_and_q,
    intrinsic=spin.union(mass).union(phase).union(tidal),
    precession_only=precession_only,
    sky=sky, distance_inclination=distance_inclination,
    measured_spin=measured_spin,
)
