import json
import os
from functools import lru_cache

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import i0e
from bilby_cython.geometry import (
    zenith_azimuth_to_theta_phi as _zenith_azimuth_to_theta_phi,
)
from bilby_cython.time import greenwich_mean_sidereal_time

from ..core.utils import (logger, run_commandline,
                          check_directory_exists_and_if_not_mkdir,
                          SamplesSummary, theta_phi_to_ra_dec)
from ..core.utils.constants import solar_mass


def asd_from_freq_series(freq_data, df):
    """
    Calculate the ASD from the frequency domain output of gaussian_noise()

    Parameters
    ==========
    freq_data: array_like
        Array of complex frequency domain data
    df: float
        Spacing of freq_data, 1/(segment length) used to generate the gaussian noise

    Returns
    =======
    array_like: array of real-valued normalized frequency domain ASD data

    """
    return np.absolute(freq_data) * 2 * df**0.5


def psd_from_freq_series(freq_data, df):
    """
    Calculate the PSD from the frequency domain output of gaussian_noise()
    Calls asd_from_freq_series() and squares the output

    Parameters
    ==========
    freq_data: array_like
        Array of complex frequency domain data
    df: float
        Spacing of freq_data, 1/(segment length) used to generate the gaussian noise

    Returns
    =======
    array_like: Real-valued normalized frequency domain PSD data

    """
    return np.power(asd_from_freq_series(freq_data, df), 2)


def get_vertex_position_geocentric(latitude, longitude, elevation):
    """
    Calculate the position of the IFO vertex in geocentric coordinates in meters.

    Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
    See Section 2.1 of LIGO-T980044-10 for the correct expression

    Parameters
    ==========
    latitude: float
        Latitude in radians
    longitude:
        Longitude in radians
    elevation:
        Elevation in meters

    Returns
    =======
    array_like: A 3D representation of the geocentric vertex position

    """
    semi_major_axis = 6378137  # for ellipsoid model of Earth, in m
    semi_minor_axis = 6356752.314  # in m
    radius = semi_major_axis**2 * (semi_major_axis**2 * np.cos(latitude)**2 +
                                   semi_minor_axis**2 * np.sin(latitude)**2)**(-0.5)
    x_comp = (radius + elevation) * np.cos(latitude) * np.cos(longitude)
    y_comp = (radius + elevation) * np.cos(latitude) * np.sin(longitude)
    z_comp = ((semi_minor_axis / semi_major_axis)**2 * radius + elevation) * np.sin(latitude)
    return np.array([x_comp, y_comp, z_comp])


def inner_product(aa, bb, frequency, PSD):
    """
    Calculate the inner product defined in the matched filter statistic

    Parameters
    ==========
    aa, bb: array_like
        Single-sided Fourier transform, created, e.g., by the nfft function above
    frequency: array_like
        An array of frequencies associated with aa, bb, also returned by nfft
    PSD: bilby.gw.detector.PowerSpectralDensity

    Returns
    =======
    The matched filter inner product for aa and bb

    """
    psd_interp = PSD.power_spectral_density_interpolated(frequency)

    # calculate the inner product
    integrand = np.conj(aa) * bb / psd_interp

    df = frequency[1] - frequency[0]
    integral = np.sum(integrand) * df
    return 4. * np.real(integral)


def noise_weighted_inner_product(aa, bb, power_spectral_density, duration):
    """
    Calculate the noise weighted inner product between two arrays.

    Parameters
    ==========
    aa: array_like
        Array to be complex conjugated
    bb: array_like
        Array not to be complex conjugated
    power_spectral_density: array_like
        Power spectral density of the noise
    duration: float
        duration of the data

    Returns
    =======
    Noise-weighted inner product.
    """

    integrand = np.conj(aa) * bb / power_spectral_density
    return 4 / duration * np.sum(integrand)


def matched_filter_snr(signal, frequency_domain_strain, power_spectral_density, duration):
    """
    Calculate the _complex_ matched filter snr of a signal.
    This is <signal|frequency_domain_strain> / optimal_snr

    Parameters
    ==========
    signal: array_like
        Array containing the signal
    frequency_domain_strain: array_like

    power_spectral_density: array_like

    duration: float
        Time duration of the signal

    Returns
    =======
    float: The matched filter signal to noise ratio squared

    """
    rho_mf = noise_weighted_inner_product(
        aa=signal, bb=frequency_domain_strain,
        power_spectral_density=power_spectral_density, duration=duration)
    rho_mf /= optimal_snr_squared(
        signal=signal, power_spectral_density=power_spectral_density,
        duration=duration)**0.5
    return rho_mf


def optimal_snr_squared(signal, power_spectral_density, duration):
    """
    Compute the square of the optimal matched filter SNR for the provided
    signal.


    Parameters
    ==========
    signal: array_like
        Array containing the signal
    power_spectral_density: array_like

    duration: float
        Time duration of the signal

    Returns
    =======
    float: The matched filter signal to noise ratio squared

    """
    return noise_weighted_inner_product(signal, signal, power_spectral_density, duration)


def overlap(signal_a, signal_b, power_spectral_density=None, delta_frequency=None,
            lower_cut_off=None, upper_cut_off=None, norm_a=None, norm_b=None):
    r"""
    Compute the overlap between two signals

    .. math::

        {\cal O} = \frac{4 \Delta f}{N_{a} N_{b}} \sum_{i} \frac{h^{*}_{a,i} h_{b,i}}{S_{i}}

    Parameters
    ----------
    signal_a: array-like
    signal_b: array-like
    power_spectral_density : array-like
    delta_frequency: float
        Frequency spacing of the signals
    lower_cut_off: float
        Minimum frequency for the integral
    upper_cut_off: float
        Maximum frequency for the integral
    norm_a: float
        Normalizing factor for signal_a
    norm_b: float
        Normalizing factor for signal_b

    Returns
    -------
    float
        The overlap
    """
    low_index = int(lower_cut_off / delta_frequency)
    up_index = int(upper_cut_off / delta_frequency)
    integrand = np.conj(signal_a) * signal_b
    integrand = integrand[low_index:up_index] / power_spectral_density[low_index:up_index]
    integral = (4 * delta_frequency * integrand) / norm_a / norm_b
    return sum(integral).real


def zenith_azimuth_to_theta_phi(zenith, azimuth, ifos):
    """
    Convert from the 'detector frame' to the Earth frame.

    Parameters
    ==========
    kappa: float
        The zenith angle in the detector frame
    eta: float
        The azimuthal angle in the detector frame
    ifos: list
        List of Interferometer objects defining the detector frame

    Returns
    =======
    theta, phi: float
        The zenith and azimuthal angles in the earth frame.
    """
    delta_x = ifos[0].geometry.vertex - ifos[1].geometry.vertex
    return _zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x)


def zenith_azimuth_to_ra_dec(zenith, azimuth, geocent_time, ifos):
    """
    Convert from the 'detector frame' to the Earth frame.

    Parameters
    ==========
    kappa: float
        The zenith angle in the detector frame
    eta: float
        The azimuthal angle in the detector frame
    geocent_time: float
        GPS time at geocenter
    ifos: list
        List of Interferometer objects defining the detector frame

    Returns
    =======
    ra, dec: float
        The zenith and azimuthal angles in the sky frame.
    """
    theta, phi = zenith_azimuth_to_theta_phi(zenith, azimuth, ifos)
    gmst = greenwich_mean_sidereal_time(geocent_time)
    ra, dec = theta_phi_to_ra_dec(theta, phi, gmst)
    ra = ra % (2 * np.pi)
    return ra, dec


def get_event_time(event):
    """
    Get the merger time for known GW events using the gwosc package

    Parameters
    ----------
    event: str
        Event name, e.g. GW150914

    Returns
    -------
    event_time: float
        Merger time

    Raises
    ------
    ImportError
        If the gwosc package is not installed
    ValueError
        If the event is not in the gwosc dataset
    """
    try:
        from gwosc import datasets
    except ImportError:
        raise ImportError("You do not have the gwosc package installed")

    return datasets.event_gps(event)


def get_open_strain_data(
        name, start_time, end_time, outdir, cache=False, buffer_time=0, **kwargs):
    """ A function which accesses the open strain data

    This uses `gwpy` to download the open data and then saves a cached copy for
    later use

    Parameters
    ==========
    name: str
        The name of the detector to get data for
    start_time, end_time: float
        The GPS time of the start and end of the data
    outdir: str
        The output directory to place data in
    cache: bool
        If true, cache the data
    buffer_time: float
        Time to add to the beginning and end of the segment.
    **kwargs:
        Passed to `gwpy.timeseries.TimeSeries.fetch_open_data`

    Returns
    =======
    strain: gwpy.timeseries.TimeSeries
        The object containing the strain data. If the connection to the open-data server
        fails, this function returns `None`.

    """
    from gwpy.timeseries import TimeSeries
    filename = '{}/{}_{}_{}.txt'.format(outdir, name, start_time, end_time)

    if buffer_time < 0:
        raise ValueError("buffer_time < 0")
    start_time = start_time - buffer_time
    end_time = end_time + buffer_time

    if os.path.isfile(filename) and cache:
        logger.info('Using cached data from {}'.format(filename))
        strain = TimeSeries.read(filename)
    else:
        logger.info('Fetching open data from {} to {} with buffer time {}'
                    .format(start_time, end_time, buffer_time))
        try:
            strain = TimeSeries.fetch_open_data(name, start_time, end_time, **kwargs)
            logger.info('Saving cache of data to {}'.format(filename))
            strain.write(filename)
        except Exception as e:
            logger.info("Unable to fetch open data, see debug for detailed info")
            logger.info("Call to gwpy.timeseries.TimeSeries.fetch_open_data returned {}"
                        .format(e))
            strain = None

    return strain


def read_frame_file(file_name, start_time, end_time, resample=None, channel=None, buffer_time=0, **kwargs):
    """ A function which accesses the open strain data

    This uses `gwpy` to download the open data and then saves a cached copy for
    later use

    Parameters
    ==========
    file_name: str
        The name of the frame to read
    start_time, end_time: float
        The GPS time of the start and end of the data
    buffer_time: float
        Read in data with `t1-buffer_time` and `end_time+buffer_time`
    channel: str
        The name of the channel being searched for, some standard channel names are attempted
        if channel is not specified or if specified channel is not found.
    resample: float
        The sampling frequency to use for the TimeSeries in Hz
    **kwargs:
        Passed to `gwpy.timeseries.TimeSeries.fetch_open_data`

    Returns
    =======
    strain: gwpy.timeseries.TimeSeries

    """
    from gwpy.timeseries import TimeSeries
    loaded = False
    strain = None

    if channel is not None:
        try:
            strain = TimeSeries.read(source=file_name, channel=channel, start=start_time, end=end_time, **kwargs)
            loaded = True
            logger.info('Successfully loaded {}.'.format(channel))
        except (RuntimeError, ValueError):
            logger.warning('Channel {} not found. Trying preset channel names'.format(channel))

    if loaded is False:
        ligo_channel_types = ['GDS-CALIB_STRAIN', 'DCS-CALIB_STRAIN_C01', 'DCS-CALIB_STRAIN_C02',
                              'DCH-CLEAN_STRAIN_C02', 'GWOSC-16KHZ_R1_STRAIN',
                              'GWOSC-4KHZ_R1_STRAIN']
        virgo_channel_types = ['Hrec_hoft_V1O2Repro2A_16384Hz', 'FAKE_h_16384Hz_4R',
                               'GWOSC-16KHZ_R1_STRAIN', 'GWOSC-4KHZ_R1_STRAIN']
        channel_types = dict(H1=ligo_channel_types, L1=ligo_channel_types, V1=virgo_channel_types)
        for detector in channel_types.keys():
            for channel_type in channel_types[detector]:
                if loaded:
                    break
                channel = '{}:{}'.format(detector, channel_type)
                try:
                    strain = TimeSeries.read(source=file_name, channel=channel, start=start_time, end=end_time,
                                             **kwargs)
                    loaded = True
                    logger.info('Successfully read strain data for channel {}.'.format(channel))
                except (RuntimeError, ValueError):
                    pass

    if loaded:
        if resample and (strain.sample_rate.value != resample):
            strain = strain.resample(resample)
        return strain
    else:
        logger.warning('No data loaded.')
        return None


def get_gracedb(gracedb, outdir, duration, calibration, detectors, query_types=None, server=None):
    """
    Download information about a trigger from GraceDb and create cache files
    for finding gravitational-wave strain data.

    Parameters
    ----------
    gracedb: str
        The GraceDb ID of the trigger.
    outdir: str
        Directory to write output.
    duration: int
        Duration of data to find about the trigger time (units: :code:`s`).
    calibration: int
        Calibration label of the data, should be one of :code:`1, 2`.
    detectors: list
        The detectors to look for data for.
    query_types: str
        The LDR query type
    server: str
        The LIGO datafind server to look for data on.

    Returns
    -------
    candidate: dict
        Information downloaded from GraceDb about the trigger.
    cache_files: list
        List of cache filenames, one per interferometer.
    """
    candidate = gracedb_to_json(gracedb, outdir=outdir)
    trigger_time = candidate['gpstime']
    gps_start_time = trigger_time - duration
    cache_files = []
    if query_types is None:
        query_types = [None] * len(detectors)
    for i, det in enumerate(detectors):
        output_cache_file = gw_data_find(
            det, gps_start_time, duration, calibration,
            outdir=outdir, query_type=query_types[i], server=server)
        cache_files.append(output_cache_file)
    return candidate, cache_files


def gracedb_to_json(gracedb, cred=None, service_url='https://gracedb.ligo.org/api/', outdir=None):
    """ Script to download a GraceDB candidate

    Parameters
    ==========
    gracedb: str
        The UID of the GraceDB candidate
    cred:
        Credentials for authentications, see ligo.gracedb.rest.GraceDb
    service_url:
        The url of the GraceDB candidate
        GraceDB 'https://gracedb.ligo.org/api/' (default)
        GraceDB-playground 'https://gracedb-playground.ligo.org/api/'
    outdir: str, optional
        If given, a string identfying the location in which to store the json
    """
    logger.info(
        'Starting routine to download GraceDb candidate {}'.format(gracedb))
    from ligo.gracedb.rest import GraceDb

    logger.info('Initialise client and attempt to download')
    logger.info('Fetching from {}'.format(service_url))
    try:
        client = GraceDb(cred=cred, service_url=service_url)
    except IOError:
        raise ValueError(
            'Failed to authenticate with gracedb: check your X509 '
            'certificate is accessible and valid')
    try:
        candidate = client.event(gracedb)
        logger.info('Successfully downloaded candidate')
    except Exception as e:
        raise ValueError("Unable to obtain GraceDB candidate, exception: {}".format(e))

    json_output = candidate.json()

    if outdir is not None:
        check_directory_exists_and_if_not_mkdir(outdir)
        outfilepath = os.path.join(outdir, '{}.json'.format(gracedb))
        logger.info('Writing candidate to {}'.format(outfilepath))
        with open(outfilepath, 'w') as outfile:
            json.dump(json_output, outfile, indent=2)

    return json_output


def gw_data_find(observatory, gps_start_time, duration, calibration,
                 outdir='.', query_type=None, server=None):
    """ Builds a gw_data_find call and process output

    Parameters
    ==========
    observatory: str, {H1, L1, V1}
        Observatory description
    gps_start_time: float
        The start time in gps to look for data
    duration: int
        The duration (integer) in s
    calibration: int {1, 2}
        Use C01 or C02 calibration
    outdir: string
        A path to the directory where output is stored
    query_type: string
        The LDRDataFind query type

    Returns
    =======
    output_cache_file: str
        Path to the output cache file

    """
    logger.info('Building gw_data_find command line')

    observatory_lookup = dict(H1='H', L1='L', V1='V')
    observatory_code = observatory_lookup[observatory]

    if query_type is None:
        logger.warning('No query type provided. This may prevent data from being read.')
        if observatory_code == 'V':
            query_type = 'V1Online'
        else:
            query_type = '{}_HOFT_C0{}'.format(observatory, calibration)

    logger.info('Using LDRDataFind query type {}'.format(query_type))

    cache_file = '{}-{}_CACHE-{}-{}.lcf'.format(
        observatory, query_type, gps_start_time, duration)
    output_cache_file = os.path.join(outdir, cache_file)

    gps_end_time = gps_start_time + duration
    if server is None:
        server = os.environ.get("LIGO_DATAFIND_SERVER")
        if server is None:
            logger.warning('No data_find server found, defaulting to CIT server. '
                           'To run on other clusters, pass the output of '
                           '`echo $LIGO_DATAFIND_SERVER`')
            server = 'ldr.ldas.cit:80'

    cl_list = ['gw_data_find']
    cl_list.append('--observatory {}'.format(observatory_code))
    cl_list.append('--gps-start-time {}'.format(int(np.floor(gps_start_time))))
    cl_list.append('--gps-end-time {}'.format(int(np.ceil(gps_end_time))))
    cl_list.append('--type {}'.format(query_type))
    cl_list.append('--output {}'.format(output_cache_file))
    cl_list.append('--server {}'.format(server))
    cl_list.append('--url-type file')
    cl_list.append('--lal-cache')
    cl = ' '.join(cl_list)
    run_commandline(cl)
    return output_cache_file


def convert_args_list_to_float(*args_list):
    """ Converts inputs to floats, returns a list in the same order as the input"""
    try:
        args_list = [float(arg) for arg in args_list]
    except ValueError:
        raise ValueError("Unable to convert inputs to floats")
    return args_list


def lalsim_SimInspiralTransformPrecessingNewInitialConditions(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase):
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions

    args_list = convert_args_list_to_float(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase)

    return SimInspiralTransformPrecessingNewInitialConditions(*args_list)


@lru_cache(maxsize=10)
def lalsim_GetApproximantFromString(waveform_approximant):
    from lalsimulation import GetApproximantFromString
    if isinstance(waveform_approximant, str):
        return GetApproximantFromString(waveform_approximant)
    else:
        raise ValueError("waveform_approximant must be of type str")


def lalsim_SimInspiralFD(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant):
    """
    Safely call lalsimulation.SimInspiralFD

    Parameters
    ==========
    phase: float, int
    mass_1: float, int
    mass_2: float, int
    spin_1x: float, int
    spin_1y: float, int
    spin_1z: float, int
    spin_2x: float, int
    spin_2y: float, int
    spin_2z: float, int
    reference_frequency: float, int
    luminosity_distance: float, int
    iota: float, int
    longitude_ascending_nodes: float, int
    eccentricity: float, int
    mean_per_ano: float, int
    delta_frequency: float, int
    minimum_frequency: float, int
    maximum_frequency: float, int
    waveform_dictionary: None, lal.Dict
    approximant: int, str
    """
    from lalsimulation import SimInspiralFD

    args = convert_args_list_to_float(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, iota, phase, longitude_ascending_nodes,
        eccentricity, mean_per_ano, delta_frequency, minimum_frequency,
        maximum_frequency, reference_frequency)

    approximant = _get_lalsim_approximant(approximant)

    return SimInspiralFD(*args, waveform_dictionary, approximant)


def lalsim_SimInspiralChooseFDWaveform(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant):
    """
    Safely call lalsimulation.SimInspiralChooseFDWaveform

    Parameters
    ==========
    phase: float, int
    mass_1: float, int
    mass_2: float, int
    spin_1x: float, int
    spin_1y: float, int
    spin_1z: float, int
    spin_2x: float, int
    spin_2y: float, int
    spin_2z: float, int
    reference_frequency: float, int
    luminosity_distance: float, int
    iota: float, int
    longitude_ascending_nodes: float, int
    eccentricity: float, int
    mean_per_ano: float, int
    delta_frequency: float, int
    minimum_frequency: float, int
    maximum_frequency: float, int
    waveform_dictionary: None, lal.Dict
    approximant: int, str
    """
    from lalsimulation import SimInspiralChooseFDWaveform

    args = convert_args_list_to_float(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, iota, phase, longitude_ascending_nodes,
        eccentricity, mean_per_ano, delta_frequency, minimum_frequency,
        maximum_frequency, reference_frequency)

    approximant = _get_lalsim_approximant(approximant)

    return SimInspiralChooseFDWaveform(*args, waveform_dictionary, approximant)


@lru_cache(maxsize=10)
def _get_lalsim_approximant(approximant):
    if isinstance(approximant, int):
        pass
    elif isinstance(approximant, str):
        approximant = lalsim_GetApproximantFromString(approximant)
    else:
        raise ValueError("approximant not an int")
    return approximant


def lalsim_SimInspiralChooseFDWaveformSequence(
        phase, mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, reference_frequency, luminosity_distance, iota,
        waveform_dictionary, approximant, frequency_array):
    """
    Safely call lalsimulation.SimInspiralChooseFDWaveformSequence

    Parameters
    ==========
    phase: float, int
    mass_1: float, int
    mass_2: float, int
    spin_1x: float, int
    spin_1y: float, int
    spin_1z: float, int
    spin_2x: float, int
    spin_2y: float, int
    spin_2z: float, int
    reference_frequency: float, int
    luminosity_distance: float, int
    iota: float, int
    waveform_dictionary: None, lal.Dict
    approximant: int, str
    frequency_array: np.ndarray, lal.REAL8Vector
    """
    from lal import REAL8Vector, CreateREAL8Vector
    from lalsimulation import SimInspiralChooseFDWaveformSequence

    [mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
     luminosity_distance, iota, phase, reference_frequency] = convert_args_list_to_float(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, iota, phase, reference_frequency)

    if isinstance(approximant, int):
        pass
    elif isinstance(approximant, str):
        approximant = lalsim_GetApproximantFromString(approximant)
    else:
        raise ValueError("approximant not an int")

    if not isinstance(frequency_array, REAL8Vector):
        old_frequency_array = frequency_array.copy()
        frequency_array = CreateREAL8Vector(len(old_frequency_array))
        frequency_array.data = old_frequency_array

    return SimInspiralChooseFDWaveformSequence(
        phase, mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, reference_frequency, luminosity_distance, iota,
        waveform_dictionary, approximant, frequency_array)


def lalsim_SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1):
    from lalsimulation import SimInspiralWaveformParamsInsertTidalLambda1
    try:
        lambda_1 = float(lambda_1)
    except ValueError:
        raise ValueError("Unable to convert lambda_1 to float")

    return SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1)


def lalsim_SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2):
    from lalsimulation import SimInspiralWaveformParamsInsertTidalLambda2
    try:
        lambda_2 = float(lambda_2)
    except ValueError:
        raise ValueError("Unable to convert lambda_2 to float")

    return SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2)


def lalsim_SimNeutronStarEOS4ParamSDGammaCheck(g0, g1, g2, g3):
    from lalsimulation import SimNeutronStarEOS4ParamSDGammaCheck
    try:
        g0 = float(g0)
        g1 = float(g1)
        g2 = float(g2)
        g3 = float(g3)
    except ValueError:
        raise ValueError("Unable to convert EOS spectral parameters to floats")
    except TypeError:
        raise TypeError("Unable to convert EOS spectral parameters to floats")

    return SimNeutronStarEOS4ParamSDGammaCheck(g0, g1, g2, g3)


def lalsim_SimNeutronStarEOS4ParameterSpectralDecomposition(g0, g1, g2, g3):
    from lalsimulation import SimNeutronStarEOS4ParameterSpectralDecomposition
    try:
        g0 = float(g0)
        g1 = float(g1)
        g2 = float(g2)
        g3 = float(g3)
    except ValueError:
        raise ValueError("Unable to convert EOS spectral parameters to floats")
    except TypeError:
        raise TypeError("Unable to convert EOS spectral parameters to floats")

    return SimNeutronStarEOS4ParameterSpectralDecomposition(g0, g1, g2, g3)


def lalsim_SimNeutronStarEOS4ParamSDViableFamilyCheck(g0, g1, g2, g3):
    from lalsimulation import SimNeutronStarEOS4ParamSDViableFamilyCheck
    try:
        g0 = float(g0)
        g1 = float(g1)
        g2 = float(g2)
        g3 = float(g3)
    except ValueError:
        raise ValueError("Unable to convert EOS spectral parameters to floats")
    except TypeError:
        raise TypeError("Unable to convert EOS spectral parameters to floats")

    return SimNeutronStarEOS4ParamSDViableFamilyCheck(g0, g1, g2, g3)


def lalsim_SimNeutronStarEOS3PieceDynamicPolytrope(g0, log10p1_si, g1, log10p2_si, g2):
    from lalsimulation import SimNeutronStarEOS3PieceDynamicPolytrope
    try:
        g0 = float(g0)
        g1 = float(g1)
        g2 = float(g2)
        log10p1_si = float(log10p1_si)
        log10p2_si = float(log10p2_si)
    except ValueError:
        raise ValueError("Unable to convert EOS polytrope parameters to floats")
    except TypeError:
        raise TypeError("Unable to convert EOS polytrope parameters to floats")

    return SimNeutronStarEOS3PieceDynamicPolytrope(g0, log10p1_si, g1, log10p2_si, g2)


def lalsim_SimNeutronStarEOS3PieceCausalAnalytic(v1, log10p1_si, v2, log10p2_si, v3):
    from lalsimulation import SimNeutronStarEOS3PieceCausalAnalytic
    try:
        v1 = float(v1)
        v2 = float(v2)
        v3 = float(v3)
        log10p1_si = float(log10p1_si)
        log10p2_si = float(log10p2_si)
    except ValueError:
        raise ValueError("Unable to convert EOS causal parameters to floats")
    except TypeError:
        raise TypeError("Unable to convert EOS causal parameters to floats")

    return SimNeutronStarEOS3PieceCausalAnalytic(v1, log10p1_si, v2, log10p2_si, v3)


def lalsim_SimNeutronStarEOS3PDViableFamilyCheck(p0, log10p1_si, p1, log10p2_si, p2, causal):
    from lalsimulation import SimNeutronStarEOS3PDViableFamilyCheck
    try:
        p0 = float(p0)
        p1 = float(p1)
        p2 = float(p2)
        log10p1_si = float(log10p1_si)
        log10p2_si = float(log10p2_si)
        causal = int(causal)
    except ValueError:
        raise ValueError("Unable to convert EOS parameters to floats or int")
    except TypeError:
        raise TypeError("Unable to convert EOS parameters to floats or int")

    return SimNeutronStarEOS3PDViableFamilyCheck(p0, log10p1_si, p1, log10p2_si, p2, causal)


def lalsim_CreateSimNeutronStarFamily(eos):
    from lalsimulation import CreateSimNeutronStarFamily

    return CreateSimNeutronStarFamily(eos)


def lalsim_SimNeutronStarEOSMaxPseudoEnthalpy(eos):
    from lalsimulation import SimNeutronStarEOSMaxPseudoEnthalpy

    return SimNeutronStarEOSMaxPseudoEnthalpy(eos)


def lalsim_SimNeutronStarEOSSpeedOfSoundGeometerized(max_pseudo_enthalpy, eos):
    from lalsimulation import SimNeutronStarEOSSpeedOfSoundGeometerized
    try:
        max_pseudo_enthalpy = float(max_pseudo_enthalpy)
    except ValueError:
        raise ValueError("Unable to convert max_pseudo_enthalpy to float.")
    except TypeError:
        raise TypeError("Unable to convert max_pseudo_enthalpy to float.")

    return SimNeutronStarEOSSpeedOfSoundGeometerized(max_pseudo_enthalpy, eos)


def lalsim_SimNeutronStarFamMinimumMass(fam):
    from lalsimulation import SimNeutronStarFamMinimumMass

    return SimNeutronStarFamMinimumMass(fam)


def lalsim_SimNeutronStarMaximumMass(fam):
    from lalsimulation import SimNeutronStarMaximumMass

    return SimNeutronStarMaximumMass(fam)


def lalsim_SimNeutronStarRadius(mass_in_SI, fam):
    from lalsimulation import SimNeutronStarRadius
    try:
        mass_in_SI = float(mass_in_SI)
    except ValueError:
        raise ValueError("Unable to convert mass_in_SI to float.")
    except TypeError:
        raise TypeError("Unable to convert mass_in_SI to float.")

    return SimNeutronStarRadius(mass_in_SI, fam)


def lalsim_SimNeutronStarLoveNumberK2(mass_in_SI, fam):
    from lalsimulation import SimNeutronStarLoveNumberK2
    try:
        mass_in_SI = float(mass_in_SI)
    except ValueError:
        raise ValueError("Unable to convert mass_in_SI to float.")
    except TypeError:
        raise TypeError("Unable to convert mass_in_SI to float.")

    return SimNeutronStarLoveNumberK2(mass_in_SI, fam)


def spline_angle_xform(delta_psi):
    """
    Returns the angle in degrees corresponding to the spline
    calibration parameters delta_psi.
    Based on the same function in lalinference.bayespputils

    Parameters
    ==========
    delta_psi: calibration phase uncertainty

    Returns
    =======
    float: delta_psi in degrees

    """
    rotation = (2.0 + 1.0j * delta_psi) / (2.0 - 1.0j * delta_psi)

    return 180.0 / np.pi * np.arctan2(np.imag(rotation), np.real(rotation))


def plot_spline_pos(log_freqs, samples, nfreqs=100, level=0.9, color='k', label=None, xform=None):
    """
    Plot calibration posterior estimates for a spline model in log space.
    Adapted from the same function in lalinference.bayespputils

    Parameters
    ==========
    log_freqs: array-like
        The (log) location of spline control points.
    samples: array-like
        List of posterior draws of function at control points ``log_freqs``
    nfreqs: int
        Number of points to evaluate spline at for plotting.
    level: float
        Credible level to fill in.
    color: str
        Color to plot with.
    label: str
        Label for plot.
    xform: callable
        Function to transform the spline into plotted values.

    """
    import matplotlib.pyplot as plt
    freq_points = np.exp(log_freqs)
    freqs = np.logspace(min(log_freqs), max(log_freqs), nfreqs, base=np.exp(1))

    data = np.zeros((samples.shape[0], nfreqs))

    if xform is None:
        scaled_samples = samples
    else:
        scaled_samples = xform(samples)

    scaled_samples_summary = SamplesSummary(scaled_samples, average='mean')
    data_summary = SamplesSummary(data, average='mean')

    plt.errorbar(freq_points, scaled_samples_summary.average,
                 yerr=[-scaled_samples_summary.lower_relative_credible_interval,
                       scaled_samples_summary.upper_relative_credible_interval],
                 fmt='.', color=color, lw=4, alpha=0.5, capsize=0)

    for i, sample in enumerate(samples):
        temp = interp1d(
            log_freqs, sample, kind="cubic", fill_value=0,
            bounds_error=False)(np.log(freqs))
        if xform is None:
            data[i] = temp
        else:
            data[i] = xform(temp)

    plt.plot(freqs, np.mean(data, axis=0), color=color, label=label)
    plt.fill_between(freqs, data_summary.lower_absolute_credible_interval,
                     data_summary.upper_absolute_credible_interval,
                     color=color, alpha=.1, linewidth=0.1)
    plt.xlim(freq_points.min() - .5, freq_points.max() + 50)


def ln_i0(value):
    """
    A numerically stable method to evaluate ln(I_0) a modified Bessel function
    of order 0 used in the phase-marginalized likelihood.

    Parameters
    ==========
    value: array-like
        Value(s) at which to evaluate the function

    Returns
    =======
    array-like:
        The natural logarithm of the bessel function
    """
    return np.log(i0e(value)) + np.abs(value)


def calculate_time_to_merger(frequency, mass_1, mass_2, chi=0, safety=1.1):
    """ Leading-order calculation of the time to merger from frequency

    This uses the XLALSimInspiralTaylorF2ReducedSpinChirpTime routine to
    estimate the time to merger. Based on the implementation in
    `pycbc.pnutils._get_imr_duration`.

    Parameters
    ==========
    frequency: float
        The frequency (Hz) of the signal from which to calculate the time to merger
    mass_1, mass_2: float
        The detector frame component masses
    chi: float
        Dimensionless aligned-spin param
    safety:
        Multiplicitive safety factor

    Returns
    =======
    time_to_merger: float
        The time to merger from frequency in seconds
    """

    import lalsimulation
    return safety * lalsimulation.SimInspiralTaylorF2ReducedSpinChirpTime(
        frequency,
        mass_1 * solar_mass,
        mass_2 * solar_mass,
        chi,
        -1
    )


def gwsignal_from_lal_dict(ldict):
    """
    Convert LALDict object to a Python dictionary

    Parameters
    ==========
    ldict: LALDict
        LALDict object

    Returns
    =======
    d: dict
        Python dictionary
    """
    from lalsimulation.gwsignal.core.utils import from_lal_dict

    return from_lal_dict(ldict)
