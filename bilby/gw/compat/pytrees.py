from jax.tree_util import register_pytree_node

from bilby.gw.likelihood.base import GravitationalWaveTransient
from bilby.gw.detector.calibration import Recalibrate
from bilby.gw.detector.geometry import InterferometerGeometry
from bilby.gw.detector.interferometer import Interferometer
from bilby.gw.detector.networks import InterferometerList
from bilby.gw.detector.psd import PowerSpectralDensity
from bilby.gw.detector.strain_data import InterferometerStrainData, NotchList
from bilby.gw.waveform_generator import WaveformGenerator


def likelihood_flatten(likelihood: GravitationalWaveTransient):
    children = (
        likelihood.interferometers,
        likelihood.waveform_generator,
        getattr(likelihood, "_times", None),
        getattr(likelihood, "time_prior_array", None),
        getattr(likelihood, "_delta_tc", None),
        likelihood._noise_log_likelihood_value,
        likelihood.reference_frame,
        likelihood.reference_ifo,
    )
    aux_data = (
        likelihood.time_marginalization,
        likelihood.distance_marginalization,
        likelihood.phase_marginalization,
        likelihood.calibration_marginalization,
        likelihood.jitter_time,
        getattr(likelihood, "number_of_response_curves", 1000),
        getattr(likelihood, "starting_index", 0),
        likelihood.time_reference,
        likelihood.marginalized_parameters,
        likelihood.priors,
    )
    return children, aux_data


def likelihood_unflatten(aux_data, flat) -> GravitationalWaveTransient:
    likelihood = GravitationalWaveTransient.__new__(GravitationalWaveTransient)

    likelihood._interferometers = flat[0]
    likelihood.waveform_generator = flat[1]
    likelihood._times = flat[2]
    likelihood.time_prior_array = flat[3]
    likelihood._delta_tc = flat[4]
    likelihood._noise_log_likelihood_value = flat[5]
    likelihood._reference_frame = flat[6]
    likelihood.reference_ifo = flat[7]

    likelihood.time_marginalization = aux_data[0]
    likelihood.distance_marginalization = aux_data[1]
    likelihood.phase_marginalization = aux_data[2]
    likelihood.calibration_marginalization = aux_data[3]
    likelihood.jitter_time = aux_data[4]
    likelihood.number_of_response_curves = aux_data[5]
    likelihood.starting_index = aux_data[6]
    likelihood.time_reference = aux_data[7]
    likelihood._marginalized_parameters = aux_data[8]
    likelihood._prior = aux_data[9]
    return likelihood


def interferometer_list_flatten(ifos: InterferometerList):
    return tuple(ifos), None


def interferometer_list_unflatten(aux_data, flat) -> InterferometerList:
    ifos = InterferometerList.__new__(InterferometerList)
    list.extend(ifos, flat)
    return ifos


def interferometer_flatten(ifo: Interferometer):
    children = (
        ifo.strain_data,
        ifo.power_spectral_density,
        ifo.geometry,
        ifo.calibration_model,
        ifo.reference_time,
    )
    aux_data = (
        ifo.name,
    )
    return children, aux_data


def interferometer_unflatten(aux_data, flat) -> Interferometer:
    ifo = Interferometer.__new__(Interferometer)
    ifo.name = aux_data[0]
    ifo.strain_data = flat[0]
    ifo.power_spectral_density = flat[1]
    ifo.geometry = flat[2]
    ifo.calibration_model = flat[3]
    ifo.reference_time = flat[4]
    ifo.meta_data = dict(name=ifo.name)
    return ifo


def strain_data_flatten(strain_data: InterferometerStrainData):
    children = (
        strain_data._minimum_frequency,
        strain_data._maximum_frequency,
        strain_data._frequency_domain_strain,
        strain_data._time_domain_strain,
        strain_data._times_and_frequencies,
        strain_data.notch_list,
        strain_data.frequency_mask,
    )
    aux_data = ()
    return children, aux_data


def strain_data_unflatten(aux_data, children) -> InterferometerStrainData:
    strain_data = InterferometerStrainData.__new__(InterferometerStrainData)

    strain_data.notch_list = children[5]

    strain_data._frequency_mask = children[6]
    strain_data._frequency_mask_updated = True

    strain_data._minimum_frequency = children[0]
    strain_data._maximum_frequency = children[1]
    strain_data._frequency_domain_strain = children[2]
    strain_data._time_domain_strain = children[3]
    strain_data._times_and_frequencies = children[4]

    return strain_data


def psd_flatten(psd: PowerSpectralDensity):
    children = (
        psd._cache,
        psd._asd_array,
        psd._psd_array,
        psd.frequency_array,
    )
    aux_data = (
        psd._asd_file,
        psd._psd_file,
    )
    return children, aux_data


def psd_unflatten(aux_data, children) -> PowerSpectralDensity:
    import jax.numpy as jnp
    psd = PowerSpectralDensity.__new__(PowerSpectralDensity)

    psd._cache = children[0]
    psd._asd_array = children[1]
    psd._psd_array = children[2]
    psd.frequency_array = children[3]

    psd._asd_file = aux_data[0]
    psd._psd_file = aux_data[1]

    if isinstance(psd.frequency_array, jnp.ndarray):
        psd._interpolate_power_spectral_density()

    return psd


def geometry_flatten(geometry: InterferometerGeometry):
    children = (
        geometry._latitude,
        geometry._longitude,
        geometry._elevation,
        geometry._xarm_azimuth,
        geometry._yarm_azimuth,
        geometry._xarm_tilt,
        geometry._yarm_tilt,
        geometry._vertex,
        geometry._x,
        geometry._y,
        geometry._detector_tensor,
    )
    aux_data = (
        geometry.length,
    )
    return children, aux_data


def geometry_unflatten(aux_data, children):
    geometry = InterferometerGeometry.__new__(InterferometerGeometry)
    geometry.length = aux_data[0]
    geometry._latitude = children[0]
    geometry._longitude = children[1]
    geometry._elevation = children[2]
    geometry._xarm_azimuth = children[3]
    geometry._yarm_azimuth = children[4]
    geometry._xarm_tilt = children[5]
    geometry._yarm_tilt = children[6]
    geometry._vertex = children[7]
    geometry._x = children[8]
    geometry._y = children[9]
    geometry._detector_tensor = children[10]
    geometry._detector_tensor_updated = True
    geometry._vertex_updated = True
    geometry._x_updated = True
    geometry._y_updated = True
    return geometry


def recalibrate_flatten(recalib: Recalibrate):
    children = (recalib.params,)
    aux_data = (recalib.prefix,)
    return children, aux_data


def recalibrate_unflatten(aux_data, children) -> Recalibrate:
    recalib = Recalibrate.__new__(Recalibrate)
    recalib.prefix = aux_data[0]
    recalib.params = children[0]
    return recalib


def notch_list_flatten(notches: NotchList):
    return (), (notches,)


def notch_list_unflatten(aux_data, children) -> NotchList:
    return aux_data[0]


def wfg_flatten(wfg: WaveformGenerator):
    children = (
        wfg._times_and_frequencies,
    )
    aux_data = (
        wfg.frequency_domain_source_model,
        wfg.time_domain_source_model,
        wfg.source_parameter_keys,
        wfg.parameter_conversion,
        wfg.use_cache,
        wfg._cache,
        wfg.waveform_arguments,
    )
    return children, aux_data


def wfg_unflatten(aux_data, children) -> WaveformGenerator:
    wfg = WaveformGenerator.__new__(WaveformGenerator)
    wfg._times_and_frequencies = children[0]
    wfg.frequency_domain_source_model = aux_data[0]
    wfg.time_domain_source_model = aux_data[1]
    wfg.source_parameter_keys = aux_data[2]
    wfg.parameter_conversion = aux_data[3]
    wfg.use_cache = aux_data[4]
    wfg._cache = aux_data[5]
    wfg.waveform_arguments = aux_data[6]
    return wfg


for tpl in [
    (GravitationalWaveTransient, likelihood_flatten, likelihood_unflatten),
    (InterferometerList, interferometer_list_flatten, interferometer_list_unflatten),
    (Interferometer, interferometer_flatten, interferometer_unflatten),
    (InterferometerStrainData, strain_data_flatten, strain_data_unflatten),
    (PowerSpectralDensity, psd_flatten, psd_unflatten),
    (InterferometerGeometry, geometry_flatten, geometry_unflatten),
    (Recalibrate, recalibrate_flatten, recalibrate_unflatten),
    (NotchList, notch_list_flatten, notch_list_unflatten),
    (WaveformGenerator, wfg_flatten, wfg_unflatten),
]:
    register_pytree_node(*tpl)
