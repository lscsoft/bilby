""" Functions for adding calibration factors to waveform templates.
"""
import copy
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ...core.utils.log import logger
from ...core.prior.dict import PriorDict
from ..prior import CalibrationPriorDict


def read_calibration_file(filename, frequency_array, number_of_response_curves, starting_index=0, correction=None):
    r"""
    Function to read the hdf5 files from the calibration group containing the
    physical calibration response curves.

    These curves are defined to convert calibrated strain (:code:`d`) to
    theoretical waveform templates (:code:`h`), in the likelihood we apply
    the correction in the opposite direction and so must invert the curves
    stored in the file.

    Parameters
    ----------
    filename: str
        Location of the HDF5 file that contains the curves
    frequency_array: array-like
        The frequency values to calculate the calibration response curves at
    number_of_response_curves: int
        Number of random draws to use from the calibration file
    starting_index: int
        Index of the first curve to use within the array. This allows for segmenting the calibration curve array
        into smaller pieces.
    correction: str
        How the correction is defined, either to the data (default) or
        the template, see :func:`bilby.gw.prior.CalibrationPriorDict.from_envelope_file`.

    Returns
    -------
    calibration_draws: array-like
        Array which contains the calibration responses as a function of the frequency array specified.
        Shape is (number_of_response_curves x len(frequency_array))

    """
    import tables

    if correction is None:
        logger.warning(
            "Calibration envelope correction type is not specified. "
            "Assuming this correction maps calibrated data to theoretical "
            "strain. If this is correct, this should be explicitly "
            "specified via CalibrationPriorDict.from_envelope_file(..., "
            "correction='data')."
        )
        correction = "data"
    if correction.lower() not in ["data", "template"]:
        raise ValueError(
            "Calibration envelope correction should be one of 'data' or "
            f"'template', found {correction}."
        )

    logger.info(f"Reading calibration draws from {filename}")
    calibration_file = tables.open_file(filename, 'r')
    calibration_amplitude = \
        calibration_file.root.deltaR.draws_amp_rel[starting_index:number_of_response_curves + starting_index]
    calibration_phase = \
        calibration_file.root.deltaR.draws_phase[starting_index:number_of_response_curves + starting_index]

    calibration_frequencies = calibration_file.root.deltaR.freq[:]

    calibration_file.close()

    if len(calibration_amplitude.dtype) != 0:  # handling if this is a calibration group hdf5 file
        calibration_amplitude = calibration_amplitude.view(np.float64).reshape(calibration_amplitude.shape + (-1,))
        calibration_phase = calibration_phase.view(np.float64).reshape(calibration_phase.shape + (-1,))
        calibration_frequencies = calibration_frequencies.view(np.float64)

    # interpolate to the frequency array (where if outside the range of the calibration uncertainty its fixed to 1)
    calibration_draws = calibration_amplitude * np.exp(1j * calibration_phase)
    calibration_draws = interp1d(
        calibration_frequencies, calibration_draws, kind='cubic',
        bounds_error=False, fill_value=1)(frequency_array)
    if correction == "data":
        calibration_draws **= -1

    try:
        parameter_draws = pd.read_hdf(filename, key="CalParams")
    except KeyError:
        parameter_draws = None

    return calibration_draws, parameter_draws


def write_calibration_file(filename, frequency_array, calibration_draws, calibration_parameter_draws=None, correction=None):
    """
    Function to write the generated response curves to file

    Parameters
    ----------
    filename: str
        Location and filename to save the file
    frequency_array: array-like
        The frequency values where the calibration response was calculated
    calibration_draws: array-like
        Array which contains the calibration responses as a function of the frequency array specified.
        Shape is (number_of_response_curves x len(frequency_array))
    calibration_parameter_draws: data_frame
        Parameters used to generate the random draws of the calibration response curves
    correction: str
        How the correction is defined, either to the data (default) or
        the template, see :func:`bilby.gw.prior.CalibrationPriorDict.from_envelope_file`.
        If :code:`data`, the calibration draws will be inverted before being saved so that the format
        matches files produced by detector calibration pipelines.

    """
    import tables

    if correction is None:
        logger.warning(
            "Calibration envelope correction type is not specified. "
            "Assuming this correction maps calibrated data to theoretical "
            "strain. If this is correct, this should be explicitly "
            "specified via CalibrationPriorDict.from_envelope_file(..., "
            "correction='data')."
        )
        correction = "data"
    if correction.lower() not in ["data", "template"]:
        raise ValueError(
            "Calibration envelope correction should be one of 'data' or "
            f"'template', found {correction}."
        )

    if correction == "data":
        calibration_draws **= -1

    logger.info(f"Writing calibration draws to {filename}")
    calibration_file = tables.open_file(filename, 'w')
    deltaR_group = calibration_file.create_group(calibration_file.root, 'deltaR')

    # Save output
    calibration_file.create_carray(deltaR_group, 'draws_amp_rel', obj=np.abs(calibration_draws))
    calibration_file.create_carray(deltaR_group, 'draws_phase', obj=np.angle(calibration_draws))
    calibration_file.create_carray(deltaR_group, 'freq', obj=frequency_array)

    calibration_file.close()

    # Save calibration parameter draws
    if calibration_parameter_draws is not None:
        calibration_parameter_draws.to_hdf(filename, key='CalParams', data_columns=True, format='table')


class Recalibrate(object):

    name = 'none'

    def __init__(self, prefix='recalib_'):
        """
        Base calibration object. This applies no transformation

        Parameters
        ==========
        prefix: str
            Prefix on parameters relating to the calibration.
        """
        self.params = dict()
        self.prefix = prefix

    def __repr__(self):
        return self.__class__.__name__ + '(prefix=\'{}\')'.format(self.prefix)

    def get_calibration_factor(self, frequency_array, **params):
        """Apply calibration model

        This method should be overwritten by subclasses

        Parameters
        ==========
        frequency_array: array-like
            The frequency values to calculate the calibration factor for.
        params : dict
            Dictionary of sampling parameters which includes
            calibration parameters.

        Returns
        =======
        calibration_factor : array-like
            The factor to multiply the strain by.
        """
        return np.ones_like(frequency_array)

    def set_calibration_parameters(self, **params):
        self.params.update({key[len(self.prefix):]: params[key] for key in params
                            if self.prefix in key})

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class CubicSpline(Recalibrate):

    name = 'cubic_spline'

    def __init__(self, prefix, minimum_frequency, maximum_frequency, n_points):
        """
        Cubic spline recalibration

        see https://dcc.ligo.org/DocDB/0116/T1400682/001/calnote.pdf

        This assumes the spline points follow
        np.logspace(np.log(minimum_frequency), np.log(maximum_frequency), n_points)

        Parameters
        ==========
        prefix: str
            Prefix on parameters relating to the calibration.
        minimum_frequency: float
            minimum frequency of spline points
        maximum_frequency: float
            maximum frequency of spline points
        n_points: int
            number of spline points
        """
        super(CubicSpline, self).__init__(prefix=prefix)
        if n_points < 4:
            raise ValueError('Cubic spline calibration requires at least 4 spline nodes.')
        self.n_points = n_points
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self._log_spline_points = np.linspace(
            np.log10(minimum_frequency), np.log10(maximum_frequency), n_points)

    @property
    def delta_log_spline_points(self):
        if not hasattr(self, "_delta_log_spline_points"):
            self._delta_log_spline_points = self._log_spline_points[1] - self._log_spline_points[0]
        return self._delta_log_spline_points

    @property
    def nodes_to_spline_coefficients(self):
        if not hasattr(self, "_nodes_to_spline_coefficients"):
            self._setup_spline_coefficients()
        return self._nodes_to_spline_coefficients

    def _setup_spline_coefficients(self):
        """
        Precompute matrix converting values at nodes to spline coefficients.
        The algorithm for interpolation is described in
        https://dcc.ligo.org/LIGO-T2300140, and the matrix calculated here is
        to solve Eq. (9) in the note.
        """
        tmp1 = np.zeros(shape=(self.n_points, self.n_points))
        tmp1[0, 0] = -1
        tmp1[0, 1] = 2
        tmp1[0, 2] = -1
        tmp1[-1, -3] = -1
        tmp1[-1, -2] = 2
        tmp1[-1, -1] = -1
        for i in range(1, self.n_points - 1):
            tmp1[i, i - 1] = 1 / 6
            tmp1[i, i] = 2 / 3
            tmp1[i, i + 1] = 1 / 6
        tmp2 = np.zeros(shape=(self.n_points, self.n_points))
        for i in range(1, self.n_points - 1):
            tmp2[i, i - 1] = 1
            tmp2[i, i] = -2
            tmp2[i, i + 1] = 1
        self._nodes_to_spline_coefficients = np.linalg.solve(tmp1, tmp2)

    @property
    def log_spline_points(self):
        return self._log_spline_points

    def __repr__(self):
        return self.__class__.__name__ + '(prefix=\'{}\', minimum_frequency={}, maximum_frequency={}, n_points={})'\
            .format(self.prefix, self.minimum_frequency, self.maximum_frequency, self.n_points)

    def _evaluate_spline(self, kind, a, b, c, d, previous_nodes):
        """Evaluate Eq. (1) in https://dcc.ligo.org/LIGO-T2300140"""
        parameters = np.array([self.params[f"{kind}_{ii}"] for ii in range(self.n_points)])
        next_nodes = previous_nodes + 1
        spline_coefficients = self.nodes_to_spline_coefficients.dot(parameters)
        return (
            a * parameters[previous_nodes]
            + b * parameters[next_nodes]
            + c * spline_coefficients[previous_nodes]
            + d * spline_coefficients[next_nodes]
        )

    def get_calibration_factor(self, frequency_array, **params):
        """Apply calibration model

        Parameters
        ==========
        frequency_array: array-like
            The frequency values to calculate the calibration factor for.
        prefix: str
            Prefix for calibration parameter names
        params : dict
            Dictionary of sampling parameters which includes
            calibration parameters.

        Returns
        =======
        calibration_factor : array-like
            The factor to multiply the strain by.
        """
        log10f_per_deltalog10f = (
            np.log10(frequency_array) - self.log_spline_points[0]
        ) / self.delta_log_spline_points
        previous_nodes = np.clip(np.floor(log10f_per_deltalog10f).astype(int), a_min=0, a_max=self.n_points - 2)
        b = log10f_per_deltalog10f - previous_nodes
        a = 1 - b
        c = (a**3 - a) / 6
        d = (b**3 - b) / 6

        self.set_calibration_parameters(**params)

        delta_amplitude = self._evaluate_spline("amplitude", a, b, c, d, previous_nodes)
        delta_phase = self._evaluate_spline("phase", a, b, c, d, previous_nodes)
        calibration_factor = (1 + delta_amplitude) * (2 + 1j * delta_phase) / (2 - 1j * delta_phase)

        return calibration_factor


class Precomputed(Recalibrate):

    name = "precomputed"

    def __init__(self, label, curves, frequency_array, parameters=None):
        """
        A class for accessing an array of precomputed recalibration curves.

        Parameters
        ==========
        label: str
            The label for the interferometer, e.g., H1. The corresponding
            parameter is :code:`recalib_index_{label}`.
        curves: array-like
            Array with shape (n_curves, n_frequencies) with the recalibration
            curves.
        frequency_array: array-like
            Array of frequencies at which the curves are evaluated.
        """
        self.label = label
        self.curves = curves
        self.frequency_array = frequency_array
        self.parameters = parameters
        super(Precomputed, self).__init__(prefix=f"recalib_index_{self.label}")

    def get_calibration_factor(self, frequency_array, **params):
        idx = int(params.get(self.prefix, None))
        if idx is None:
            raise KeyError(f"Calibration index for {self.label} not found.")
        if not np.array_equal(frequency_array, self.frequency_array):
            raise ValueError("Frequency grid passed to calibrator doesn't match.")
        return self.curves[idx]

    @classmethod
    def constant_uncertainty_spline(
        cls, amplitude_sigma, phase_sigma, frequency_array, n_nodes, label, n_curves
    ):
        priors = CalibrationPriorDict.constant_uncertainty_spline(
            amplitude_sigma=amplitude_sigma,
            phase_sigma=phase_sigma,
            minimum_frequency=frequency_array[0],
            maximum_frequency=frequency_array[-1],
            n_nodes=n_nodes,
            label=label,
        )
        parameters = pd.DataFrame(priors.sample(n_curves))
        curves = curves_from_spline_and_prior(
            label=label,
            frequency_array=frequency_array,
            n_points=n_nodes,
            parameters=parameters,
            n_curves=n_curves
        )
        return cls(
            label=label,
            curves=np.array(curves),
            frequency_array=frequency_array,
            parameters=parameters,
        )

    @classmethod
    def from_envelope_file(
        cls, envelope, frequency_array, n_nodes, label, n_curves
    ):
        priors = CalibrationPriorDict.from_envelope_file(
            envelope_file=envelope,
            minimum_frequency=frequency_array[0],
            maximum_frequency=frequency_array[-1],
            n_nodes=n_nodes,
            label=label,
        )
        parameters = pd.DataFrame(priors.sample(n_curves))
        curves = curves_from_spline_and_prior(
            label=label,
            frequency_array=frequency_array,
            n_points=n_nodes,
            parameters=parameters,
            n_curves=n_curves,
        )
        return cls(
            label=label,
            curves=np.array(curves),
            frequency_array=frequency_array,
            parameters=parameters,
        )

    @classmethod
    def from_calibration_file(cls, label, filename, frequency_array, n_curves, starting_index=0):
        curves, parameters = read_calibration_file(
            filename=filename,
            frequency_array=frequency_array,
            number_of_response_curves=n_curves,
            starting_index=starting_index,
        )
        return cls(
            label=label,
            curves=np.array(curves),
            frequency_array=frequency_array,
            parameters=parameters,
        )


def build_calibration_lookup(
    interferometers,
    lookup_files=None,
    priors=None,
    number_of_response_curves=1000,
    starting_index=0,
):
    if lookup_files is None and priors is None:
        raise ValueError(
            "One of calibration_lookup_table or priors must be specified for "
            "building calibration marginalization lookup table."
        )
    elif lookup_files is None:
        lookup_files = dict()

    draws = dict()
    parameters = dict()
    for interferometer in interferometers:
        name = interferometer.name
        frequencies = interferometer.frequency_array
        frequencies = frequencies[interferometer.frequency_mask]
        filename = lookup_files.get(name, f"{name}_calibration_file.h5")

        if os.path.exists(filename):
            draws[name], parameters[name] = read_calibration_file(
                filename,
                frequencies,
                number_of_response_curves,
                starting_index,
            )
        elif isinstance(interferometer.calibration_model, Precomputed):
            model = interferometer.calibration_model
            idxs = np.arange(number_of_response_curves, dtype=int) + starting_index
            draws[name] = model.curves[idxs]
            parameters[name] = pd.DataFrame(model.parameters.iloc[idxs])
            parameters[name][model.prefix] = idxs
        else:
            if priors is None:
                raise ValueError(
                    "Priors must be passed to generate calibration response curves "
                    "for cubic spline."
                )
            draws[name], parameters[name] = _generate_calibration_draws(
                interferometer=interferometer,
                priors=priors,
                n_curves=number_of_response_curves,
            )
            write_calibration_file(filename, frequencies, draws[name], parameters[name])

        interferometer.calibration_model = Recalibrate()

    return draws, parameters


def _generate_calibration_draws(interferometer, priors, n_curves):
    name = interferometer.name
    frequencies = interferometer.frequency_array
    frequencies = frequencies[interferometer.frequency_mask]
    calibration_priors = PriorDict()
    for key in priors.keys():
        if "recalib" in key and name in key:
            calibration_priors[key] = copy.copy(priors[key])

    parameters = pd.DataFrame(calibration_priors.sample(n_curves))

    draws = np.array(curves_from_spline_and_prior(
        parameters=parameters,
        label=name,
        n_points=interferometer.calibration_model.n_points,
        frequency_array=frequencies,
        n_curves=n_curves,
    ))
    return draws, parameters


def curves_from_spline_and_prior(parameters, label, n_points, frequency_array, n_curves):
    spline = CubicSpline(
        prefix=f"recalib_{label}_",
        minimum_frequency=frequency_array[0],
        maximum_frequency=frequency_array[-1],
        n_points=n_points,
    )
    curves = list()
    for ii in range(n_curves):
        curves.append(spline.get_calibration_factor(
            frequency_array=frequency_array,
            **parameters.iloc[ii]
        ))
    return curves
