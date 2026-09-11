import os

import numpy as np

from ...core import utils
from ...core.utils import PropertyAccessor, docstring, logger, safe_file_dump
from ...compat.utils import array_module
from .. import utils as gwutils
from ..geometry import (
    get_polarization_tensor,
    three_by_three_matrix_contraction,
    time_delay_from_geocenter,
)
from .calibration import Recalibrate
from .geometry import InterferometerGeometry
from .strain_data import InterferometerStrainData
from ..conversion import generate_all_bbh_parameters


class Interferometer(object):
    """Class for the Interferometer """

    length = PropertyAccessor('geometry', 'length')
    latitude = PropertyAccessor('geometry', 'latitude')
    latitude_radians = PropertyAccessor('geometry', 'latitude_radians')
    longitude = PropertyAccessor('geometry', 'longitude')
    longitude_radians = PropertyAccessor('geometry', 'longitude_radians')
    elevation = PropertyAccessor('geometry', 'elevation')
    x = PropertyAccessor('geometry', 'x')
    y = PropertyAccessor('geometry', 'y')
    xarm_azimuth = PropertyAccessor('geometry', 'xarm_azimuth')
    yarm_azimuth = PropertyAccessor('geometry', 'yarm_azimuth')
    xarm_tilt = PropertyAccessor('geometry', 'xarm_tilt')
    yarm_tilt = PropertyAccessor('geometry', 'yarm_tilt')
    vertex = PropertyAccessor('geometry', 'vertex')
    detector_tensor = PropertyAccessor('geometry', 'detector_tensor')

    duration = PropertyAccessor('strain_data', 'duration')
    sampling_frequency = PropertyAccessor('strain_data', 'sampling_frequency')
    start_time = PropertyAccessor('strain_data', 'start_time')
    frequency_array = PropertyAccessor('strain_data', 'frequency_array')
    time_array = PropertyAccessor('strain_data', 'time_array')
    minimum_frequency = PropertyAccessor('strain_data', 'minimum_frequency')
    maximum_frequency = PropertyAccessor('strain_data', 'maximum_frequency')
    frequency_mask = PropertyAccessor('strain_data', 'frequency_mask')
    frequency_domain_strain = PropertyAccessor('strain_data', 'frequency_domain_strain')
    time_domain_strain = PropertyAccessor('strain_data', 'time_domain_strain')

    def __init__(self, name, power_spectral_density, minimum_frequency, maximum_frequency, length, latitude, longitude,
                 elevation, xarm_azimuth, yarm_azimuth, xarm_tilt=0., yarm_tilt=0., calibration_model=Recalibrate()):
        """
        Instantiate an Interferometer object.

        Parameters
        ==========
        name: str
            Interferometer name, e.g., H1.
        power_spectral_density: bilby.gw.detector.PowerSpectralDensity
            Power spectral density determining the sensitivity of the detector.
        minimum_frequency: float
            Minimum frequency to analyse for detector.
        maximum_frequency: float
            Maximum frequency to analyse for detector.
        length: float
            Length of the interferometer in km.
        latitude: float
            Latitude North in degrees (South is negative).
        longitude: float
            Longitude East in degrees (West is negative).
        elevation: float
            Height above surface in metres.
        xarm_azimuth: float
            Orientation of the x arm in degrees North of East.
        yarm_azimuth: float
            Orientation of the y arm in degrees North of East.
        xarm_tilt: float, optional
            Tilt of the x arm in radians above the horizontal defined by
            ellipsoid earth model in LIGO-T980044-08.
        yarm_tilt: float, optional
            Tilt of the y arm in radians above the horizontal.
        calibration_model: Recalibration
            Calibration model, this applies the calibration correction to the
            template, the default model applies no correction.
        """
        self.geometry = InterferometerGeometry(length, latitude, longitude, elevation,
                                               xarm_azimuth, yarm_azimuth, xarm_tilt, yarm_tilt)

        self.name = name
        self.power_spectral_density = power_spectral_density
        self.calibration_model = calibration_model
        self.strain_data = InterferometerStrainData(
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency)
        self.meta_data = dict(name=name)
        self.reference_time = None

    def __eq__(self, other):
        if self.name == other.name and \
                self.geometry == other.geometry and \
                self.power_spectral_density.__eq__(other.power_spectral_density) and \
                self.calibration_model == other.calibration_model and \
                self.strain_data == other.strain_data:
            return True
        return False

    def __repr__(self):
        return self.__class__.__name__ + '(name=\'{}\', power_spectral_density={}, minimum_frequency={}, ' \
                                         'maximum_frequency={}, length={}, latitude={}, longitude={}, elevation={}, ' \
                                         'xarm_azimuth={}, yarm_azimuth={}, xarm_tilt={}, yarm_tilt={})' \
            .format(self.name, self.power_spectral_density, float(self.strain_data.minimum_frequency),
                    float(self.strain_data.maximum_frequency), float(self.geometry.length),
                    float(self.geometry.latitude), float(self.geometry.longitude),
                    float(self.geometry.elevation), float(self.geometry.xarm_azimuth),
                    float(self.geometry.yarm_azimuth), float(self.geometry.xarm_tilt),
                    float(self.geometry.yarm_tilt))

    def set_strain_data_from_gwpy_timeseries(self, time_series, *, xp=None):
        """ Set the `Interferometer.strain_data` from a gwpy TimeSeries

        Parameters
        ==========
        time_series: gwpy.timeseries.timeseries.TimeSeries
            The data to set.
        xp: array module, optional
            The array module to use, e.g., :code:`numpy` or :code:`jax.numpy`.
            If not specified :code:`numpy` will be used.

        """
        self.strain_data.set_from_gwpy_timeseries(time_series=time_series, xp=xp)

    def set_strain_data_from_frequency_domain_strain(
            self, frequency_domain_strain, sampling_frequency=None,
            duration=None, start_time=0, frequency_array=None):
        """ Set the `Interferometer.strain_data` from a numpy array

        Parameters
        ==========
        frequency_domain_strain: array_like
            The data to set.
        sampling_frequency: float
            The sampling frequency (in Hz).
        duration: float
            The data duration (in s).
        start_time: float
            The GPS start-time of the data.
        frequency_array: array_like
            The array of frequencies, if sampling_frequency and duration not
            given.

        """
        self.strain_data.set_from_frequency_domain_strain(
            frequency_domain_strain=frequency_domain_strain,
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=start_time, frequency_array=frequency_array)

    def set_strain_data_from_power_spectral_density(
            self, sampling_frequency, duration, start_time=0, *, random_state=None):
        """ Set the `Interferometer.strain_data` from a power spectal density

        This uses the `interferometer.power_spectral_density` object to set
        the `strain_data` to a noise realization. See
        `bilby.gw.detector.InterferometerStrainData` for further information.

        Parameters
        ==========
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """
        self.strain_data.set_from_power_spectral_density(
            self.power_spectral_density, sampling_frequency=sampling_frequency,
            duration=duration, start_time=start_time, random_state=random_state)

    def set_strain_data_from_frame_file(
            self, frame_file, sampling_frequency, duration, start_time=0,
            channel=None, buffer_time=1, *, xp=None):
        """ Set the `Interferometer.strain_data` from a frame file

        Parameters
        ==========
        frame_file: str
            File from which to load data.
        channel: str
            Channel to read from frame.
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data
        buffer_time: float
            Read in data with `start_time-buffer_time` and
            `start_time+duration+buffer_time`
        xp: array module, optional
            The array module to use, e.g., :code:`numpy` or :code:`jax.numpy`.
            If not specified :code:`numpy` will be used.

        """
        self.strain_data.set_from_frame_file(
            frame_file=frame_file, sampling_frequency=sampling_frequency,
            duration=duration, start_time=start_time,
            channel=channel, buffer_time=buffer_time, xp=xp)

    def set_strain_data_from_channel_name(
            self, channel, sampling_frequency, duration, start_time=0, *, xp=None):
        """
        Set the `Interferometer.strain_data` by fetching from given channel
        using strain_data.set_from_channel_name()

        Parameters
        ==========
        channel: str
            Channel to look for using gwpy in the format `IFO:Channel`
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data
        xp: array module, optional
            The array module to use, e.g., :code:`numpy` or :code:`jax.numpy`.
            If not specified :code:`numpy` will be used.

        """
        self.strain_data.set_from_channel_name(
            channel=channel, sampling_frequency=sampling_frequency,
            duration=duration, start_time=start_time, xp=xp)

    def set_strain_data_from_csv(self, filename, *, xp=None):
        """ Set the `Interferometer.strain_data` from a csv file

        Parameters
        ==========
        filename: str
            The path to the file to read in
        xp: array module, optional
            The array module to use, e.g., :code:`numpy` or :code:`jax.numpy`.
            If not specified :code:`numpy` will be used.

        """
        self.strain_data.set_from_csv(filename, xp=xp)

    def set_strain_data_from_zero_noise(
            self, sampling_frequency, duration, start_time=0):
        """ Set the `Interferometer.strain_data` to zero noise

        Parameters
        ==========
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """

        self.strain_data.set_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=start_time)

    def antenna_response(self, ra, dec, time, psi, mode):
        """
        Calculate the antenna response function for a given sky location

        See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
        [u, v, w] represent the Earth-frame
        [m, n, omega] represent the wave-frame
        Note: there is a typo in the definition of the wave-frame in Nishizawa et al.

        Parameters
        ==========
        ra: float
            right ascension in radians
        dec: float
            declination in radians
        time: float
            geocentric GPS time
        psi: float
            binary polarisation angle counter-clockwise about the direction of propagation
        mode: str
            polarisation mode (e.g. 'plus', 'cross') or the name of a specific detector.
            If mode == self.name, return 1

        Returns
        =======
        float: The antenna response for the specified mode and time/location

        """
        if mode in ["plus", "cross", "x", "y", "breathing", "longitudinal"]:
            polarization_tensor = get_polarization_tensor(ra, dec, time, psi, mode)
            return three_by_three_matrix_contraction(self.geometry.detector_tensor, polarization_tensor)
        elif mode == self.name:
            return 1
        else:
            return 0

    def get_detector_response(self, waveform_polarizations, parameters, frequencies=None):
        """ Get the detector response for a particular waveform

        Parameters
        ==========
        waveform_polarizations: dict
            polarizations of the waveform
        parameters: dict
            parameters describing position and time of arrival of the signal
        frequencies: array-like, optional
        The frequency values to evaluate the response at. If
        not provided, the response is computed using
        :code:`self.frequency_array`. If the frequencies are
        specified, no frequency masking is performed.

        Returns
        =======
        array_like: A 3x3 array representation of the detector response (signal observed in the interferometer)

        Notes
        =====
        If the :code:`reference_time` attribute is not :code:`None`, this is
        used to set the time at which the antenna response is evaluated,
        otherwise the provided :code:`Parameters["geocent_time"]` is used.
        """
        xp = array_module(waveform_polarizations)
        if frequencies is None:
            frequencies = self.frequency_array
            mask = self.frequency_mask
        else:
            mask = xp.ones(len(frequencies), dtype=bool)
        frequencies = xp.asarray(frequencies)

        if self.reference_time is None:
            antenna_time = parameters["geocent_time"]
        else:
            antenna_time = self.reference_time

        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = self.antenna_response(
                parameters['ra'],
                parameters['dec'],
                antenna_time,
                parameters['psi'], mode)

            signal[mode] = waveform_polarizations[mode] * mask * det_response
        signal_ifo = sum(signal.values())

        time_shift = self.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time'])

        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - self.strain_data.start_time
        dt = dt_geocent + time_shift

        xp = array_module(signal_ifo)

        signal_ifo = signal_ifo * xp.exp(-1j * 2 * np.pi * dt * frequencies)

        signal_ifo *= self.calibration_model.get_calibration_factor(
            frequencies, prefix=f'recalib_{self.name}_', xp=xp, **parameters
        )

        return signal_ifo

    def check_signal_duration(self, parameters, raise_error=True):
        """ Check that the signal with the given parameters fits in the data

        Parameters
        ==========
        parameters: dict
            A dictionary of the injection parameters
        raise_error: bool
            If True, raise an error in the signal does not fit. Otherwise, print
            a warning message.
        """
        try:
            parameters = generate_all_bbh_parameters(parameters)
        except AttributeError:
            logger.debug(
                "generate_all_bbh_parameters parameters failed during check_signal_duration"
            )
            return

        if ("mass_1" not in parameters) and ("mass_2" not in parameters):
            if raise_error:
                raise AttributeError("Unable to check signal duration as mass not given")
            else:
                return

        # Calculate the time to merger
        deltaT = gwutils.calculate_time_to_merger(
            frequency=self.minimum_frequency,
            mass_1=parameters["mass_1"],
            mass_2=parameters["mass_2"],
        )
        deltaT = np.round(deltaT, 1)
        if deltaT > self.duration:
            msg = (
                f"The injected signal has a duration in-band of {deltaT}s, but "
                f"the data for detector {self.name} has a duration of {self.duration}s"
            )
            if raise_error:
                raise ValueError(msg)
            else:
                logger.warning(msg)

    def inject_signal(self, parameters, injection_polarizations=None,
                      waveform_generator=None, raise_error=True):
        """ General signal injection method.
        Provide the injection parameters and either the injection polarizations
        or the waveform generator to inject a signal into the detector.
        Defaults to the injection polarizations is both are given.

        Parameters
        ==========
        parameters: dict
            Parameters of the injection.
        injection_polarizations: dict, optional
           Polarizations of waveform to inject, output of
           `waveform_generator.frequency_domain_strain()`. If
           `waveform_generator` is also given, the injection_polarizations will
           be calculated directly and this argument can be ignored.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator, optional
            A WaveformGenerator instance using the source model to inject. If
            `injection_polarizations` is given, this will be ignored.
        raise_error: bool
            If true, raise an error if the injected signal has a duration
            longer than the data duration. If False, a warning will be printed
            instead.

        Notes
        =====
        if your signal takes a substantial amount of time to generate, or
        you experience buggy behaviour. It is preferable to provide the
        injection_polarizations directly.

        Returns
        =======
        injection_polarizations: dict
            The injected polarizations. This is the same as the injection_polarizations parameters
            if it was passed in. Otherwise it is the return value of waveform_generator.frequency_domain_strain().

        """
        self.check_signal_duration(parameters, raise_error)

        if injection_polarizations is None and waveform_generator is None:
            raise ValueError(
                "inject_signal needs one of waveform_generator or "
                "injection_polarizations.")
        elif injection_polarizations is not None:
            self.inject_signal_from_waveform_polarizations(parameters=parameters,
                                                           injection_polarizations=injection_polarizations)
        elif waveform_generator is not None:
            injection_polarizations = self.inject_signal_from_waveform_generator(parameters=parameters,
                                                                                 waveform_generator=waveform_generator)
        return injection_polarizations

    def inject_signal_from_waveform_generator(self, parameters, waveform_generator):
        """ Inject a signal using a waveform generator and a set of parameters.
        Alternative to `inject_signal` and `inject_signal_from_waveform_polarizations`

        Parameters
        ==========
        parameters: dict
            Parameters of the injection.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.

        Notes
        =====
        if your signal takes a substantial amount of time to generate, or
        you experience buggy behaviour. It is preferable to use the
        inject_signal_from_waveform_polarizations() method.

        Returns
        =======
        injection_polarizations: dict
            The internally generated injection parameters

        """
        injection_polarizations = \
            waveform_generator.frequency_domain_strain(parameters)
        self.inject_signal_from_waveform_polarizations(parameters=parameters,
                                                       injection_polarizations=injection_polarizations)
        return injection_polarizations

    def inject_signal_from_waveform_polarizations(self, parameters, injection_polarizations):
        """ Inject a signal into the detector from a dict of waveform polarizations.
        Alternative to `inject_signal` and `inject_signal_from_waveform_generator`.

        Parameters
        ==========
        parameters: dict
            Parameters of the injection.
        injection_polarizations: dict
           Polarizations of waveform to inject, output of
           `waveform_generator.frequency_domain_strain()`.

        """
        if not self.strain_data.time_within_data(parameters['geocent_time']):
            logger.warning(
                'Injecting signal outside segment, start_time={}, merger time={}.'
                .format(self.strain_data.start_time, parameters['geocent_time']))

        signal_ifo = self.get_detector_response(injection_polarizations, parameters)
        self.strain_data.frequency_domain_strain += signal_ifo

        self.meta_data['optimal_SNR'] = (
            self.optimal_snr_squared(signal=signal_ifo)).real ** 0.5
        self.meta_data['matched_filter_SNR'] = (
            self.matched_filter_snr(signal=signal_ifo))
        self.meta_data['parameters'] = parameters

        logger.info("Injected signal in {}:".format(self.name))
        logger.info("  optimal SNR = {:.2f}".format(self.meta_data['optimal_SNR']))
        logger.info("  matched filter SNR = {:.2f}".format(self.meta_data['matched_filter_SNR']))
        for key in parameters:
            logger.info('  {} = {}'.format(key, parameters[key]))

    @property
    def amplitude_spectral_density_array(self):
        """ Returns the amplitude spectral density (ASD) given we know a power spectral density (PSD)

        Returns
        =======
        array_like: An array representation of the ASD

        """
        return self.power_spectral_density.get_amplitude_spectral_density_array(
            frequency_array=self.strain_data.frequency_array
        )

    @property
    def power_spectral_density_array(self):
        """ Returns the power spectral density (PSD)

        This accounts for whether the data in the interferometer has been windowed.

        Returns
        =======
        array_like: An array representation of the PSD

        """
        return self.power_spectral_density.get_power_spectral_density_array(
            frequency_array=self.strain_data.frequency_array
        )

    def unit_vector_along_arm(self, arm):
        logger.warning("This method has been moved and will be removed in the future."
                       "Use Interferometer.geometry.unit_vector_along_arm instead.")
        return self.geometry.unit_vector_along_arm(arm)

    def time_delay_from_geocenter(self, ra, dec, time):
        """
        Calculate the time delay from the geocenter for the interferometer.

        Use the time delay function from utils.

        Parameters
        ==========
        ra: float
            right ascension of source in radians
        dec: float
            declination of source in radians
        time: float
            GPS time

        Returns
        =======
        float: The time delay from geocenter in seconds
        """
        return time_delay_from_geocenter(self.geometry.vertex, ra, dec, time)

    def vertex_position_geocentric(self):
        """
        Calculate the position of the IFO vertex in geocentric coordinates in meters.

        Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
        See Section 2.1 of LIGO-T980044-10 for the correct expression

        Returns
        =======
        array_like: A 3D array representation of the vertex
        """
        return gwutils.get_vertex_position_geocentric(self.geometry.latitude_radians,
                                                      self.geometry.longitude_radians,
                                                      self.geometry.elevation)

    def optimal_snr_squared(self, signal):
        """

        Parameters
        ==========
        signal: array_like
            Array containing the signal

        Returns
        =======
        float: The optimal signal to noise ratio possible squared
        """
        mask = self.strain_data.frequency_mask
        xp = array_module(signal)
        return gwutils.optimal_snr_squared(
            signal=signal * mask,
            power_spectral_density=xp.where(mask, self.power_spectral_density_array, xp.inf),
            duration=self.strain_data.duration)

    def inner_product(self, signal):
        """

        Parameters
        ==========
        signal: array_like
            Array containing the signal

        Returns
        =======
        float: The optimal signal to noise ratio possible squared
        """
        mask = self.strain_data.frequency_mask
        xp = array_module(signal)
        return gwutils.noise_weighted_inner_product(
            aa=signal * mask,
            bb=self.strain_data.frequency_domain_strain * mask,
            power_spectral_density=xp.where(mask, self.power_spectral_density_array, xp.inf),
            duration=self.strain_data.duration)

    def template_template_inner_product(self, signal_1, signal_2):
        """A noise weighted inner product between two templates, using this ifo's PSD.

        Parameters
        ==========
        signal_1 : array_like
            An array containing the first signal
        signal_2 : array_like
            an array containing the second signal

        Returns
        =======
        float: The noise weighted inner product of the two templates
        """
        return gwutils.noise_weighted_inner_product(
            aa=signal_1[self.strain_data.frequency_mask],
            bb=signal_2[self.strain_data.frequency_mask],
            power_spectral_density=self.power_spectral_density_array[self.strain_data.frequency_mask],
            duration=self.strain_data.duration)

    def matched_filter_snr(self, signal):
        """

        Parameters
        ==========
        signal: array_like
            Array containing the signal

        Returns
        =======
        complex: The matched filter signal to noise ratio

        """
        return gwutils.matched_filter_snr(
            signal=signal[self.strain_data.frequency_mask],
            frequency_domain_strain=self.strain_data.frequency_domain_strain[self.strain_data.frequency_mask],
            power_spectral_density=self.power_spectral_density_array[self.strain_data.frequency_mask],
            duration=self.strain_data.duration)

    def whiten_frequency_series(self, frequency_series : np.array) -> np.array:
        """Whitens a frequency series with the noise properties of the detector

        .. math::
            \\tilde{a}_w(f) = \\tilde{a}(f) \\sqrt{\\frac{4}{T S_n(f)}}

        Such that

        .. math::
            Var(n) = \\frac{1}{N} \\sum_{k=0}^N n_W(f_k)n_W^*(f_k) = 2

        Where the factor of two is due to the independent real and imaginary
        components.

        Parameters
        ==========
        frequency_series : np.array
            The frequency series, whitened by the ASD
        """
        return frequency_series / (self.amplitude_spectral_density_array * (self.duration / 4)**0.5)

    def get_whitened_time_series_from_whitened_frequency_series(
        self,
        whitened_frequency_series : np.array
    ) -> np.array:
        """Gets the whitened time series from a whitened frequency series.

        This ifft's and also applies a windowing factor,
        since when f_min and f_max are set bilby applies a mask to the series.

        Per 6.2a-b in https://arxiv.org/pdf/gr-qc/0509116 since our window
        is just a band pass,
        this coefficient is :math:`w/W` where

        .. math::

            W = \\frac{1}{N} \\sum_{k=0}^N w^2[j]

        Since our window :math:`w` is simply 1 or 0, depending on the mask, we get

        .. math::

            W = \\frac{1}{N} \\sum_{k=0}^N \\Theta(f_{max} - f_k)\\Theta(f_k - f_{min})

        and accordingly the termwise window factor is

        .. math::
            w = \\sqrt{N W} = \\sqrt{\\sum_{k=0}^N \\Theta(f_{max} - f_k)\\Theta(f_k - f_{min})}

        """
        xp = array_module(whitened_frequency_series)

        frequency_window_factor = self.frequency_mask.mean()

        whitened_time_series = (
            xp.fft.irfft(whitened_frequency_series)
            * self.frequency_mask.sum()**0.5 / frequency_window_factor
        )

        return whitened_time_series

    @property
    def whitened_frequency_domain_strain(self):
        r"""Whitens the frequency domain data by dividing through by ASD,
        with appropriate normalization.

        See `whiten_frequency_series()` for details.

        Returns
        =======
        array_like: The whitened data
        """
        return self.whiten_frequency_series(self.strain_data.frequency_domain_strain)

    @property
    def whitened_time_domain_strain(self) -> np.array:
        """Calculates the whitened time domain strain
        by iffting the whitened frequency domain strain,
        with the appropriate normalization.

        See `get_whitened_time_series_from_whitened_frequency_series()` for details

        Returns
        =======
        array_like
            The whitened data in the time domain
        """
        return self.get_whitened_time_series_from_whitened_frequency_series(self.whitened_frequency_domain_strain)

    def save_data(self, outdir, label=None):
        """ Creates save files for interferometer data in plain text format.

        Saves two files: the frequency domain strain data with three columns [f, real part of h(f),
        imaginary part of h(f)], and the amplitude spectral density with two columns [f, ASD(f)].

        Note that in v1.3.0 and below, the ASD was saved in a file called *_psd.dat.

        Parameters
        ==========
        outdir: str
            The output directory in which the data is supposed to be saved
        label: str
            The name of the output files
        """

        if label is None:
            filename_asd = '{}/{}_asd.dat'.format(outdir, self.name)
            filename_data = '{}/{}_frequency_domain_data.dat'.format(outdir, self.name)
        else:
            filename_asd = '{}/{}_{}_asd.dat'.format(outdir, self.name, label)
            filename_data = '{}/{}_{}_frequency_domain_data.dat'.format(outdir, self.name, label)
        np.savetxt(filename_data,
                   np.array(
                       [self.strain_data.frequency_array,
                        self.strain_data.frequency_domain_strain.real,
                        self.strain_data.frequency_domain_strain.imag]).T,
                   header='f real_h(f) imag_h(f)')
        np.savetxt(filename_asd,
                   np.array(
                       [self.strain_data.frequency_array,
                        self.amplitude_spectral_density_array]).T,
                   header='f h(f)')

    def plot_data(self, signal=None, outdir='.', label=None):
        import matplotlib.pyplot as plt
        if utils.command_line_args.bilby_test_mode:
            return

        fig, ax = plt.subplots()
        df = self.strain_data.frequency_array[1] - self.strain_data.frequency_array[0]
        asd = gwutils.asd_from_freq_series(
            freq_data=self.strain_data.frequency_domain_strain, df=df)

        ax.loglog(self.strain_data.frequency_array[self.strain_data.frequency_mask],
                  asd[self.strain_data.frequency_mask],
                  color='C0', label=self.name)
        ax.loglog(self.strain_data.frequency_array[self.strain_data.frequency_mask],
                  self.amplitude_spectral_density_array[self.strain_data.frequency_mask],
                  color='C1', lw=1.0, label=self.name + ' ASD')
        if signal is not None:
            signal_asd = gwutils.asd_from_freq_series(
                freq_data=signal, df=df)

            ax.loglog(self.strain_data.frequency_array[self.strain_data.frequency_mask],
                      signal_asd[self.strain_data.frequency_mask],
                      color='C2',
                      label='Signal')
        ax.grid(True)
        ax.set_ylabel(r'Strain [strain/$\sqrt{\rm Hz}$]')
        ax.set_xlabel(r'Frequency [Hz]')
        ax.legend(loc='best')
        fig.tight_layout()
        if label is None:
            fig.savefig(
                '{}/{}_frequency_domain_data.png'.format(outdir, self.name))
        else:
            fig.savefig(
                '{}/{}_{}_frequency_domain_data.png'.format(
                    outdir, self.name, label))
        plt.close(fig)

    def plot_time_domain_data(
            self, outdir='.', label=None, bandpass_frequencies=(50, 250),
            notches=None, start_end=None, t0=None):
        """ Plots the strain data in the time domain

        Parameters
        ==========
        outdir, label: str
            Used in setting the saved filename.
        bandpass: tuple, optional
            A tuple of the (low, high) frequencies to use when bandpassing the
            data, if None no bandpass is applied.
        notches: list, optional
            A list of frequencies specifying any lines to notch
        start_end: tuple
            A tuple of the (start, end) range of GPS times to plot
        t0: float
            If given, the reference time to subtract from the time series before
            plotting.

        """
        import matplotlib.pyplot as plt
        from gwpy.timeseries import TimeSeries
        from gwpy.signal.filter_design import bandpass, concatenate_zpks, notch

        # We use the gwpy timeseries to perform bandpass and notching
        if notches is None:
            notches = list()
        timeseries = TimeSeries(
            data=self.strain_data.time_domain_strain, times=self.strain_data.time_array)
        zpks = []
        if bandpass_frequencies is not None:
            zpks.append(bandpass(
                bandpass_frequencies[0], bandpass_frequencies[1],
                self.strain_data.sampling_frequency))
        if notches is not None:
            for line in notches:
                zpks.append(notch(
                    line, self.strain_data.sampling_frequency))
        if len(zpks) > 0:
            zpk = concatenate_zpks(*zpks)
            strain = timeseries.filter(zpk, filtfilt=False)
        else:
            strain = timeseries

        fig, ax = plt.subplots()

        if t0:
            x = self.strain_data.time_array - t0
            xlabel = 'GPS time [s] - {}'.format(t0)
        else:
            x = self.strain_data.time_array
            xlabel = 'GPS time [s]'

        ax.plot(x, strain)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Strain')

        if start_end is not None:
            ax.set_xlim(*start_end)

        fig.tight_layout()

        if label is None:
            fig.savefig(
                '{}/{}_time_domain_data.png'.format(outdir, self.name))
        else:
            fig.savefig(
                '{}/{}_{}_time_domain_data.png'.format(outdir, self.name, label))
        plt.close(fig)

    @staticmethod
    def _filename_from_outdir_label_extension(outdir, label, extension="h5"):
        return os.path.join(outdir, label + f'.{extension}')

    _save_ifo_docstring = """ Save the object to a {format} file

    {extra}

    Attributes
    ==========
    outdir: str, optional
        Output directory name of the file, defaults to 'outdir'.
    label: str, optional
        Output file name, is self.name if not given otherwise.
    """

    _load_docstring = """ Loads in an Interferometer object from a {format} file

    Parameters
    ==========
    filename: str
        If given, try to load from this filename

    """

    @docstring(_save_ifo_docstring.format(
        format="pickle", extra=".. versionadded:: 1.1.0"
    ))
    def to_pickle(self, outdir="outdir", label=None):
        utils.check_directory_exists_and_if_not_mkdir(outdir)
        filename = self._filename_from_outdir_label_extension(outdir, label, extension="pkl")
        safe_file_dump(self, filename, "dill")

    @classmethod
    @docstring(_load_docstring.format(format="pickle"))
    def from_pickle(cls, filename=None):
        import dill
        with open(filename, "rb") as ff:
            res = dill.load(ff)
        if res.__class__ != cls:
            raise TypeError('The loaded object is not an Interferometer')
        return res

    def set_array_backend(self, xp):
        self.geometry.set_array_backend(xp=xp)
        self.power_spectral_density.set_array_backend(xp=xp)

    @property
    def array_backend(self):
        return array_module(self.geometry.length)
