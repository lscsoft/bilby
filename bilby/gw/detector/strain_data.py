import array_api_compat as aac
import numpy as np

from ...compat.utils import array_module
from ...core import utils
from ...core.series import CoupledTimeAndFrequencySeries
from ...core.utils import logger, PropertyAccessor
from .. import utils as gwutils


class InterferometerStrainData(object):
    """ Strain data for an interferometer """

    duration = PropertyAccessor('_times_and_frequencies', 'duration')
    sampling_frequency = PropertyAccessor('_times_and_frequencies', 'sampling_frequency')
    start_time = PropertyAccessor('_times_and_frequencies', 'start_time')
    frequency_array = PropertyAccessor('_times_and_frequencies', 'frequency_array')
    time_array = PropertyAccessor('_times_and_frequencies', 'time_array')

    def __init__(self, minimum_frequency=0, maximum_frequency=np.inf,
                 roll_off=0.2, notch_list=None):
        """ Initiate an InterferometerStrainData object

        The initialised object contains no data, this should be added using one
        of the `set_from..` methods.

        Parameters
        ==========
        minimum_frequency: float
            Minimum frequency to analyse for detector. Default is 0.
        maximum_frequency: float
            Maximum frequency to analyse for detector. Default is infinity.
        roll_off: float
            The roll-off (in seconds) used in the Tukey window, default=0.2s.
            This corresponds to alpha * duration / 2 for scipy tukey window.
        notch_list: bilby.gw.detector.strain_data.NotchList
            A list of notches

        """

        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.notch_list = notch_list
        self.roll_off = roll_off
        self.window_factor = 1

        self._times_and_frequencies = CoupledTimeAndFrequencySeries()

        self._frequency_mask_updated = False
        self._frequency_mask = None
        self._frequency_domain_strain = None
        self._time_domain_strain = None
        self._channel = None

    def __eq__(self, other):
        if self.minimum_frequency == other.minimum_frequency \
                and self.maximum_frequency == other.maximum_frequency \
                and self.roll_off == other.roll_off \
                and self.window_factor == other.window_factor \
                and self.sampling_frequency == other.sampling_frequency \
                and self.duration == other.duration \
                and self.start_time == other.start_time \
                and np.array_equal(self.time_array, other.time_array) \
                and np.array_equal(self.frequency_array, other.frequency_array) \
                and np.array_equal(self.frequency_domain_strain, other.frequency_domain_strain) \
                and np.array_equal(self.time_domain_strain, other.time_domain_strain):
            return True
        return False

    def time_within_data(self, time):
        """ Check if time is within the data span

        Parameters
        ==========
        time: float
            The time to check

        Returns
        =======
        bool:
            A boolean stating whether the time is inside or outside the span

        """
        if time < self.start_time:
            logger.debug("Time is before the start_time")
            return False
        elif time > self.start_time + self.duration:
            logger.debug("Time is after the start_time + duration")
            return False
        else:
            return True

    @property
    def minimum_frequency(self):
        return self._minimum_frequency

    @minimum_frequency.setter
    def minimum_frequency(self, minimum_frequency):
        self._minimum_frequency = minimum_frequency
        self._frequency_mask_updated = False

    @property
    def maximum_frequency(self):
        """ Force the maximum frequency be less than the Nyquist frequency """
        if self.sampling_frequency is not None:
            xp = array_module(self._maximum_frequency)
            self._maximum_frequency = xp.minimum(
                self._maximum_frequency, self.sampling_frequency / 2
            )
        return self._maximum_frequency

    @maximum_frequency.setter
    def maximum_frequency(self, maximum_frequency):
        self._maximum_frequency = maximum_frequency
        self._frequency_mask_updated = False

    @property
    def notch_list(self):
        return self._notch_list

    @notch_list.setter
    def notch_list(self, notch_list):
        """ Set the notch_list

        Parameters
        ==========
        notch_list: list, bilby.gw.detector.strain_data.NotchList
            A list of length-2 tuples of the (max, min) frequency for the
            notches or a pre-made bilby NotchList.

        """
        if notch_list is None:
            self._notch_list = NotchList(None)
        elif isinstance(notch_list, list):
            self._notch_list = NotchList(notch_list)
        elif isinstance(notch_list, NotchList):
            self._notch_list = notch_list
        else:
            raise ValueError("notch_list {} not understood".format(notch_list))
        self._frequency_mask_updated = False

    @property
    def frequency_mask(self):
        """ Masking array for limiting the frequency band.

        Returns
        =======
        mask: np.ndarray
            An array of boolean values
        """
        if not self._frequency_mask_updated:
            frequency_array = self._times_and_frequencies.frequency_array
            mask = ((frequency_array >= self.minimum_frequency) &
                    (frequency_array <= self.maximum_frequency))
            for notch in self.notch_list:
                mask[notch.get_idxs(frequency_array)] = False
            self._frequency_mask = mask
            self._frequency_mask_updated = True
        return self._frequency_mask

    @frequency_mask.setter
    def frequency_mask(self, mask):
        self._frequency_mask = mask
        self._frequency_mask_updated = True

    @property
    def alpha(self):
        return 2 * self.roll_off / self.duration

    def time_domain_window(self, roll_off=None, alpha=None):
        """
        Window function to apply to time domain data before FFTing.

        This defines self.window_factor as the power loss due to the windowing.
        See https://dcc.ligo.org/DocDB/0027/T040089/000/T040089-00.pdf

        Parameters
        ==========
        roll_off: float
            Rise time of window in seconds
        alpha: float
            Parameter to pass to tukey window, how much of segment falls
            into windowed part

        Returns
        =======
        window: array
            Window function over time array
        """
        from scipy.signal.windows import tukey
        if roll_off is not None:
            self.roll_off = roll_off
        elif alpha is not None:
            self.roll_off = alpha * self.duration / 2
        window = tukey(len(self._time_domain_strain), alpha=self.alpha)
        self.window_factor = np.mean(window ** 2)
        return window

    @property
    def time_domain_strain(self):
        """ The time domain strain, in units of strain """
        if self._time_domain_strain is not None:
            return self._time_domain_strain
        elif self._frequency_domain_strain is not None:
            self._time_domain_strain = utils.infft(
                self.frequency_domain_strain, self.sampling_frequency)
            return self._time_domain_strain

        else:
            raise ValueError("time domain strain data not yet set")

    @property
    def frequency_domain_strain(self):
        """ Returns the frequency domain strain

        This is the frequency domain strain normalised to units of
        strain / Hz, obtained by a one-sided Fourier transform of the
        time domain data, divided by the sampling frequency.
        """
        if self._frequency_domain_strain is not None:
            return self._frequency_domain_strain * self.frequency_mask
        elif self._time_domain_strain is not None:
            logger.debug("Generating frequency domain strain from given time "
                         "domain strain.")
            logger.debug("Applying a tukey window with alpha={}, roll off={}".format(
                self.alpha, self.roll_off))
            # self.low_pass_filter()
            window = self.time_domain_window()
            self._frequency_domain_strain, self.frequency_array = utils.nfft(
                self._time_domain_strain * window, self.sampling_frequency)
            return self._frequency_domain_strain * self.frequency_mask
        else:
            raise ValueError("frequency domain strain data not yet set")

    @frequency_domain_strain.setter
    def frequency_domain_strain(self, frequency_domain_strain):
        if not len(self.frequency_array) == len(frequency_domain_strain):
            raise ValueError("The frequency_array and the set strain have different lengths")
        self._frequency_domain_strain = frequency_domain_strain
        self._time_domain_strain = None

    def to_gwpy_timeseries(self):
        """
        Output the time series strain data as a :class:`gwpy.timeseries.TimeSeries`.
        """
        try:
            from gwpy.timeseries import TimeSeries
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Cannot output strain data as gwpy TimeSeries")

        return TimeSeries(
            self.time_domain_strain, sample_rate=self.sampling_frequency,
            t0=self.start_time, channel=self.channel
        )

    def to_pycbc_timeseries(self):
        """
        Output the time series strain data as a :class:`pycbc.types.timeseries.TimeSeries`.
        """

        try:
            from pycbc.types.timeseries import TimeSeries
            from lal import LIGOTimeGPS
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Cannot output strain data as PyCBC TimeSeries")

        return TimeSeries(
            self.time_domain_strain, delta_t=(1. / self.sampling_frequency),
            epoch=LIGOTimeGPS(self.start_time)
        )

    def to_lal_timeseries(self):
        """
        Output the time series strain data as a LAL TimeSeries object.
        """
        try:
            from lal import CreateREAL8TimeSeries, LIGOTimeGPS, SecondUnit
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Cannot output strain data as PyCBC TimeSeries")

        lal_data = CreateREAL8TimeSeries(
            "", LIGOTimeGPS(self.start_time), 0, 1 / self.sampling_frequency,
            SecondUnit, len(self.time_domain_strain)
        )
        lal_data.data.data[:] = self.time_domain_strain

        return lal_data

    def to_gwpy_frequencyseries(self):
        """
        Output the frequency series strain data as a :class:`gwpy.frequencyseries.FrequencySeries`.
        """
        try:
            from gwpy.frequencyseries import FrequencySeries
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Cannot output strain data as gwpy FrequencySeries")

        return FrequencySeries(
            self.frequency_domain_strain,
            frequencies=self.frequency_array,
            epoch=self.start_time,
            channel=self.channel
        )

    def to_pycbc_frequencyseries(self):
        """
        Output the frequency series strain data as a :class:`pycbc.types.frequencyseries.FrequencySeries`.
        """

        try:
            from pycbc.types.frequencyseries import FrequencySeries
            from lal import LIGOTimeGPS
        except ImportError:
            raise ImportError("Cannot output strain data as PyCBC FrequencySeries")

        return FrequencySeries(
            self.frequency_domain_strain,
            delta_f=1 / self.duration,
            epoch=LIGOTimeGPS(self.start_time)
        )

    def to_lal_frequencyseries(self):
        """
        Output the frequency series strain data as a LAL FrequencySeries object.
        """
        try:
            from lal import CreateCOMPLEX16FrequencySeries, LIGOTimeGPS, SecondUnit
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Cannot output strain data as PyCBC TimeSeries")

        lal_data = CreateCOMPLEX16FrequencySeries(
            "",
            LIGOTimeGPS(self.start_time),
            self.frequency_array[0],
            1 / self.duration,
            SecondUnit,
            len(self.frequency_domain_strain)
        )
        lal_data.data.data[:] = self.frequency_domain_strain

        return lal_data

    def low_pass_filter(self, filter_freq=None):
        """ Low pass filter the data """
        from gwpy.signal.filter_design import lowpass
        from gwpy.timeseries import TimeSeries

        if filter_freq is None:
            logger.debug(
                "Setting low pass filter_freq using given maximum frequency")
            filter_freq = self.maximum_frequency

        if 2 * filter_freq >= self.sampling_frequency:
            logger.info(
                "Low pass filter frequency of {}Hz requested, this is equal"
                " or greater than the Nyquist frequency so no filter applied"
                .format(filter_freq))
            return

        logger.debug("Applying low pass filter with filter frequency {}".format(filter_freq))
        bp = lowpass(filter_freq, self.sampling_frequency)
        strain = TimeSeries(self.time_domain_strain, sample_rate=self.sampling_frequency)
        strain = strain.filter(bp, filtfilt=True)
        self._time_domain_strain = strain.value

    def create_power_spectral_density(
            self, fft_length, overlap=0, name='unknown', outdir=None,
            analysis_segment_start_time=None):
        """ Use the time domain strain to generate a power spectral density

        This create a Tukey-windowed power spectral density and writes it to a
        PSD file.

        Parameters
        ==========
        fft_length: float
            Duration of the analysis segment.
        overlap: float
            Number of seconds of overlap between FFTs.
        name: str
            The name of the detector, used in storing the PSD. Defaults to
            "unknown".
        outdir: str
            The output directory to write the PSD file too. If not given,
            the PSD will not be written to file.
        analysis_segment_start_time: float
            The start time of the analysis segment, if given, this data will
            be removed before creating the PSD.

        Returns
        =======
        frequency_array, psd : array_like
            The frequencies and power spectral density array

        """
        from gwpy.timeseries import TimeSeries

        data = self.time_domain_strain

        if analysis_segment_start_time is not None:
            analysis_segment_end_time = analysis_segment_start_time + fft_length
            inside = (analysis_segment_start_time > self.time_array[0] +
                      analysis_segment_end_time < self.time_array[-1])
            if inside:
                logger.info("Removing analysis segment data from the PSD data")
                idxs = (
                    (self.time_array < analysis_segment_start_time) +
                    (self.time_array > analysis_segment_end_time))
                data = data[idxs]

        # WARNING this line can cause issues if the data is non-contiguous
        strain = TimeSeries(data=data, sample_rate=self.sampling_frequency)
        psd_alpha = 2 * self.roll_off / fft_length
        logger.info(
            "Tukey window PSD data with alpha={}, roll off={}".format(
                psd_alpha, self.roll_off))
        psd = strain.psd(
            fftlength=fft_length, overlap=overlap, window=('tukey', psd_alpha))

        if outdir:
            psd_file = '{}/{}_PSD_{}_{}.txt'.format(outdir, name, self.start_time, self.duration)
            with open('{}'.format(psd_file), 'w+') as opened_file:
                for f, p in zip(psd.frequencies.value, psd.value):
                    opened_file.write('{} {}\n'.format(f, p))

        return psd.frequencies.value, psd.value

    def _infer_time_domain_dependence(
            self, start_time, sampling_frequency, duration, time_array):
        """ Helper function to figure out if the time_array, or
            sampling_frequency and duration where given
        """
        self._infer_dependence(domain='time', array=time_array, duration=duration,
                               sampling_frequency=sampling_frequency, start_time=start_time)

    def _infer_frequency_domain_dependence(
            self, start_time, sampling_frequency, duration, frequency_array):
        """ Helper function to figure out if the frequency_array, or
            sampling_frequency and duration where given
        """

        self._infer_dependence(domain='frequency', array=frequency_array,
                               duration=duration, sampling_frequency=sampling_frequency, start_time=start_time)

    def _infer_dependence(self, domain, array, duration, sampling_frequency, start_time):
        if (sampling_frequency is not None) and (duration is not None):
            if array is not None:
                raise ValueError(
                    "You have given the sampling_frequency, duration, and "
                    "an array")
            pass
        elif array is not None:
            if domain == 'time':
                self.time_array = array
            elif domain == 'frequency':
                self.frequency_array = array
                self.start_time = start_time
            return
        elif sampling_frequency is None or duration is None:
            raise ValueError(
                "You must provide both sampling_frequency and duration")
        else:
            raise ValueError(
                "Insufficient information given to set arrays")
        self._times_and_frequencies = CoupledTimeAndFrequencySeries(duration=duration,
                                                                    sampling_frequency=sampling_frequency,
                                                                    start_time=start_time)

    def set_from_time_domain_strain(
            self, time_domain_strain, sampling_frequency=None, duration=None,
            start_time=0, time_array=None):
        """ Set the strain data from a time domain strain array

        This sets the time_domain_strain attribute, the frequency_domain_strain
        is automatically calculated after a low-pass filter and Tukey window
        is applied.

        Parameters
        ==========
        time_domain_strain: array_like
            An array of the time domain strain.
        sampling_frequency: float
            The sampling frequency (in Hz).
        duration: float
            The data duration (in s).
        start_time: float
            The GPS start-time of the data.
        time_array: array_like
            The array of times, if sampling_frequency and duration not
            given.

        """
        self._infer_time_domain_dependence(start_time=start_time,
                                           sampling_frequency=sampling_frequency,
                                           duration=duration,
                                           time_array=time_array)

        logger.debug('Setting data using provided time_domain_strain')
        if np.shape(time_domain_strain) == np.shape(self.time_array):
            self._time_domain_strain = time_domain_strain
            self._frequency_domain_strain = None
        else:
            raise ValueError("Data times do not match time array")

    def set_from_gwpy_timeseries(self, time_series, *, xp=np):
        """ Set the strain data from a gwpy TimeSeries

        This sets the time_domain_strain attribute, the frequency_domain_strain
        is automatically calculated after a low-pass filter and Tukey window
        is applied.

        Parameters
        ==========
        time_series: gwpy.timeseries.timeseries.TimeSeries
            The data to use
        xp: array module, optional
            The array module to use, e.g., :code:`numpy` or :code:`jax.numpy`.
            If not specified :code:`numpy` will be used.

        """
        from gwpy.timeseries import TimeSeries
        logger.debug('Setting data using provided gwpy TimeSeries object')
        if not isinstance(time_series, TimeSeries):
            raise ValueError("Input time_series is not a gwpy TimeSeries")
        duration = xp.asarray(time_series.duration.value)
        sampling_frequency = xp.asarray(time_series.sample_rate.value)
        start_time = xp.asarray(time_series.epoch.value)
        self._times_and_frequencies = \
            CoupledTimeAndFrequencySeries(duration=duration,
                                          sampling_frequency=sampling_frequency,
                                          start_time=start_time)
        self._time_domain_strain = xp.asarray(time_series.value)
        self._frequency_domain_strain = None
        self._channel = time_series.channel

    @property
    def channel(self):
        return self._channel

    def set_from_open_data(
            self, name, start_time, duration=4, outdir='outdir', cache=True,
            *, xp=None, **kwargs):
        """ Set the strain data from open LOSC data

        This sets the time_domain_strain attribute, the frequency_domain_strain
        is automatically calculated after a low-pass filter and Tukey window
        is applied.

        Parameters
        ==========
        name: str
            Detector name, e.g., 'H1'.
        start_time: float
            Start GPS time of segment.
        duration: float, optional
            The total time (in seconds) to analyse. Defaults to 4s.
        outdir: str
            Directory where the psd files are saved
        cache: bool, optional
            Whether or not to store/use the acquired data.
        xp: array module, optional
            The array module to use, e.g., :code:`numpy` or :code:`jax.numpy`.
            If not specified :code:`numpy` will be used.
        **kwargs:
            All keyword arguments are passed to
            `gwpy.timeseries.TimeSeries.fetch_open_data()`.

        """
        timeseries = gwutils.get_open_strain_data(
            name, float(start_time), float(start_time + duration), outdir=outdir, cache=cache,
            **kwargs)

        if xp is None:
            xp = array_module((duration, start_time))

        self.set_from_gwpy_timeseries(timeseries, xp=xp)

    def set_from_csv(self, filename, xp=None):
        """ Set the strain data from a csv file

        Parameters
        ==========
        filename: str
            The path to the file to read in
        xp: array module, optional
            The array module to use, e.g., :code:`numpy` or :code:`jax.numpy`.
            If not specified :code:`numpy` will be used.

        """
        from gwpy.timeseries import TimeSeries
        timeseries = TimeSeries.read(filename, format='csv')
        self.set_from_gwpy_timeseries(timeseries, xp=xp)

    def set_from_frequency_domain_strain(
            self, frequency_domain_strain, sampling_frequency=None,
            duration=None, start_time=0, frequency_array=None):
        """ Set the `frequency_domain_strain` from a numpy array

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

        self._infer_frequency_domain_dependence(start_time=start_time,
                                                sampling_frequency=sampling_frequency,
                                                duration=duration,
                                                frequency_array=frequency_array)

        logger.debug('Setting data using provided frequency_domain_strain')
        if np.shape(frequency_domain_strain) == np.shape(self.frequency_array):
            self._frequency_domain_strain = frequency_domain_strain
            self.window_factor = 1
        else:
            raise ValueError("Data frequencies do not match frequency_array")

    def set_from_power_spectral_density(
            self, power_spectral_density, sampling_frequency, duration,
            start_time=0, *, random_state=None):
        """ Set the `frequency_domain_strain` by generating a noise realisation

        Parameters
        ==========
        power_spectral_density: bilby.gw.detector.PowerSpectralDensity
            A PowerSpectralDensity object used to generate the data
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """

        self._times_and_frequencies = CoupledTimeAndFrequencySeries(duration=duration,
                                                                    sampling_frequency=sampling_frequency,
                                                                    start_time=start_time)
        logger.debug(
            'Setting data using noise realization from provided'
            'power_spectal_density')
        frequency_domain_strain, frequency_array = \
            power_spectral_density.get_noise_realisation(
                self.frequency_array.shape[0], duration, random_state=random_state)

        xp = aac.array_namespace(frequency_domain_strain)
        self._frequency_array = xp.asarray(self.frequency_array)
        self._time_array = xp.asarray(self.time_array)

        if self.duration == duration and frequency_array.shape == self.frequency_array.shape:
            self._frequency_domain_strain = frequency_domain_strain
        else:
            raise ValueError("Data frequencies do not match frequency_array")

    def set_from_zero_noise(self, sampling_frequency, duration, start_time=0):
        """ Set the `frequency_domain_strain` to zero noise

        This sets the `strain_data` to an array of (complex) zeros,  while
        also making sure the frequency and time arrays match
        the sampling frequency and duration given.

        Parameters
        ==========
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data

        """

        self._times_and_frequencies = CoupledTimeAndFrequencySeries(duration=duration,
                                                                    sampling_frequency=sampling_frequency,
                                                                    start_time=start_time)
        logger.debug('Setting zero noise data')
        xp = aac.get_namespace(self.frequency_array)
        self._frequency_domain_strain = xp.zeros_like(self.frequency_array,
                                                      dtype=complex)

    def set_from_frame_file(
            self, frame_file, sampling_frequency, duration, start_time=0,
            channel=None, buffer_time=1, *, xp=None):
        """ Set the `frequency_domain_strain` from a frame fiile

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
            If not specified, it will be inferred from the provided duration/
            sampling frequency.

        """

        self._times_and_frequencies = CoupledTimeAndFrequencySeries(
            duration=duration, sampling_frequency=sampling_frequency,
            start_time=start_time)

        logger.info('Reading data from frame file {}'.format(frame_file))
        strain = gwutils.read_frame_file(
            frame_file, start_time=start_time, end_time=start_time + duration,
            buffer_time=buffer_time, channel=channel,
            resample=sampling_frequency)

        if xp is None:
            xp = aac.get_namespace(self.frequency_array)

        self.set_from_gwpy_timeseries(strain, xp=xp)

    def set_from_channel_name(self, channel, duration, start_time, sampling_frequency, *, xp=None):
        """ Set the `frequency_domain_strain` by fetching from given channel
        using gwpy.TimesSeries.get(), which dynamically accesses either frames
        on disk, or a remote NDS2 server to find and return data. This function
        also verifies that the specified channel is given in the correct format.

        Parameters
        ==========
        channel: str
            Channel to look for using gwpy in the format `IFO:Channel`
        duration: float
            The data duration (in s)
        start_time: float
            The GPS start-time of the data
        sampling_frequency: float
            The sampling frequency (in Hz)
        xp: array module, optional
            The array module to use, e.g., :code:`numpy` or :code:`jax.numpy`.
            If not specified, it will be inferred from the provided duration/
            sampling frequency.

        """
        from gwpy.timeseries import TimeSeries
        channel_comp = channel.split(':')
        if len(channel_comp) != 2:
            raise IndexError('Channel name must have format `IFO:Channel`')

        self._times_and_frequencies = CoupledTimeAndFrequencySeries(
            duration=duration, sampling_frequency=sampling_frequency,
            start_time=start_time)

        logger.info('Fetching data using channel {}'.format(channel))
        strain = TimeSeries.get(channel, start_time, start_time + duration)
        strain = strain.resample(sampling_frequency)

        if xp is None:
            xp = aac.get_namespace(self.frequency_array)

        self.set_from_gwpy_timeseries(strain, xp=xp)


class Notch(object):
    def __init__(self, minimum_frequency, maximum_frequency):
        """ A notch object storing the maximum and minimum frequency of the notch

        Parameters
        ==========
        minimum_frequency, maximum_frequency: float
            The minimum and maximum frequency of the notch

        """

        if 0 < minimum_frequency < maximum_frequency < np.inf:
            self.minimum_frequency = minimum_frequency
            self.maximum_frequency = maximum_frequency
        else:
            msg = ("Your notch minimum_frequency {} and maximum_frequency {} are invalid"
                   .format(minimum_frequency, maximum_frequency))
            raise ValueError(msg)

    def get_idxs(self, frequency_array):
        """ Get a boolean mask for the frequencies in frequency_array in the notch

        Parameters
        ==========
        frequency_array: np.ndarray
            An array of frequencies

        Returns
        =======
        idxs: np.ndarray
            An array of booleans which are True for frequencies in the notch

        """
        lower = (frequency_array > self.minimum_frequency)
        upper = (frequency_array < self.maximum_frequency)
        return lower & upper

    def check_frequency(self, freq):
        """ Check if freq is inside the notch

        Parameters
        ==========
        freq: float
            The frequency to check

        Returns
        =======
        True/False:
            If freq inside the notch, return True, else False
        """

        if self.minimum_frequency < freq < self.maximum_frequency:
            return True
        else:
            return False


class NotchList(list):
    def __init__(self, notch_list):
        """ A list of notches

        Parameters
        ==========
        notch_list: list
            A list of length-2 tuples of the (max, min) frequency for the
            notches.

        Raises
        ======
        ValueError
            If the list is malformed.
        """

        if notch_list is not None:
            for notch in notch_list:
                if isinstance(notch, tuple) and len(notch) == 2:
                    self.append(Notch(*notch))
                else:
                    msg = "notch_list {} is malformed".format(notch_list)
                    raise ValueError(msg)

    def check_frequency(self, freq):
        """ Check if freq is inside the notch list

        Parameters
        ==========
        freq: float
            The frequency to check

        Returns
        =======
        True/False:
            If freq inside any of the notches, return True, else False
        """

        for notch in self:
            if notch.check_frequency(freq):
                return True
        return False
