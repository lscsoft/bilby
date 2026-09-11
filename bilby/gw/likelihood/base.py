
import os
import copy

import array_api_compat as aac
import attr
import numpy as np
from scipy.special import logsumexp

from ...core.likelihood import Likelihood
from ...core.utils import logger, BoundedRectBivariateSpline, create_time_series
from ...core.prior import Interped, Prior, Uniform, DeltaFunction
from ..detector import InterferometerList, get_empty_interferometer, calibration
from ..prior import BBHPriorDict, Cosmological
from ..utils import noise_weighted_inner_product, zenith_azimuth_to_ra_dec, ln_i0


class GravitationalWaveTransient(Likelihood):
    """ A gravitational-wave transient likelihood object

    This is the usual likelihood object to use for transient gravitational
    wave parameter estimation. It computes the log-likelihood in the frequency
    domain assuming a colored Gaussian noise model described by a power
    spectral density. See Thrane & Talbot (2019), arxiv.org/abs/1809.02293.

    Parameters
    ==========
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
        This uses a look up table calculated at run time.
        The distance prior is set to be a delta function at the minimum
        distance allowed in the prior being marginalised over.
    time_marginalization: bool, optional
        If true, marginalize over time in the likelihood.
        This uses a FFT to calculate the likelihood over a regularly spaced
        grid.
        In order to cover the whole space the prior is set to be uniform over
        the spacing of the array of times.
        If using time marginalisation and jitter_time is True a "jitter"
        parameter is added to the prior which modifies the position of the
        grid of times.
    phase_marginalization: bool, optional
        If true, marginalize over phase in the likelihood.
        This is done analytically using a Bessel function.
        The phase prior is set to be a delta function at phase=0.
    calibration_marginalization: bool, optional
        If true, marginalize over calibration response curves in the likelihood.
        This is done numerically over a number of calibration response curve realizations.
    priors: dict, optional
        If given, used in the distance and phase marginalization.
        Warning: when using marginalisation the dict is overwritten which will change the
        the dict you are passing in. If this behaviour is undesired, pass `priors.copy()`.
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table.
        If a string the name of a file containing these quantities.
        The lookup table is stored after construction in either the
        provided string or a default location:
        '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'
    calibration_lookup_table: dict, optional
        If a dict, contains the arrays over which to marginalize for each interferometer or the filepaths of the
        calibration files.
        If not provided, but calibration_marginalization is used, then the appropriate file is created to
        contain the curves.
    number_of_response_curves: int, optional
        Number of curves from the calibration lookup table to use.
        Default is 1000.
    starting_index: int, optional
        Sets the index for the first realization of the calibration curve to be considered.
        This, coupled with number_of_response_curves, allows for restricting the set of curves used. This can be used
        when dealing with large frequency arrays to split the calculation into sections.
        Defaults to 0.
    jitter_time: bool, optional
        Whether to introduce a `time_jitter` parameter. This avoids either
        missing the likelihood peak, or introducing biases in the
        reconstructed time posterior due to an insufficient sampling frequency.
        Default is :code:`True` if :code:`time_marginalization` is :code:`True`.
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.

        - :code:`sky`: sample in RA/dec, this is the default
        - e.g., :code:`"H1L1", ["H1", "L1"], InterferometerList(["H1", "L1"])`:
          sample in azimuth and zenith, `azimuth` and `zenith` defined in the
          frame where the z-axis is aligned the the vector connecting H1
          and L1.

    time_reference: str, optional
        Name of the reference for the sampled time parameter.

        - :code:`geocent`/:code:`geocenter`: sample in the time at the
          Earth's center, this is the default
        - e.g., :code:`H1`: sample in the time of arrival at H1

    Returns
    =======
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters

    """

    @attr.s(slots=True, weakref_slot=False)
    class _CalculatedSNRs:
        # the complex converted breaks JAX compilation
        # d_inner_h = attr.ib(default=0j, converter=complex)
        # optimal_snr_squared = attr.ib(default=0, converter=float)
        # complex_matched_filter_snr = attr.ib(default=0j, converter=complex)
        d_inner_h = attr.ib(default=0j)
        optimal_snr_squared = attr.ib(default=0)
        complex_matched_filter_snr = attr.ib(default=0j)
        d_inner_h_array = attr.ib(default=None)
        optimal_snr_squared_array = attr.ib(default=None)

        def __add__(self, other_snr):
            new = copy.deepcopy(self)
            new += other_snr
            return new

        def __iadd__(self, other_snr):
            for key in self.__slots__:
                this = getattr(self, key)
                other = getattr(other_snr, key)
                if this is not None and other is not None:
                    setattr(self, key, this + other)
                elif this is None:
                    setattr(self, key, other)
            return self

        @property
        def snrs_as_sample(self) -> dict:
            """Get the SNRs of this object as a sample dictionary

            Returns
            =======
            dict
                The dictionary of SNRs labelled accordingly
            """
            return {
                "matched_filter_snr" : self.complex_matched_filter_snr,
                "optimal_snr" : self.optimal_snr_squared.real ** 0.5
            }

    def __init__(
            self, interferometers, waveform_generator, time_marginalization=False,
            distance_marginalization=False, phase_marginalization=False, calibration_marginalization=False, priors=None,
            distance_marginalization_lookup_table=None, calibration_lookup_table=None,
            number_of_response_curves=1000, starting_index=0, jitter_time=True, reference_frame="sky",
            time_reference="geocenter"
    ):

        self.waveform_generator = waveform_generator
        super(GravitationalWaveTransient, self).__init__()
        self.interferometers = InterferometerList(interferometers)
        self.interferometers.set_array_backend(interferometers.array_backend)
        self.time_marginalization = time_marginalization
        self.distance_marginalization = distance_marginalization
        self.phase_marginalization = phase_marginalization
        self.calibration_marginalization = calibration_marginalization
        self.priors = priors
        self._check_set_duration_and_sampling_frequency_of_waveform_generator()
        self._noise_log_likelihood_value = None
        self.jitter_time = jitter_time
        self.reference_frame = reference_frame
        if "geocent" not in time_reference:
            self.time_reference = time_reference
            self.reference_ifo = get_empty_interferometer(self.time_reference)
            self.reference_ifo.set_array_backend(self.interferometers.array_backend)
            if self.time_marginalization:
                logger.info("Cannot marginalise over non-geocenter time.")
                self.time_marginalization = False
                self.jitter_time = False
        else:
            self.time_reference = "geocent"
            self.reference_ifo = None

        if self.time_marginalization:
            self._check_marginalized_prior_is_set(key='geocent_time')
            self._setup_time_marginalization()
            priors['geocent_time'] = float(self.interferometers.start_time)
            if self.jitter_time:
                priors['time_jitter'] = Uniform(
                    minimum=- self._delta_tc / 2,
                    maximum=self._delta_tc / 2,
                    boundary='periodic',
                    name="time_jitter",
                    latex_label="$t_j$"
                )
            self._marginalized_parameters.append('geocent_time')
        elif self.jitter_time:
            logger.debug(
                "Time jittering requested with non-time-marginalised "
                "likelihood, ignoring.")
            self.jitter_time = False

        if self.phase_marginalization:
            self._check_marginalized_prior_is_set(key='phase')
            priors['phase'] = float(0)
            self._marginalized_parameters.append('phase')

        if self.distance_marginalization:
            self._lookup_table_filename = None
            self._check_marginalized_prior_is_set(key='luminosity_distance')
            self._distance_array = np.linspace(
                self.priors['luminosity_distance'].minimum,
                self.priors['luminosity_distance'].maximum, int(1e4))
            self.distance_prior_array = np.array(
                [self.priors['luminosity_distance'].prob(distance)
                 for distance in self._distance_array])
            self._ref_dist = self.priors['luminosity_distance'].rescale(0.5)
            self._setup_distance_marginalization(
                distance_marginalization_lookup_table)
            for key in ['redshift', 'comoving_distance']:
                if key in priors:
                    del priors[key]
            priors['luminosity_distance'] = float(self._ref_dist)
            self._marginalized_parameters.append('luminosity_distance')

        if self.calibration_marginalization:
            self.number_of_response_curves = number_of_response_curves
            self.starting_index = starting_index
            self._setup_calibration_marginalization(calibration_lookup_table, priors)
            self._marginalized_parameters.append('recalib_index')

    def __repr__(self):
        return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={},\n\ttime_marginalization={}, ' \
                                         'distance_marginalization={}, phase_marginalization={}, ' \
                                         'calibration_marginalization={}, priors={})' \
            .format(self.interferometers, self.waveform_generator, self.time_marginalization,
                    self.distance_marginalization, self.phase_marginalization, self.calibration_marginalization,
                    self.priors)

    def _check_set_duration_and_sampling_frequency_of_waveform_generator(self):
        """ Check the waveform_generator has the same duration and
        sampling_frequency as the interferometers. If they are unset, then
        set them, if they differ, raise an error
        """

        attributes = ['duration', 'sampling_frequency', 'start_time']
        for attribute in attributes:
            wfg_attr = getattr(self.waveform_generator, attribute)
            ifo_attr = getattr(self.interferometers, attribute)
            if wfg_attr is None:
                logger.debug(
                    "The waveform_generator {} is None. Setting from the "
                    "provided interferometers.".format(attribute))
            elif wfg_attr != ifo_attr:
                logger.debug(
                    "The waveform_generator {} is not equal to that of the "
                    "provided interferometers. Overwriting the "
                    "waveform_generator.".format(attribute))
            setattr(self.waveform_generator, attribute, ifo_attr)

    def calculate_snrs(self, waveform_polarizations, interferometer, *, return_array=True, parameters):
        """
        Compute the snrs

        Parameters
        ----------
        waveform_polarizations: dict
            A dictionary of waveform polarizations and the corresponding array
        interferometer: bilby.gw.detector.Interferometer
            The bilby interferometer object
        return_array: bool
            If true, calculate and return internal array objects
            (d_inner_h_array and optimal_snr_squared_array), otherwise
            these are returned as None.

        Returns
        -------
        calculated_snrs: _CalculatedSNRs
            An object containing the SNR quantities and (if return_array=True)
            the internal array objects.

        """
        signal = self._compute_full_waveform(
            signal_polarizations=waveform_polarizations,
            interferometer=interferometer,
            parameters=parameters,
        )
        _mask = interferometer.frequency_mask

        if 'recalib_index' in parameters:
            signal[_mask] *= self.calibration_draws[interferometer.name][int(parameters['recalib_index'])]

        d_inner_h = interferometer.inner_product(signal=signal)
        optimal_snr_squared = interferometer.optimal_snr_squared(signal=signal)
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        d_inner_h_array = None
        optimal_snr_squared_array = None

        normalization = 4 / self.waveform_generator.duration
        xp = aac.array_namespace(signal)

        if return_array is False:
            d_inner_h_array = None
            optimal_snr_squared_array = None
        elif self.time_marginalization and self.calibration_marginalization:

            d_inner_h_integrand = xp.tile(
                interferometer.frequency_domain_strain.conj() * signal /
                interferometer.power_spectral_density_array, (self.number_of_response_curves, 1)).T

            d_inner_h_integrand[_mask] *= self.calibration_draws[interferometer.name].T

            d_inner_h_array = 4 / self.waveform_generator.duration * xp.fft.fft(
                d_inner_h_integrand[0:-1], axis=0
            ).T

            optimal_snr_squared_integrand = (
                normalization * xp.abs(signal)**2 / interferometer.power_spectral_density_array
            )
            optimal_snr_squared_array = xp.dot(
                optimal_snr_squared_integrand[_mask],
                self.calibration_abs_draws[interferometer.name].T
            )

        elif self.time_marginalization and not self.calibration_marginalization:
            d_inner_h_array = normalization * xp.fft.fft(
                signal[0:-1]
                * interferometer.frequency_domain_strain.conj()[0:-1]
                / interferometer.power_spectral_density_array[0:-1]
            )

        elif self.calibration_marginalization and ('recalib_index' not in parameters):
            d_inner_h_integrand = (
                normalization *
                interferometer.frequency_domain_strain.conj() * signal
                / interferometer.power_spectral_density_array
            )
            d_inner_h_array = xp.dot(d_inner_h_integrand[_mask], self.calibration_draws[interferometer.name].T)

            optimal_snr_squared_integrand = (
                normalization * xp.abs(signal)**2 / interferometer.power_spectral_density_array
            )
            optimal_snr_squared_array = xp.dot(
                optimal_snr_squared_integrand[_mask],
                self.calibration_abs_draws[interferometer.name].T
            )

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared.real,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array,
            optimal_snr_squared_array=optimal_snr_squared_array,
        )

    def _check_marginalized_prior_is_set(self, key):
        if key in self.priors and self.priors[key].is_fixed:
            raise ValueError(
                "Cannot use marginalized likelihood for {}: prior is fixed".format(key)
            )
        if key not in self.priors or not isinstance(
                self.priors[key], Prior):
            if key == 'geocent_time':
                logger.warning(
                    'Prior not provided for geocent time, using the full segment.'
                )
                self.priors[key] = Uniform(
                    self.interferometers.start_time,
                    self.interferometers.start_time + self.interferometers.duration)
            elif key == 'luminosity_distance':
                for key in ['redshift', 'comoving_distance']:
                    if key in self.priors:
                        if not isinstance(self.priors[key], Cosmological):
                            raise TypeError(
                                "To marginalize over {}, the prior must be specified as a "
                                "subclass of bilby.gw.prior.Cosmological.".format(key)
                            )
                        self.priors['luminosity_distance'] = self.priors[key].get_corresponding_prior(
                            'luminosity_distance'
                        )
                        del self.priors[key]
            else:
                logger.warning(
                    'Prior not provided for {}, using the BBH default.'.format(key)
                )
                self.priors[key] = BBHPriorDict()[key]

    @property
    def priors(self):
        return self._prior

    @priors.setter
    def priors(self, priors):
        if priors is not None:
            self._prior = priors.copy()
        elif any([self.time_marginalization, self.phase_marginalization,
                  self.distance_marginalization]):
            raise ValueError("You can't use a marginalized likelihood without specifying a priors")
        else:
            self._prior = None

    def _calculate_noise_log_likelihood(self):
        log_l = 0
        for interferometer in self.interferometers:
            mask = interferometer.frequency_mask
            log_l -= abs(noise_weighted_inner_product(
                interferometer.frequency_domain_strain[mask],
                interferometer.frequency_domain_strain[mask],
                interferometer.power_spectral_density_array[mask],
                self.waveform_generator.duration) / 2)
        return log_l

    def noise_log_likelihood(self):
        # only compute likelihood if called for the 1st time
        if self._noise_log_likelihood_value is None:
            self._noise_log_likelihood_value = self._calculate_noise_log_likelihood()
        return self._noise_log_likelihood_value

    def log_likelihood_ratio(self, parameters):
        parameters = copy.deepcopy(parameters)
        parameters.update(self.get_sky_frame_parameters(parameters))
        waveform_polarizations = \
            self.waveform_generator.frequency_domain_strain(parameters)
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        if self.time_marginalization and self.jitter_time:
            parameters['geocent_time'] += parameters['time_jitter']

        total_snrs = self._CalculatedSNRs()

        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )

            total_snrs += per_detector_snr

        log_l = self.compute_log_likelihood_from_snrs(total_snrs, parameters=parameters)

        if self.time_marginalization and self.jitter_time:
            parameters['geocent_time'] -= parameters['time_jitter']

        return log_l.real

    def compute_log_likelihood_from_snrs(self, total_snrs, parameters):

        if self.calibration_marginalization:
            log_l = self.calibration_marginalized_likelihood(
                d_inner_h_calibration_array=total_snrs.d_inner_h_array,
                h_inner_h=total_snrs.optimal_snr_squared_array,
                parameters=parameters,
            )

        elif self.time_marginalization:
            log_l = self.time_marginalized_likelihood(
                d_inner_h_tc_array=total_snrs.d_inner_h_array,
                h_inner_h=total_snrs.optimal_snr_squared,
                parameters=parameters,
            )

        elif self.distance_marginalization:
            log_l = self.distance_marginalized_likelihood(
                d_inner_h=total_snrs.d_inner_h, h_inner_h=total_snrs.optimal_snr_squared,
                parameters=parameters,
            )

        elif self.phase_marginalization:
            log_l = self.phase_marginalized_likelihood(
                d_inner_h=total_snrs.d_inner_h, h_inner_h=total_snrs.optimal_snr_squared)

        else:
            log_l = np.real(total_snrs.d_inner_h) - total_snrs.optimal_snr_squared / 2

        return log_l

    def compute_per_detector_log_likelihood(self, parameters):
        parameters.update(self.get_sky_frame_parameters(parameters))
        waveform_polarizations = \
            self.waveform_generator.frequency_domain_strain(parameters)

        if self.time_marginalization and self.jitter_time:
            parameters['geocent_time'] += parameters['time_jitter']

        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )

            parameters['{}_log_likelihood'.format(interferometer.name)] = \
                self.compute_log_likelihood_from_snrs(per_detector_snr, parameters=parameters)

        if self.time_marginalization and self.jitter_time:
            parameters['geocent_time'] -= parameters['time_jitter']

        return parameters.copy()

    def generate_posterior_sample_from_marginalized_likelihood(self, parameters):
        """
        Reconstruct the distance posterior from a run which used a likelihood
        which explicitly marginalised over time/distance/phase.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Returns
        =======
        sample: dict
            Returns the parameters with new samples.

        Notes
        =====
        This involves a deepcopy of the signal to avoid issues with waveform
        caching, as the signal is overwritten in place.
        """
        if len(self._marginalized_parameters) > 0:
            signal_polarizations = copy.deepcopy(
                self.waveform_generator.frequency_domain_strain(
                    parameters))
        else:
            return parameters

        if self.calibration_marginalization:
            new_calibration = self.generate_calibration_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations, parameters=parameters)
            parameters['recalib_index'] = new_calibration
        if self.time_marginalization:
            new_time = self.generate_time_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations, parameters=parameters)
            parameters['geocent_time'] = new_time
        if self.distance_marginalization:
            new_distance = self.generate_distance_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations, parameters=parameters)
            parameters['luminosity_distance'] = new_distance
        if self.phase_marginalization:
            new_phase = self.generate_phase_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations, parameters=parameters)
            parameters['phase'] = new_phase
        return parameters.copy()

    def generate_calibration_sample_from_marginalized_likelihood(
            self, signal_polarizations=None, *, parameters):
        """
        Generate a single sample from the posterior distribution for the set of calibration response curves when
        explicitly marginalizing over the calibration uncertainty.

        Parameters
        ----------
        signal_polarizations: dict, optional
            Polarizations modes of the template.

        Returns
        -------
        new_calibration: dict
            Sample set from the calibration posterior
        """
        from ...core.utils import random

        if 'recalib_index' in parameters:
            parameters.pop('recalib_index')
        parameters.update(self.get_sky_frame_parameters(parameters))
        if signal_polarizations is None:
            signal_polarizations = \
                self.waveform_generator.frequency_domain_strain(parameters)

        log_like = self.get_calibration_log_likelihoods(
            signal_polarizations=signal_polarizations, parameters=parameters
        )

        calibration_post = np.exp(log_like - max(log_like))
        calibration_post /= np.sum(calibration_post)

        new_calibration = random.rng.choice(self.number_of_response_curves, p=calibration_post)

        return new_calibration

    def generate_time_sample_from_marginalized_likelihood(
            self, signal_polarizations=None, *, parameters):
        """
        Generate a single sample from the posterior distribution for coalescence
        time when using a likelihood which explicitly marginalises over time.

        In order to resolve the posterior we artificially upsample to 16kHz.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Parameters
        ==========
        signal_polarizations: dict, optional
            Polarizations modes of the template.

        Returns
        =======
        new_time: float
            Sample from the time posterior.
        """
        parameters.update(self.get_sky_frame_parameters(parameters))
        if self.jitter_time:
            parameters['geocent_time'] += parameters['time_jitter']
        if signal_polarizations is None:
            signal_polarizations = \
                self.waveform_generator.frequency_domain_strain(parameters)

        times = create_time_series(
            sampling_frequency=16384,
            starting_time=parameters['geocent_time'] - self.waveform_generator.start_time,
            duration=self.waveform_generator.duration)
        times = times % self.waveform_generator.duration
        times += self.waveform_generator.start_time

        prior = self.priors["geocent_time"]
        in_prior = (times >= prior.minimum) & (times < prior.maximum)
        times = times[in_prior]

        n_time_steps = int(self.waveform_generator.duration * 16384)
        d_inner_h = np.zeros(len(times), dtype=complex)
        psd = np.ones(n_time_steps)
        signal_long = np.zeros(n_time_steps, dtype=complex)
        data = np.zeros(n_time_steps, dtype=complex)
        h_inner_h = np.zeros(1)
        for ifo in self.interferometers:
            ifo_length = len(ifo.frequency_domain_strain)
            mask = ifo.frequency_mask
            signal = self._compute_full_waveform(
                signal_polarizations=signal_polarizations,
                interferometer=ifo,
                parameters=parameters,
            )
            signal_long[:ifo_length] = signal
            data[:ifo_length] = np.conj(ifo.frequency_domain_strain)
            psd[:ifo_length][mask] = ifo.power_spectral_density_array[mask]
            d_inner_h += np.fft.fft(signal_long * data / psd)[in_prior]
            h_inner_h += ifo.optimal_snr_squared(signal=signal).real

        if self.distance_marginalization:
            time_log_like = self.distance_marginalized_likelihood(
                d_inner_h, h_inner_h, parameters=parameters)
        elif self.phase_marginalization:
            time_log_like = ln_i0(abs(d_inner_h)) - h_inner_h.real / 2
        else:
            time_log_like = (d_inner_h.real - h_inner_h.real / 2)

        time_prior_array = self.priors['geocent_time'].prob(times)
        time_post = np.exp(time_log_like - max(time_log_like)) * time_prior_array

        keep = (time_post > max(time_post) / 1000)
        if sum(keep) < 3:
            keep[1:-1] = keep[1:-1] | keep[2:] | keep[:-2]
        time_post = time_post[keep]
        times = times[keep]

        new_time = Interped(times, time_post).sample()
        return new_time

    def generate_distance_sample_from_marginalized_likelihood(
            self, signal_polarizations=None, *, parameters):
        """
        Generate a single sample from the posterior distribution for luminosity
        distance when using a likelihood which explicitly marginalises over
        distance.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Parameters
        ==========
        signal_polarizations: dict, optional
            Polarizations modes of the template.
            Note: These are rescaled in place after the distance sample is
            generated to allow further parameter reconstruction to occur.

        Returns
        =======
        new_distance: float
            Sample from the distance posterior.
        """
        parameters.update(self.get_sky_frame_parameters(parameters))
        if signal_polarizations is None:
            signal_polarizations = \
                self.waveform_generator.frequency_domain_strain(parameters)

        d_inner_h, h_inner_h = self._calculate_inner_products(
            signal_polarizations, parameters=parameters
        )

        d_inner_h_dist = (
            d_inner_h * parameters['luminosity_distance'] / self._distance_array
        )

        h_inner_h_dist = (
            h_inner_h * parameters['luminosity_distance']**2 / self._distance_array**2
        )

        if self.phase_marginalization:
            distance_log_like = ln_i0(abs(d_inner_h_dist)) - h_inner_h_dist.real / 2
        else:
            distance_log_like = (d_inner_h_dist.real - h_inner_h_dist.real / 2)

        distance_post = (np.exp(distance_log_like - max(distance_log_like)) *
                         self.distance_prior_array)

        new_distance = Interped(
            self._distance_array, distance_post).sample()

        self._rescale_signal(signal_polarizations, new_distance)
        return new_distance

    def _calculate_inner_products(self, signal_polarizations, parameters):
        d_inner_h = 0
        h_inner_h = 0
        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                signal_polarizations, interferometer, parameters=parameters)

            d_inner_h += per_detector_snr.d_inner_h
            h_inner_h += per_detector_snr.optimal_snr_squared
        return d_inner_h, h_inner_h

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters):
        """
        Project the waveform polarizations against the interferometer
        response. This is useful for likelihood classes that don't
        use the full frequency array, e.g., the relative binning
        likelihood.

        Parameters
        ==========
        signal_polarizations: dict
            Dictionary containing the waveform evaluated at
            :code:`interferometer.frequency_array`.
        interferometer: bilby.gw.detector.Interferometer
            Interferometer to compute the response with respect to.
        """
        return interferometer.get_detector_response(signal_polarizations, parameters)

    def generate_phase_sample_from_marginalized_likelihood(
            self, signal_polarizations=None, *, parameters):
        r"""
        Generate a single sample from the posterior distribution for phase when
        using a likelihood which explicitly marginalises over phase.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Parameters
        ==========
        signal_polarizations: dict, optional
            Polarizations modes of the template.

        Returns
        =======
        new_phase: float
            Sample from the phase posterior.

        Notes
        =====
        This is only valid when assumes that mu(phi) \propto exp(-2i phi).
        """
        parameters.update(self.get_sky_frame_parameters(parameters))
        if signal_polarizations is None:
            signal_polarizations = \
                self.waveform_generator.frequency_domain_strain(parameters)
        d_inner_h, h_inner_h = self._calculate_inner_products(
            signal_polarizations, parameters=parameters
        )

        phases = np.linspace(0, 2 * np.pi, 101)
        phasor = np.exp(-2j * phases)
        phase_log_post = d_inner_h * phasor - h_inner_h / 2
        phase_post = np.exp(phase_log_post.real - max(phase_log_post.real))
        new_phase = Interped(phases, phase_post).sample()
        return new_phase

    def distance_marginalized_likelihood(self, d_inner_h, h_inner_h, *, parameters):
        d_inner_h_ref, h_inner_h_ref = self._setup_rho(
            d_inner_h, h_inner_h, parameters=parameters)
        if self.phase_marginalization:
            d_inner_h_ref = abs(d_inner_h_ref)
        else:
            d_inner_h_ref = d_inner_h_ref.real

        return self._interp_dist_margd_loglikelihood(
            d_inner_h_ref, h_inner_h_ref)

    def phase_marginalized_likelihood(self, d_inner_h, h_inner_h):
        d_inner_h = ln_i0(abs(d_inner_h))

        if self.calibration_marginalization and self.time_marginalization:
            return d_inner_h - h_inner_h[:, np.newaxis] / 2
        else:
            return d_inner_h - h_inner_h / 2

    def time_marginalized_likelihood(self, d_inner_h_tc_array, h_inner_h, *, parameters):
        times = self._times
        if self.jitter_time:
            times = self._times + parameters['time_jitter']

        if not aac.is_jax_array(d_inner_h_tc_array):
            _time_prior = self.priors['geocent_time']
            time_mask = (times >= _time_prior.minimum) & (times <= _time_prior.maximum)
            times = times[time_mask]
            if self.calibration_marginalization:
                d_inner_h_tc_array = d_inner_h_tc_array[:, time_mask]
            else:
                d_inner_h_tc_array = d_inner_h_tc_array[time_mask]
        time_prior_array = self.priors['geocent_time'].prob(times) * self._delta_tc

        if self.distance_marginalization:
            log_l_tc_array = self.distance_marginalized_likelihood(
                d_inner_h=d_inner_h_tc_array, h_inner_h=h_inner_h, parameters=parameters)
        elif self.phase_marginalization:
            log_l_tc_array = self.phase_marginalized_likelihood(
                d_inner_h=d_inner_h_tc_array,
                h_inner_h=h_inner_h)
        elif self.calibration_marginalization:
            log_l_tc_array = d_inner_h_tc_array.real - h_inner_h[:, np.newaxis] / 2
        else:
            log_l_tc_array = d_inner_h_tc_array.real - h_inner_h / 2
        return logsumexp(log_l_tc_array, b=time_prior_array, axis=-1)

    def get_calibration_log_likelihoods(self, signal_polarizations=None, *, parameters):
        parameters.update(self.get_sky_frame_parameters(parameters))
        if signal_polarizations is None:
            signal_polarizations = \
                self.waveform_generator.frequency_domain_strain(parameters)

        total_snrs = self._CalculatedSNRs()

        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                waveform_polarizations=signal_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )
            total_snrs += per_detector_snr

        if self.time_marginalization:
            log_l_cal_array = self.time_marginalized_likelihood(
                d_inner_h_tc_array=total_snrs.d_inner_h_array,
                h_inner_h=total_snrs.optimal_snr_squared_array,
                parameters=parameters,
            )
        elif self.distance_marginalization:
            log_l_cal_array = self.distance_marginalized_likelihood(
                d_inner_h=total_snrs.d_inner_h_array,
                h_inner_h=total_snrs.optimal_snr_squared_array,
                parameters=parameters,
            )
        elif self.phase_marginalization:
            log_l_cal_array = self.phase_marginalized_likelihood(
                d_inner_h=total_snrs.d_inner_h_array,
                h_inner_h=total_snrs.optimal_snr_squared_array)
        else:
            log_l_cal_array = \
                np.real(total_snrs.d_inner_h_array - total_snrs.optimal_snr_squared_array / 2)

        return log_l_cal_array

    def calibration_marginalized_likelihood(self, d_inner_h_calibration_array, h_inner_h, *, parameters):
        if self.time_marginalization:
            log_l_cal_array = self.time_marginalized_likelihood(
                d_inner_h_tc_array=d_inner_h_calibration_array,
                h_inner_h=h_inner_h,
                parameters=parameters,
            )
        elif self.distance_marginalization:
            log_l_cal_array = self.distance_marginalized_likelihood(
                d_inner_h=d_inner_h_calibration_array, h_inner_h=h_inner_h, parameters=parameters)
        elif self.phase_marginalization:
            log_l_cal_array = self.phase_marginalized_likelihood(
                d_inner_h=d_inner_h_calibration_array,
                h_inner_h=h_inner_h)
        else:
            log_l_cal_array = np.real(d_inner_h_calibration_array - h_inner_h / 2)

        return logsumexp(log_l_cal_array) - np.log(self.number_of_response_curves)

    def _setup_rho(self, d_inner_h, optimal_snr_squared, parameters):
        optimal_snr_squared_ref = (optimal_snr_squared.real *
                                   parameters['luminosity_distance'] ** 2 /
                                   self._ref_dist ** 2.)
        d_inner_h_ref = (d_inner_h * parameters['luminosity_distance'] /
                         self._ref_dist)
        return d_inner_h_ref, optimal_snr_squared_ref

    def log_likelihood(self, parameters):
        return self.log_likelihood_ratio(parameters=parameters) + self.noise_log_likelihood()

    @property
    def _delta_distance(self):
        return self._distance_array[1] - self._distance_array[0]

    @property
    def _dist_multiplier(self):
        ''' Maximum value of ref_dist/dist_array '''
        return self._ref_dist / self._distance_array[0]

    @property
    def _optimal_snr_squared_ref_array(self):
        """ Optimal filter snr at fiducial distance of ref_dist Mpc """
        return np.logspace(-5, 10, self._dist_margd_loglikelihood_array.shape[0])

    @property
    def _d_inner_h_ref_array(self):
        """ Matched filter snr at fiducial distance of ref_dist Mpc """
        if self.phase_marginalization:
            return np.logspace(-5, 10, self._dist_margd_loglikelihood_array.shape[1])
        else:
            n_negative = self._dist_margd_loglikelihood_array.shape[1] // 2
            n_positive = self._dist_margd_loglikelihood_array.shape[1] - n_negative
            return np.hstack((
                -np.logspace(3, -3, n_negative), np.logspace(-3, 10, n_positive)
            ))

    def _setup_distance_marginalization(self, lookup_table=None):
        if isinstance(lookup_table, str) or lookup_table is None:
            self.cached_lookup_table_filename = lookup_table
            lookup_table = self.load_lookup_table(
                self.cached_lookup_table_filename)
        if isinstance(lookup_table, dict):
            if self._test_cached_lookup_table(lookup_table):
                self._dist_margd_loglikelihood_array = lookup_table[
                    'lookup_table']
            else:
                self._create_lookup_table()
        else:
            self._create_lookup_table()
        self._interp_dist_margd_loglikelihood = BoundedRectBivariateSpline(
            self._d_inner_h_ref_array,
            self._optimal_snr_squared_ref_array,
            self._dist_margd_loglikelihood_array.T,
            fill_value=-np.inf,
        )

    @property
    def cached_lookup_table_filename(self):
        if self._lookup_table_filename is None:
            self._lookup_table_filename = (
                '.distance_marginalization_lookup.npz')
        return self._lookup_table_filename

    @cached_lookup_table_filename.setter
    def cached_lookup_table_filename(self, filename):
        if isinstance(filename, str):
            if filename[-4:] != '.npz':
                filename += '.npz'
        self._lookup_table_filename = filename

    def load_lookup_table(self, filename):
        if os.path.exists(filename):
            try:
                loaded_file = dict(np.load(filename))
            except AttributeError as e:
                logger.warning(e)
                self._create_lookup_table()
                return None
            match, failure = self._test_cached_lookup_table(loaded_file)
            if match:
                logger.info('Loaded distance marginalisation lookup table from '
                            '{}.'.format(filename))
                return loaded_file
            else:
                logger.info('Loaded distance marginalisation lookup table does '
                            'not match for {}.'.format(failure))
        elif isinstance(filename, str):
            logger.info('Distance marginalisation file {} does not '
                        'exist'.format(filename))
        return None

    def cache_lookup_table(self):
        np.savez(self.cached_lookup_table_filename,
                 distance_array=self._distance_array,
                 prior_array=self.distance_prior_array,
                 lookup_table=self._dist_margd_loglikelihood_array,
                 reference_distance=self._ref_dist,
                 phase_marginalization=self.phase_marginalization)

    def _test_cached_lookup_table(self, loaded_file):
        pairs = dict(
            distance_array=self._distance_array,
            prior_array=self.distance_prior_array,
            reference_distance=self._ref_dist,
            phase_marginalization=self.phase_marginalization)
        for key in pairs:
            if key not in loaded_file:
                return False, key
            elif not np.allclose(np.atleast_1d(loaded_file[key]),
                                 np.atleast_1d(pairs[key]),
                                 rtol=1e-15):
                return False, key
        return True, None

    def _create_lookup_table(self):
        """ Make the lookup table """
        from tqdm.auto import tqdm
        logger.info('Building lookup table for distance marginalisation.')

        self._dist_margd_loglikelihood_array = np.zeros((400, 800))
        scaling = self._ref_dist / self._distance_array
        d_inner_h_array_full = np.outer(self._d_inner_h_ref_array, scaling)
        h_inner_h_array_full = np.outer(self._optimal_snr_squared_ref_array, scaling ** 2)
        if self.phase_marginalization:
            d_inner_h_array_full = ln_i0(abs(d_inner_h_array_full))
        prior_term = self.distance_prior_array * self._delta_distance
        for ii, optimal_snr_squared_array in tqdm(
                enumerate(h_inner_h_array_full), total=len(self._optimal_snr_squared_ref_array)
        ):
            for jj, d_inner_h_array in enumerate(d_inner_h_array_full):
                self._dist_margd_loglikelihood_array[ii][jj] = logsumexp(
                    d_inner_h_array - optimal_snr_squared_array / 2,
                    b=prior_term
                )
        log_norm = logsumexp(
            0 / self._distance_array, b=self.distance_prior_array * self._delta_distance
        )
        self._dist_margd_loglikelihood_array -= log_norm
        self.cache_lookup_table()

    def _setup_phase_marginalization(self, min_bound=-5, max_bound=10):
        logger.warning(
            "The _setup_phase_marginalization method is deprecated and will be removed, "
            "please update the implementation of phase marginalization "
            "to use bilby.gw.utils.ln_i0"
        )

    def _setup_time_marginalization(self):
        self._delta_tc = 2 / self.waveform_generator.sampling_frequency
        self._times = \
            self.interferometers.start_time + np.linspace(
                0, self.interferometers.duration,
                int(self.interferometers.duration / 2 *
                    self.waveform_generator.sampling_frequency + 1))[1:]
        self.time_prior_array = \
            self.priors['geocent_time'].prob(self._times) * self._delta_tc

    def _setup_calibration_marginalization(self, calibration_lookup_table, priors=None):
        self.calibration_draws, self.calibration_parameter_draws = calibration.build_calibration_lookup(
            interferometers=self.interferometers,
            lookup_files=calibration_lookup_table,
            priors=priors,
            number_of_response_curves=self.number_of_response_curves,
            starting_index=self.starting_index,
        )
        for name, parameters in self.calibration_parameter_draws.items():
            if parameters is not None:
                for key in set(parameters.keys()).intersection(priors.keys()):
                    priors[key] = DeltaFunction(0.0)
        self.calibration_abs_draws = dict()
        for name in self.calibration_draws:
            self.calibration_abs_draws[name] = np.abs(self.calibration_draws[name])**2

    @property
    def interferometers(self):
        return self._interferometers

    @interferometers.setter
    def interferometers(self, interferometers):
        self._interferometers = InterferometerList(interferometers)

    def _rescale_signal(self, signal, new_distance):
        for mode in signal:
            signal[mode] *= self._ref_dist / new_distance

    @property
    def reference_frame(self):
        return self._reference_frame

    @property
    def _reference_frame_str(self):
        if isinstance(self.reference_frame, str):
            return self.reference_frame
        elif self.reference_frame is None:
            return "sky"
        else:
            return "".join([ifo.name for ifo in self.reference_frame])

    @reference_frame.setter
    def reference_frame(self, frame):
        if frame == "sky":
            self._reference_frame = None
        elif isinstance(frame, InterferometerList):
            self._reference_frame = frame[:2]
        elif isinstance(frame, list):
            self._reference_frame = InterferometerList(frame[:2])
        elif isinstance(frame, str):
            self._reference_frame = InterferometerList([frame[:2], frame[2:4]])
        else:
            raise ValueError("Unable to parse reference frame {}".format(frame))
        if isinstance(self._reference_frame, InterferometerList):
            self._reference_frame.set_array_backend(self.interferometers.array_backend)

    def get_sky_frame_parameters(self, parameters):
        """
        Generate ra, dec, and geocenter time for :code:`parameters`

        This method will attempt to convert from the reference time and sky
        parameters, but if they are not present it will fall back to ra and dec.

        Parameters
        ==========
        parameters: dict, optional
            The parameters to be converted.

        Returns
        =======
        dict: dictionary containing ra, dec, and geocent_time
        """
        time = parameters.get(f'{self.time_reference}_time', None)
        if time is None and "geocent_time" in parameters:
            logger.warning(
                f"Cannot find {self.time_reference}_time in parameters. "
                "Falling back to geocent time"
            )
        if self.reference_frame is not None:
            try:
                ra, dec = zenith_azimuth_to_ra_dec(
                    parameters['zenith'], parameters['azimuth'],
                    time, self.reference_frame)
            except KeyError:
                if "ra" in parameters and "dec" in parameters:
                    ra = parameters["ra"]
                    dec = parameters["dec"]
                    logger.warning(
                        "Cannot convert from zenith/azimuth to ra/dec falling "
                        "back to provided ra/dec"
                    )
                else:
                    raise
        else:
            ra = parameters["ra"]
            dec = parameters["dec"]
        if "geocent" not in self.time_reference and f"{self.time_reference}_time" in parameters:
            geocent_time = time - self.reference_ifo.time_delay_from_geocenter(
                ra=ra, dec=dec, time=time
            )
        else:
            geocent_time = parameters["geocent_time"]
        return dict(ra=ra, dec=dec, geocent_time=geocent_time)

    @property
    def lal_version(self):
        try:
            from lal import git_version, __version__
            lal_version = str(__version__)
            logger.info("Using lal version {}".format(lal_version))
            lal_git_version = str(git_version.verbose_msg).replace("\n", ";")
            logger.info("Using lal git version {}".format(lal_git_version))
            return "lal_version={}, lal_git_version={}".format(lal_version, lal_git_version)
        except (ImportError, AttributeError):
            return "N/A"

    @property
    def lalsimulation_version(self):
        try:
            from lalsimulation import git_version, __version__
            lalsim_version = str(__version__)
            logger.info("Using lalsimulation version {}".format(lalsim_version))
            lalsim_git_version = str(git_version.verbose_msg).replace("\n", ";")
            logger.info("Using lalsimulation git version {}".format(lalsim_git_version))
            return "lalsimulation_version={}, lalsimulation_git_version={}".format(lalsim_version, lalsim_git_version)
        except (ImportError, AttributeError):
            return "N/A"

    @property
    def meta_data(self):
        return dict(
            interferometers=self.interferometers.meta_data,
            time_marginalization=self.time_marginalization,
            phase_marginalization=self.phase_marginalization,
            distance_marginalization=self.distance_marginalization,
            calibration_marginalization=self.calibration_marginalization,
            waveform_generator_class=self.waveform_generator.__class__,
            waveform_arguments=self.waveform_generator.waveform_arguments,
            frequency_domain_source_model=self.waveform_generator.frequency_domain_source_model,
            time_domain_source_model=self.waveform_generator.time_domain_source_model,
            parameter_conversion=self.waveform_generator.parameter_conversion,
            sampling_frequency=self.waveform_generator.sampling_frequency,
            duration=self.waveform_generator.duration,
            start_time=self.waveform_generator.start_time,
            time_reference=self.time_reference,
            reference_frame=self._reference_frame_str,
            lal_version=self.lal_version,
            lalsimulation_version=self.lalsimulation_version,
            waveform_generator_meta_data=self.waveform_generator.meta_data)
