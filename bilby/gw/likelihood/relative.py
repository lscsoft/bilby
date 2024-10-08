import numpy as np
from scipy.optimize import differential_evolution

from .base import GravitationalWaveTransient
from ...core.utils import logger
from ...core.prior.base import Constraint
from ...core.prior import DeltaFunction
from ..utils import noise_weighted_inner_product


class RelativeBinningGravitationalWaveTransient(GravitationalWaveTransient):
    """A gravitational-wave transient likelihood object which uses the relative
    binning procedure to calculate a fast likelihood. See Zackay et al.
    arXiv1806.08792

    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
    fiducial_parameters: dict, optional
        A starting guess for initial parameters of the event for finding the
        maximum likelihood (fiducial) waveform. These should be specified in
        the same parameter basis as the one that sampling is carried out in.
        For example, if sampling in `mass_1` and `mass_2`, the fiducial
        parameters should also be provided in `mass_1` and `mass_2.`
    parameter_bounds: dict, optional
        Dictionary of bounds (lists) for the initial parameters when finding
        the initial maximum likelihood (fiducial) waveform.
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
    priors: dict, optional
        If given, used in the distance and phase marginalization.
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table.
        If a string the name of a file containing these quantities.
        The lookup table is stored after construction in either the
        provided string or a default location:
        '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'
    jitter_time: bool, optional
        Whether to introduce a `time_jitter` parameter. This avoids either
        missing the likelihood peak, or introducing biases in the
        reconstructed time posterior due to an insufficient sampling frequency.
        Default is False, however using this parameter is strongly encouraged.
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.
        - "sky": sample in RA/dec, this is the default
        - e.g., "H1L1", ["H1", "L1"], InterferometerList(["H1", "L1"]):
          sample in azimuth and zenith, `azimuth` and `zenith` defined in the
          frame where the z-axis is aligned the the vector connecting H1
          and L1.
    time_reference: str, optional
        Name of the reference for the sampled time parameter.
        - "geocent"/"geocenter": sample in the time at the Earth's center,
          this is the default
        - e.g., "H1": sample in the time of arrival at H1
    chi: float, optional
        Tunable parameter which limits the perturbation of alpha when setting
        up the bin range. See https://arxiv.org/abs/1806.08792.
    epsilon: float, optional
        Tunable parameter which limits the differential phase change in each
        bin when setting up the bin range. See https://arxiv.org/abs/1806.08792.

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        some model parameters.

    Notes
    -----
    The relative binning likelihood does not currently support calibration marginalization.
    """

    def __init__(self, interferometers,
                 waveform_generator,
                 fiducial_parameters=None,
                 parameter_bounds=None,
                 maximization_kwargs=None,
                 update_fiducial_parameters=False,
                 distance_marginalization=False,
                 time_marginalization=False,
                 phase_marginalization=False,
                 priors=None,
                 distance_marginalization_lookup_table=None,
                 jitter_time=True,
                 reference_frame="sky",
                 time_reference="geocenter",
                 chi=1,
                 epsilon=0.5):

        super(RelativeBinningGravitationalWaveTransient, self).__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference)

        if fiducial_parameters is None:
            logger.info("Drawing fiducial parameters from prior.")
            fiducial_parameters = priors.sample()
        self.fiducial_parameters = fiducial_parameters.copy()
        self.fiducial_parameters["fiducial"] = 0
        if self.time_marginalization:
            self.fiducial_parameters["geocent_time"] = interferometers.start_time
        if self.distance_marginalization:
            self.fiducial_parameters["luminosity_distance"] = self._ref_dist
        if self.phase_marginalization:
            self.fiducial_parameters["phase"] = 0.0
        self.chi = chi
        self.epsilon = epsilon
        self.gamma = np.array([-5 / 3, -2 / 3, 1, 5 / 3, 7 / 3])
        self.maximum_frequency = waveform_generator.frequency_array[-1]
        self.fiducial_waveform_obtained = False
        self.check_if_bins_are_setup = False
        self.fiducial_polarizations = None
        self.per_detector_fiducial_waveforms = dict()
        self.per_detector_fiducial_waveform_points = dict()
        self.set_fiducial_waveforms(self.fiducial_parameters)
        logger.info("Initial fiducial waveforms set up")
        self.setup_bins()
        self.compute_summary_data()
        logger.info("Summary Data Obtained")

        if update_fiducial_parameters:
            # write a check to make sure prior is not None
            logger.info("Using scipy optimization to find maximum likelihood parameters.")
            self.parameters_to_be_updated = [key for key in priors if not isinstance(
                priors[key], (DeltaFunction, Constraint, float, int))]
            logger.info(f"Parameters over which likelihood is maximized: {self.parameters_to_be_updated}")
            if parameter_bounds is None:
                logger.info("No parameter bounds were given. Using priors instead.")
                self.parameter_bounds = self.get_bounds_from_priors(priors)
            else:
                self.parameter_bounds = self.get_parameter_list_from_dictionary(parameter_bounds)
            self.fiducial_parameters = self.find_maximum_likelihood_parameters(
                self.parameter_bounds, maximization_kwargs=maximization_kwargs)
        self.parameters.update(self.fiducial_parameters)
        logger.info(f"Fiducial likelihood: {self.log_likelihood_ratio():.2f}")
        self.parameters = dict(fiducial=0)

    def __repr__(self):
        return self.__class__.__name__ + '(interferometers={},\n\twaveform_generator={},\n\fiducial_parameters={},' \
            .format(self.interferometers, self.waveform_generator, self.fiducial_parameters)

    def setup_bins(self):
        """
        Setup the frequency bins following the method in
        https://arxiv.org/abs/1806.08792.

        If :code:`epsilon` is too small, the naive bins can be smaller than
        the frequency spacing of the data. We require that bins are at least
        as wide as this spacing.
        """
        frequency_array = self.waveform_generator.frequency_array
        gamma = self.gamma[:, np.newaxis]
        maximum_frequency = frequency_array[0]
        minimum_frequency = frequency_array[-1]
        for interferometer in self.interferometers:
            maximum_frequency = max(maximum_frequency, interferometer.maximum_frequency)
            minimum_frequency = min(minimum_frequency, interferometer.minimum_frequency)
        maximum_frequency = min(maximum_frequency, self.maximum_frequency)
        frequency_array_useful = frequency_array[
            (frequency_array >= minimum_frequency)
            & (frequency_array <= maximum_frequency)
        ]

        d_alpha = self.chi * 2 * np.pi / np.abs(
            (minimum_frequency ** gamma) * np.heaviside(-gamma, 1)
            - (maximum_frequency ** gamma) * np.heaviside(gamma, 1)
        )
        d_phi = np.sum(
            np.sign(gamma) * d_alpha * frequency_array_useful ** gamma,
            axis=0
        )
        d_phi_from_start = d_phi - d_phi[0]
        number_of_bins = int(d_phi_from_start[-1] // self.epsilon)
        bin_inds = list()
        bin_freqs = list()

        last_index = -1
        for i in range(number_of_bins + 1):
            bin_index = np.where(d_phi_from_start >= ((i / number_of_bins) * d_phi_from_start[-1]))[0][0]
            if bin_index == last_index:
                continue
            bin_freq = frequency_array_useful[bin_index]
            last_index = bin_index
            bin_index = np.where(frequency_array >= bin_freq)[0][0]
            bin_inds.append(bin_index)
            bin_freqs.append(bin_freq)
        self.bin_inds = np.array(bin_inds, dtype=int)
        self.bin_sizes = np.diff(bin_inds)
        self.bin_sizes[-1] += 1
        self.bin_freqs = np.array(bin_freqs)
        self.number_of_bins = len(self.bin_inds) - 1
        logger.debug(
            f"Set up {self.number_of_bins} bins "
            f"between {minimum_frequency} Hz and {maximum_frequency} Hz"
        )
        self.waveform_generator.waveform_arguments["frequency_bin_edges"] = self.bin_freqs
        self.bin_widths = self.bin_freqs[1:] - self.bin_freqs[:-1]
        self.bin_centers = (self.bin_freqs[1:] + self.bin_freqs[:-1]) / 2
        for interferometer in self.interferometers:
            name = interferometer.name
            self.per_detector_fiducial_waveform_points[name] = (
                self.per_detector_fiducial_waveforms[name][self.bin_inds]
            )

    def set_fiducial_waveforms(self, parameters):
        parameters = parameters.copy()
        parameters["fiducial"] = 1
        parameters.update(self.get_sky_frame_parameters(parameters=parameters))
        self.fiducial_polarizations = self.waveform_generator.frequency_domain_strain(
            parameters)

        maximum_nonzero_index = np.where(self.fiducial_polarizations["plus"] != 0j)[0][-1]
        logger.debug(f"Maximum Nonzero Index is {maximum_nonzero_index}")
        maximum_nonzero_frequency = self.waveform_generator.frequency_array[maximum_nonzero_index]
        logger.debug(f"Maximum Nonzero Frequency is {maximum_nonzero_frequency}")
        self.maximum_frequency = maximum_nonzero_frequency

        if self.fiducial_polarizations is None:
            raise ValueError(f"Cannot compute fiducial waveforms for {parameters}")

        for interferometer in self.interferometers:
            logger.debug(f"Maximum Frequency is {interferometer.maximum_frequency}")
            wf = interferometer.get_detector_response(self.fiducial_polarizations, parameters)
            wf[interferometer.frequency_array > self.maximum_frequency] = 0
            self.per_detector_fiducial_waveforms[interferometer.name] = wf

    def find_maximum_likelihood_parameters(self, parameter_bounds,
                                           iterations=5, maximization_kwargs=None):
        if maximization_kwargs is None:
            maximization_kwargs = dict()
        self.parameters.update(self.fiducial_parameters)
        self.parameters["fiducial"] = 0
        updated_parameters_list = self.get_parameter_list_from_dictionary(self.fiducial_parameters)
        old_fiducial_ln_likelihood = self.log_likelihood_ratio()
        logger.info(f"Fiducial ln likelihood ratio: {old_fiducial_ln_likelihood:.2f}")
        for it in range(iterations):
            logger.info(f"Optimizing fiducial parameters. Iteration : {it + 1}")
            output = differential_evolution(
                self.lnlike_scipy_maximize,
                bounds=parameter_bounds,
                x0=updated_parameters_list,
                **maximization_kwargs,
            )
            updated_parameters_list = output['x']
            updated_parameters = self.get_parameter_dictionary_from_list(updated_parameters_list)
            self.parameters.update(updated_parameters)
            self.set_fiducial_waveforms(updated_parameters)
            self.setup_bins()
            self.compute_summary_data()
            new_fiducial_ln_likelihood = self.log_likelihood_ratio()
            logger.info(f"Fiducial ln likelihood ratio: {new_fiducial_ln_likelihood:.2f}")
            if new_fiducial_ln_likelihood - old_fiducial_ln_likelihood < 0.1:
                break
            old_fiducial_ln_likelihood = new_fiducial_ln_likelihood

        logger.info("Fiducial waveforms updated")
        logger.info("Summary Data updated")
        return updated_parameters

    def lnlike_scipy_maximize(self, parameter_list):
        self.parameters.update(self.get_parameter_dictionary_from_list(parameter_list))
        return -self.log_likelihood_ratio()

    def get_parameter_dictionary_from_list(self, parameter_list):
        parameter_dictionary = dict(zip(self.parameters_to_be_updated, parameter_list))
        excluded_parameter_keys = set(self.fiducial_parameters) - set(self.parameters_to_be_updated)
        for key in excluded_parameter_keys:
            parameter_dictionary[key] = self.fiducial_parameters[key]
        return parameter_dictionary

    def get_parameter_list_from_dictionary(self, parameter_dict):
        return [parameter_dict[k] for k in self.parameters_to_be_updated]

    def get_bounds_from_priors(self, priors):
        bounds = []
        for key in self.parameters_to_be_updated:
            bounds.append([priors[key].minimum, priors[key].maximum])
        return bounds

    def compute_summary_data(self):
        summary_data = dict()

        for interferometer in self.interferometers:
            mask = interferometer.frequency_mask
            masked_frequency_array = interferometer.frequency_array[mask]
            masked_bin_inds = []
            for edge in self.bin_freqs:
                index = np.where(masked_frequency_array == edge)[0][0]
                masked_bin_inds.append(index)
            # For the last bin, make sure to include
            # the last point in the frequency array,
            # if it's not already included
            if masked_bin_inds[-1] < len(masked_frequency_array) - 1:
                masked_bin_inds[-1] += 1

            masked_strain = interferometer.frequency_domain_strain[mask]
            masked_h0 = self.per_detector_fiducial_waveforms[interferometer.name][mask]
            masked_psd = interferometer.power_spectral_density_array[mask]
            duration = interferometer.duration
            a0, b0, a1, b1 = np.zeros((4, self.number_of_bins), dtype=complex)

            for i in range(self.number_of_bins):
                start_idx = masked_bin_inds[i]
                end_idx = masked_bin_inds[i + 1]
                start = masked_frequency_array[start_idx]
                stop = masked_frequency_array[end_idx]
                idxs = slice(start_idx, end_idx)

                strain = masked_strain[idxs]
                h0 = masked_h0[idxs]
                psd = masked_psd[idxs]

                frequencies = masked_frequency_array[idxs]
                central_frequency = (start + stop) / 2
                delta_frequency = frequencies - central_frequency

                a0[i] = noise_weighted_inner_product(h0, strain, psd, duration)
                b0[i] = noise_weighted_inner_product(h0, h0, psd, duration)
                a1[i] = noise_weighted_inner_product(h0, strain * delta_frequency, psd, duration)
                b1[i] = noise_weighted_inner_product(h0, h0 * delta_frequency, psd, duration)

            summary_data[interferometer.name] = (a0, a1, b0, b1)

        self.summary_data = summary_data

    def compute_waveform_ratio_per_interferometer(self, waveform_polarizations, interferometer):
        name = interferometer.name
        strain = interferometer.get_detector_response(
            waveform_polarizations=waveform_polarizations,
            parameters=self.parameters,
            frequencies=self.bin_freqs,
        )
        reference_strain = self.per_detector_fiducial_waveform_points[name]
        waveform_ratio = strain / reference_strain

        r0 = (waveform_ratio[1:] + waveform_ratio[:-1]) / 2
        r1 = (waveform_ratio[1:] - waveform_ratio[:-1]) / self.bin_widths

        return [r0, r1]

    def _compute_full_waveform(self, signal_polarizations, interferometer):
        fiducial_waveform = self.per_detector_fiducial_waveforms[interferometer.name]
        r0, r1 = self.compute_waveform_ratio_per_interferometer(
            waveform_polarizations=signal_polarizations,
            interferometer=interferometer,
        )

        idxs = slice(self.bin_inds[0], self.bin_inds[-1] + 1)
        duplicated_r0 = np.repeat(r0, self.bin_sizes)
        duplicated_r1 = np.repeat(r1, self.bin_sizes)
        duplicated_fm = np.repeat(self.bin_centers, self.bin_sizes)

        f = interferometer.frequency_array
        full_waveform_ratio = np.zeros(f.shape[0], dtype=complex)
        full_waveform_ratio[idxs] = duplicated_r0 + duplicated_r1 * (f[idxs] - duplicated_fm)
        return fiducial_waveform * full_waveform_ratio

    def calculate_snrs(self, waveform_polarizations, interferometer, return_array=True):
        r0, r1 = self.compute_waveform_ratio_per_interferometer(
            waveform_polarizations=waveform_polarizations,
            interferometer=interferometer,
        )
        a0, a1, b0, b1 = self.summary_data[interferometer.name]
        d_inner_h = np.sum(a0 * np.conjugate(r0) + a1 * np.conjugate(r1))
        h_inner_h = np.sum(b0 * np.abs(r0) ** 2 + 2 * b1 * np.real(r0 * np.conjugate(r1)))
        optimal_snr_squared = h_inner_h
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared ** 0.5)

        if return_array and self.time_marginalization:
            full_waveform = self._compute_full_waveform(
                signal_polarizations=waveform_polarizations,
                interferometer=interferometer,
            )
            d_inner_h_array = 4 / self.waveform_generator.duration * np.fft.fft(
                full_waveform[0:-1]
                * interferometer.frequency_domain_strain.conjugate()[0:-1]
                / interferometer.power_spectral_density_array[0:-1])

        else:
            d_inner_h_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared.real,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array
        )
