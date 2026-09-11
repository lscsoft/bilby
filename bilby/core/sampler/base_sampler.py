import datetime
import os
import shutil
import signal
import tempfile
import time
from copy import deepcopy

import numpy as np
from pandas import DataFrame

from ..prior import Constraint, DeltaFunction, Prior, PriorDict
from ..result import Result, read_in_result
from ..utils import (
    Counter,
    check_directory_exists_and_if_not_mkdir,
    command_line_args,
    logger,
)
from ..utils.parallel import (
    close_pool,
    create_pool,
    initialize_global_variables,
    sampling_convenience_dump,
)
from ..utils.random import seed as set_seed


def signal_wrapper(method):
    """
    Decorator to wrap a method of a class to set system signals before running
    and reset them after.

    Parameters
    ==========
    method: callable
        The method to call, this assumes the first argument is `self`
        and that `self` has a `write_current_state_and_exit` method.

    Returns
    =======
    output: callable
        The wrapped method.
    """

    def wrapped(self, *args, **kwargs):
        try:
            old_term = signal.signal(signal.SIGTERM, self.write_current_state_and_exit)
            old_int = signal.signal(signal.SIGINT, self.write_current_state_and_exit)
            old_alarm = signal.signal(signal.SIGALRM, self.write_current_state_and_exit)
            _set = True
        except (AttributeError, ValueError):
            _set = False
            logger.debug(
                "Setting signal attributes unavailable on this system. "
                "This is likely the case if you are running on a Windows machine "
                "and can be safely ignored."
            )
        output = method(self, *args, **kwargs)
        if _set:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGALRM, old_alarm)
        return output

    return wrapped


class Sampler(object):
    """A sampler object to aid in setting up an inference run

    Parameters
    ==========
    likelihood: likelihood.Likelihood
        A  object with a log_l method
    priors: bilby.core.prior.PriorDict, dict
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    external_sampler: str, Sampler, optional
        A string containing the module name of the sampler or an instance of
        this class
    outdir: str, optional
        Name of the output directory
    label: str, optional
        Naming scheme of the output files
    use_ratio: bool, optional
        Switch to set whether or not you want to use the log-likelihood ratio
        or just the log-likelihood
    plot: bool, optional
        Switch to set whether or not you want to create traceplots
    injection_parameters:
        A dictionary of the injection parameters
    meta_data:
        A dictionary of extra meta data to store in the result
    result_class: bilby.core.result.Result, or child of
        The result class to use. By default, `bilby.core.result.Result` is used,
        but objects which inherit from this class can be given providing
        additional methods.
    soft_init: bool, optional
        Switch to enable a soft initialization that prevents the likelihood
        from being tested before running the sampler. This is relevant when
        using custom likelihoods that must NOT be initialized on the main thread
        when using multiprocessing, e.g. when using tensorflow in the likelihood.
    **kwargs: dict
        Additional keyword arguments

    Attributes
    ==========
    likelihood: likelihood.Likelihood
        A  object with a log_l method
    priors: bilby.core.prior.PriorDict
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    external_sampler: Module
        An external module containing an implementation of a sampler.
    outdir: str
        Name of the output directory
    label: str
        Naming scheme of the output files
    use_ratio: bool
        Switch to set whether or not you want to use the log-likelihood ratio
        or just the log-likelihood
    plot: bool
        Switch to set whether or not you want to create traceplots
    skip_import_verification: bool
        Skips the check if the sampler is installed if true. This is
        only advisable for testing environments
    result: bilby.core.result.Result
        Container for the results of the sampling run
    exit_code: int
        System exit code to return on interrupt
    kwargs: dict
        Dictionary of keyword arguments that can be used in the external sampler
    hard_exit: bool
        Whether the implemented sampler exits hard (:code:`os._exit` rather
        than :code:`sys.exit`). The latter can be escaped as :code:`SystemExit`.
        The former cannot.
    sampler_name : str
        Name of the sampler. This is used when creating the output directory for
        the sampler.
    abbreviation : str
        Abbreviated name of the sampler. Does not have to be specified in child
        classes. If set to a value other than :code:`None`, this will be used
        instead of :code:`sampler_name` when creating the output directory.

    Raises
    ======
    TypeError:
        If external_sampler is neither a string nor an instance of this class
        If not all required parameters have been defined
    ImportError:
        If the external_sampler string does not refer to a sampler that is
        installed on this system
    AttributeError:
        If some of the priors can't be sampled

    """

    sampler_name = "sampler"
    abbreviation = None
    default_kwargs = dict()
    npool_equiv_kwargs = [
        "npool",
        "queue_size",
        "threads",
        "nthreads",
        "cores",
        "n_pool",
    ]
    sampling_seed_equiv_kwargs = ["sampling_seed", "seed", "random_seed"]
    hard_exit = False
    sampling_seed_key = None
    """Name of keyword argument for setting the sampling for the specific sampler.
    If a specific sampler does not have a sampling seed option, then it should be
    left as None.
    """
    check_point_equiv_kwargs = ["check_point_deltaT", "check_point_delta_t"]

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        use_ratio=False,
        plot=False,
        skip_import_verification=False,
        injection_parameters=None,
        meta_data=None,
        result_class=None,
        likelihood_benchmark=False,
        soft_init=False,
        exit_code=130,
        npool=1,
        pool=None,
        **kwargs,
    ):
        self.likelihood = likelihood
        self.parameters = dict()
        if isinstance(priors, PriorDict):
            self.priors = priors
        else:
            self.priors = PriorDict(priors)
        self.label = label
        self.outdir = outdir
        self.injection_parameters = injection_parameters
        self.meta_data = meta_data
        self.use_ratio = use_ratio
        self.pool = pool
        self._npool = npool
        if not skip_import_verification:
            self._verify_external_sampler()
        self.external_sampler_function = None
        self.plot = plot
        self.likelihood_benchmark = likelihood_benchmark

        self._search_parameter_keys = list()
        self._fixed_parameter_keys = list()
        self._constraint_parameter_keys = list()
        self._initialise_parameters()
        self._log_information_about_priors_and_likelihood()

        self.exit_code = exit_code

        self._log_likelihood_eval_time = np.nan
        if not soft_init:
            self._verify_parameters()
            self._log_likelihood_eval_time = self._time_likelihood()
            self._verify_use_ratio()

        self.kwargs = kwargs

        self._check_cached_result(result_class)

        self._log_summary_for_sampler()

        self.result = self._initialise_result(result_class)
        self.likelihood_count = None
        if self.likelihood_benchmark:
            self.likelihood_count = Counter()

    @property
    def search_parameter_keys(self):
        """list: List of parameter keys that are being sampled"""
        return self._search_parameter_keys

    @property
    def fixed_parameter_keys(self):
        """list: List of parameter keys that are not being sampled"""
        return self._fixed_parameter_keys

    @property
    def constraint_parameter_keys(self):
        """list: List of parameters providing prior constraints"""
        return self._constraint_parameter_keys

    @property
    def ndim(self):
        """int: Number of dimensions of the search parameter space"""
        return len(self._search_parameter_keys)

    @property
    def kwargs(self):
        """dict: Container for the kwargs. Has more sophisticated logic in subclasses"""
        return self._kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self._kwargs = self.default_kwargs.copy()
        self._translate_kwargs(kwargs)
        self._kwargs.update(kwargs)
        self._verify_kwargs_against_default_kwargs()

    def _translate_kwargs(self, kwargs):
        """Translate keyword arguments.

        Default only translates the sampling seed if the sampler has
        :code:`sampling_seed_key` set.
        """
        if self.sampling_seed_key and self.sampling_seed_key not in kwargs:
            for equiv in self.sampling_seed_equiv_kwargs:
                if equiv in kwargs:
                    kwargs[self.sampling_seed_key] = kwargs.pop(equiv)
                    set_seed(kwargs[self.sampling_seed_key])
        return kwargs

    @property
    def external_sampler_name(self):
        return self.__class__.__name__.lower()

    def _verify_external_sampler(self):
        external_sampler_name = self.external_sampler_name
        try:
            __import__(external_sampler_name)
        except (ImportError, SystemExit):
            raise SamplerNotInstalledError(
                f"Sampler {external_sampler_name} is not installed on this system"
            )

    def _verify_kwargs_against_default_kwargs(self):
        """
        Check if the kwargs are contained in the list of available arguments
        of the external sampler.
        """
        args = self.default_kwargs
        bad_keys = []
        for user_input in self.kwargs.keys():
            if user_input not in args:
                logger.warning(
                    f"Supplied argument '{user_input}' not an argument of '{self.__class__.__name__}', removing."
                )
                bad_keys.append(user_input)
        for key in bad_keys:
            self.kwargs.pop(key)

    def _initialise_parameters(self):
        """
        Go through the list of priors and add keys to the fixed and search
        parameter key list depending on whether
        the respective parameter is fixed.
        """
        for key in self.priors:
            if (
                isinstance(self.priors[key], Prior)
                and self.priors[key].is_fixed is False
            ):
                self._search_parameter_keys.append(key)
            elif isinstance(self.priors[key], Constraint):
                self._constraint_parameter_keys.append(key)
            elif isinstance(self.priors[key], DeltaFunction):
                self.parameters[key] = self.priors[key].sample()
                self._fixed_parameter_keys.append(key)

    def _log_information_about_priors_and_likelihood(self):
        logger.info("Analysis priors:")
        for key in self._search_parameter_keys + self._constraint_parameter_keys:
            logger.info(f"{key}={self.priors[key]}")
        for key in self._fixed_parameter_keys:
            logger.info(f"{key}={self.priors[key].peak}")
        logger.info(f"Analysis likelihood class: {self.likelihood.__class__}")
        logger.info(
            f"Analysis likelihood noise evidence: {self.likelihood.noise_log_likelihood()}"
        )

    def _initialise_result(self, result_class):
        """
        Returns
        =======
        bilby.core.result.Result: An initial template for the result

        """
        result_kwargs = dict(
            label=self.label,
            outdir=self.outdir,
            sampler=self.__class__.__name__.lower(),
            search_parameter_keys=self._search_parameter_keys,
            fixed_parameter_keys=self._fixed_parameter_keys,
            constraint_parameter_keys=self._constraint_parameter_keys,
            priors=self.priors,
            meta_data=self.meta_data,
            injection_parameters=self.injection_parameters,
            sampler_kwargs=self.kwargs,
            use_ratio=self.use_ratio,
        )

        if result_class is None:
            result = Result(**result_kwargs)
        elif issubclass(result_class, Result):
            result = result_class(**result_kwargs)
        else:
            raise ValueError(f"Input result_class={result_class} not understood")

        return result

    def _verify_parameters(self):
        """Evaluate a set of parameters drawn from the prior

        Tests if the likelihood evaluation passes

        Raises
        ======
        TypeError
            Likelihood can't be evaluated.

        """

        if self.priors.test_has_redundant_keys():
            raise IllegalSamplingSetError(
                "Your sampling set contains redundant parameters."
            )

        theta = self.priors.sample_subset_constrained_as_array(
            self.search_parameter_keys, size=1
        )[:, 0]
        try:
            self.log_likelihood(theta)
        except TypeError as e:
            params = deepcopy(self.parameters)
            params.update(
                {key: val for key, val in zip(self.search_parameter_keys, theta)}
            )
            raise TypeError(
                f"Likelihood evaluation failed with message: \n'{e}'\n"
                f"Have you specified all the parameters:\n{params}"
            )

    def _time_likelihood(self, n_evaluations=100):
        """Times the likelihood evaluation and print an info message

        Parameters
        ==========
        n_evaluations: int
            The number of evaluations to estimate the evaluation time from

        Returns
        =======
        log_likelihood_eval_time: float
            The time (in s) it took for one likelihood evaluation
        """

        t1 = datetime.datetime.now()
        for _ in range(n_evaluations):
            theta = self.priors.sample_subset_constrained_as_array(
                self._search_parameter_keys, size=1
            )[:, 0]
            self.log_likelihood(theta)
        total_time = (datetime.datetime.now() - t1).total_seconds()
        log_likelihood_eval_time = total_time / n_evaluations

        if log_likelihood_eval_time == 0:
            log_likelihood_eval_time = np.nan
            logger.info("Unable to measure single likelihood time")
        else:
            logger.info(
                f"Single likelihood evaluation took {log_likelihood_eval_time:.3e} s"
            )
        return log_likelihood_eval_time

    def _verify_use_ratio(self):
        """
        Checks if use_ratio is set. Prints a warning if use_ratio is set but
        not properly implemented.
        """
        try:
            self.priors.sample_subset(self.search_parameter_keys)
        except (KeyError, AttributeError):
            logger.error(
                f"Cannot sample from priors with keys: {self.search_parameter_keys}."
            )
            raise
        if self.use_ratio is False:
            logger.debug("use_ratio set to False")
            return

        parameters = deepcopy(self.parameters)
        parameters.update(self.priors.sample())

        ratio_is_nan = np.isnan(self.likelihood.log_likelihood_ratio(parameters))

        if self.use_ratio is True and ratio_is_nan:
            logger.warning(
                "You have requested to use the loglikelihood_ratio, but it "
                " returns a NaN"
            )
        elif self.use_ratio is None and not ratio_is_nan:
            logger.debug("use_ratio not spec. but gives valid answer, setting True")
            self.use_ratio = True

    def prior_transform(self, theta):
        """Prior transform method that is passed into the external sampler.

        Parameters
        ==========
        theta: list
            List of sampled values on a unit interval

        Returns
        =======
        list: Properly rescaled sampled values
        """
        return self.priors.rescale(self._search_parameter_keys, theta)

    def log_prior(self, theta):
        """

        Parameters
        ==========
        theta: list
            List of sampled values on a unit interval

        Returns
        =======
        float: Joint ln prior probability of theta

        """
        params = {key: t for key, t in zip(self._search_parameter_keys, theta)}
        return self.priors.ln_prob(params)

    def log_likelihood(self, theta):
        """

        Parameters
        ==========
        theta: list
            List of values for the likelihood parameters

        Returns
        =======
        float: Log-likelihood or log-likelihood-ratio given the current
            likelihood.parameter values

        """
        if self.likelihood_benchmark:
            try:
                self.likelihood_count.increment()
            except AttributeError:
                pass

        params = deepcopy(self.parameters)
        params.update({key: t for key, t in zip(self._search_parameter_keys, theta)})
        if self.use_ratio:
            return self.likelihood.log_likelihood_ratio(params)
        else:
            return self.likelihood.log_likelihood(params)

    def get_random_draw_from_prior(self):
        """Get a random draw from the prior distribution

        Returns
        =======
        draw: array_like
            An ndim-length array of values drawn from the prior. Parameters
            with delta-function (or fixed) priors are not returned

        """
        new_sample = self.priors.sample()
        draw = np.array(list(new_sample[key] for key in self._search_parameter_keys))
        self.check_draw(draw)
        return draw

    def get_initial_points_from_prior(self, npoints=1):
        """Method to draw a set of live points from the prior

        This iterates over draws from the prior until all the samples have a
        finite prior and likelihood (relevant for constrained priors).

        Parameters
        ==========
        npoints: int
            The number of values to return

        Returns
        =======
        unit_cube, parameters, likelihood: tuple of array_like
            unit_cube (nlive, ndim) is an array of the prior samples from the
            unit cube, parameters (nlive, ndim) is the unit_cube array
            transformed to the target space, while likelihood (nlive) are the
            likelihood evaluations.

        """
        from ..utils import random

        logger.info("Generating initial points from the prior")
        unit_cube = []
        parameters = []
        likelihood = []
        while len(unit_cube) < npoints:
            unit = random.rng.uniform(0, 1, self.ndim)
            theta = self.prior_transform(unit)
            if self.check_draw(theta, warning=False):
                unit_cube.append(unit)
                parameters.append(theta)
                likelihood.append(self.log_likelihood(theta))

        return np.array(unit_cube), np.array(parameters), np.array(likelihood)

    def check_draw(self, theta, warning=True):
        """
        Checks if the draw will generate an infinite prior or likelihood

        Also catches the output of `numpy.nan_to_num`.

        Parameters
        ==========
        theta: array_like
            Parameter values at which to evaluate likelihood
        warning: bool
            Whether or not to print a warning

        Returns
        =======
        bool, cube (nlive,
            True if the likelihood and prior are finite, false otherwise

        """
        log_p = self.log_prior(theta)
        log_l = self.log_likelihood(theta)
        return self._check_bad_value(
            val=log_p, warning=warning, theta=theta, label="prior"
        ) and self._check_bad_value(
            val=log_l, warning=warning, theta=theta, label="likelihood"
        )

    @staticmethod
    def _check_bad_value(val, warning, theta, label):
        val = np.abs(val)
        bad_values = [np.inf, np.nan_to_num(np.inf)]
        if val in bad_values or np.isnan(val):
            if warning:
                logger.warning(f"Prior draw {theta} has inf {label}")
            return False
        return True

    def run_sampler(self):
        """A template method to run in subclasses"""
        pass

    def _run_test(self):
        """
        TODO: Implement this method
        Raises
        =======
        ValueError: in any case
        """
        raise ValueError("Method not yet implemented")

    def _check_cached_result(self, result_class=None):
        """Check if the cached data file exists and can be used"""

        if command_line_args.clean:
            logger.debug("Command line argument clean given, forcing rerun")
            self.cached_result = None
            return

        try:
            self.cached_result = read_in_result(
                outdir=self.outdir, label=self.label, result_class=result_class
            )
        except IOError:
            self.cached_result = None

        if command_line_args.use_cached:
            logger.debug("Command line argument cached given, no cache check performed")
            return

        logger.debug("Checking cached data")
        if self.cached_result:
            check_keys = ["search_parameter_keys", "fixed_parameter_keys"]
            use_cache = True
            for key in check_keys:
                if (
                    self.cached_result._check_attribute_match_to_other_object(key, self)
                    is False
                ):
                    logger.debug(f"Cached value {key} is unmatched")
                    use_cache = False
            try:
                # Recursive check the dictionaries allowing for numpy arrays
                np.testing.assert_equal(
                    self.meta_data["likelihood"],
                    self.cached_result.meta_data["likelihood"],
                )
            except AssertionError:
                use_cache = False
            if use_cache is False:
                self.cached_result = None

    def _log_summary_for_sampler(self):
        """Print a summary of the sampler used and its kwargs"""
        if self.cached_result is None:
            kwargs_print = self.kwargs.copy()
            for k in kwargs_print:
                if isinstance(kwargs_print[k], (list, np.ndarray)):
                    array_repr = np.array(kwargs_print[k])
                    if array_repr.size > 10:
                        kwargs_print[k] = f"array_like, shape={array_repr.shape}"
                elif isinstance(kwargs_print[k], DataFrame):
                    kwargs_print[k] = f"DataFrame, shape={kwargs_print[k].shape}"
            logger.info(
                f"Using sampler {self.__class__.__name__} with kwargs {kwargs_print}"
            )

    def calc_likelihood_count(self):
        if self.likelihood_benchmark:
            self.result.num_likelihood_evaluations = self.likelihood_count.value
        else:
            return None

    @property
    def npool(self):
        for key in self.npool_equiv_kwargs:
            if key in self.kwargs:
                return self.kwargs[key]
        return self._npool

    def _log_interruption(self, signum=None):
        if signum == 14:
            logger.info(
                f"Run interrupted by alarm signal {signum}: checkpoint and exit on {self.exit_code}"
            )
        else:
            logger.info(
                f"Run interrupted by signal {signum}: checkpoint and exit on {self.exit_code}"
            )

    def write_current_state_and_exit(self, signum=None, frame=None):
        """
        Make sure that if a pool of jobs is running only the parent tries to
        checkpoint and exit. Only the parent has a 'pool' attribute.

        For samplers that must hard exit (typically due to non-Python process)
        use :code:`os._exit` that cannot be excepted. Other samplers exiting
        can be caught as a :code:`SystemExit`.
        """
        if self.npool in (1, None) or getattr(self, "pool", None) is not None:

            self._interrupted = True
            self._interruption_signum = signum

            self._log_interruption(signum=signum)
            self.write_current_state()
            self._close_pool()

            if self.hard_exit:
                os._exit(self.exit_code)
            raise SystemExit(self.exit_code)

    def _raise_if_interrupted(self, cause):
        if getattr(self, "_interrupted", False):
            raise SystemExit(self.exit_code) from cause

    def _close_pool(self):
        if getattr(self, "pool", None) is not None and not getattr(
            self, "_user_pool", True
        ):
            logger.info("Starting to close worker pool.")
            close_pool(self.pool)
            self.pool = None
            self.kwargs["pool"] = self.pool
            logger.info("Finished closing worker pool.")

    def _setup_pool(self):
        parameters = self.priors.sample()

        if hasattr(self.pool, "map"):
            self._user_pool = True
        elif self.npool in (1, None):
            self._user_pool = False
        else:
            self._user_pool = False
            self.pool = create_pool(
                likelihood=self.likelihood,
                priors=self.priors,
                search_parameter_keys=self._search_parameter_keys,
                use_ratio=self.use_ratio,
                npool=self.npool,
                pool=self.pool,
                parameters=parameters,
            )
            if self.pool is not None:
                logger.warning(
                    "Setting up parallel pool in sampler is deprecated. Use "
                    "bilby.utils.parallel.bilby_pool context instead."
                )
        initialize_global_variables(
            likelihood=self.likelihood,
            priors=self.priors,
            search_parameter_keys=self._search_parameter_keys,
            use_ratio=self.use_ratio,
            parameters=parameters,
        )
        self.kwargs["pool"] = self.pool

    def write_current_state(self):
        raise NotImplementedError()

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        """Get lists of the expected outputs directories and files.

        These are used by :code:`bilby_pipe` when transferring files via HTCondor.
        Both can be empty. Defaults to a single directory:
        :code:`"{outdir}/{name}_{label}/"`, where :code:`name`
        is :code:`abbreviation` if it is defined for the sampler class, otherwise
        it defaults to :code:`sampler_name`.

        Parameters
        ----------
        outdir : str
            The output directory.
        label : str
            The label for the run.

        Returns
        -------
        list
            List of file names.
        list
            List of directory names.
        """
        name = cls.abbreviation or cls.sampler_name
        dirname = os.path.join(outdir, f"{name}_{label}", "")
        return [], [dirname]


class NestedSampler(Sampler):
    sampler_name = "nested_sampler"
    npoints_equiv_kwargs = [
        "nlive",
        "nlives",
        "n_live_points",
        "npoints",
        "npoint",
        "Nlive",
        "num_live_points",
        "num_particles",
    ]
    walks_equiv_kwargs = ["walks", "steps", "nmcmc"]

    @staticmethod
    def reorder_loglikelihoods(
        unsorted_loglikelihoods, unsorted_samples, sorted_samples
    ):
        """Reorders the stored log-likelihood after they have been reweighted

        This creates a sorting index by matching the reweights `result.samples`
        against the raw samples, then uses this index to sort the
        loglikelihoods

        Parameters
        ==========
        sorted_samples, unsorted_samples: array-like
            Sorted and unsorted values of the samples. These should be of the
            same shape and contain the same sample values, but in different
            orders
        unsorted_loglikelihoods: array-like
            The loglikelihoods corresponding to the unsorted_samples

        Returns
        =======
        sorted_loglikelihoods: array-like
            The loglikelihoods reordered to match that of the sorted_samples


        """

        idxs = []
        for ii in range(len(unsorted_loglikelihoods)):
            idx = np.where(np.all(sorted_samples[ii] == unsorted_samples, axis=1))[0]
            if len(idx) > 1:
                logger.warning(
                    "Multiple likelihood matches found between sorted and "
                    "unsorted samples. Taking the first match."
                )
            idxs.append(idx[0])
        return unsorted_loglikelihoods[idxs]

    def log_likelihood(self, theta):
        """
        Since some nested samplers don't call the log_prior method, evaluate
        the prior constraint here.

        Parameters
        ==========
        theta: array_like
            Parameter values at which to evaluate likelihood

        Returns
        =======
        float: log_likelihood
        """
        if self.priors.evaluate_constraints(
            {key: theta[ii] for ii, key in enumerate(self.search_parameter_keys)}
        ):
            return Sampler.log_likelihood(self, theta)
        else:
            return np.nan_to_num(-np.inf)


class MCMCSampler(Sampler):
    sampler_name = "mcmc_sampler"
    nwalkers_equiv_kwargs = ["nwalker", "nwalkers", "draws", "Niter"]
    nburn_equiv_kwargs = ["burn", "nburn"]

    def print_nburn_logging_info(self):
        """Prints logging info as to how nburn was calculated"""
        if type(self.nburn) in [float, int]:
            logger.info(f"Discarding {self.nburn} steps for burn-in")
        elif self.result.max_autocorrelation_time is None:
            logger.info(
                f"Autocorrelation time not calculated, discarding "
                f"{self.nburn} steps for burn-in"
            )
        else:
            logger.info(
                f"Discarding {self.nburn} steps for burn-in, estimated from autocorr"
            )

    def calculate_autocorrelation(self, samples, c=3):
        """Uses the `emcee.autocorr` module to estimate the autocorrelation

        Parameters
        ==========
        samples: array_like
            A chain of samples.
        c: float
            The minimum number of autocorrelation times needed to trust the
            estimate (default: `3`). See `emcee.autocorr.integrated_time`.
        """
        import emcee

        try:
            self.result.max_autocorrelation_time = int(
                np.max(emcee.autocorr.integrated_time(samples, c=c))
            )
            logger.info(f"Max autocorr time = {self.result.max_autocorrelation_time}")
        except emcee.autocorr.AutocorrError as e:
            self.result.max_autocorrelation_time = None
            logger.info(f"Unable to calculate autocorr time: {e}")


class _TemporaryFileSamplerMixin:
    """
    A mixin class to handle storing sampler intermediate products in a temporary
    location. See, e.g., `this SO <https://stackoverflow.com/a/547714>` for a
    basic background on mixins.

    This class makes sure that any subclasses can seamlessly use the temporary
    file functionality.
    """

    short_name = ""

    def __init__(self, temporary_directory, **kwargs):
        super(_TemporaryFileSamplerMixin, self).__init__(**kwargs)
        try:
            from mpi4py import MPI

            using_mpi = MPI.COMM_WORLD.Get_size() > 1
        except ImportError:
            using_mpi = False

        if using_mpi and temporary_directory:
            logger.info(
                "Temporary directory incompatible with MPI, "
                "will run in original directory"
            )
        self.use_temporary_directory = temporary_directory and not using_mpi
        self._outputfiles_basename = None
        self._temporary_outputfiles_basename = None

    def _check_and_load_sampling_time_file(self):
        if os.path.exists(self.time_file_path):
            with open(self.time_file_path, "r") as time_file:
                self.total_sampling_time = float(time_file.readline())
        else:
            self.total_sampling_time = 0

    def _calculate_and_save_sampling_time(self):
        current_time = time.time()
        new_sampling_time = current_time - self.start_time
        self.total_sampling_time += new_sampling_time

        with open(self.time_file_path, "w") as time_file:
            time_file.write(str(self.total_sampling_time))

        self.start_time = current_time

    def _clean_up_run_directory(self):
        if self.use_temporary_directory:
            self._move_temporary_directory_to_proper_path()
            self.kwargs["outputfiles_basename"] = self.outputfiles_basename

    @property
    def outputfiles_basename(self):
        return self._outputfiles_basename

    @outputfiles_basename.setter
    def outputfiles_basename(self, outputfiles_basename):
        if outputfiles_basename is None:
            outputfiles_basename = f"{self.outdir}/{self.short_name}_{self.label}/"
        if not outputfiles_basename.endswith("/"):
            outputfiles_basename += "/"
        check_directory_exists_and_if_not_mkdir(self.outdir)
        self._outputfiles_basename = outputfiles_basename

    @property
    def temporary_outputfiles_basename(self):
        return self._temporary_outputfiles_basename

    @temporary_outputfiles_basename.setter
    def temporary_outputfiles_basename(self, temporary_outputfiles_basename):
        if not temporary_outputfiles_basename.endswith("/"):
            temporary_outputfiles_basename += "/"
        self._temporary_outputfiles_basename = temporary_outputfiles_basename
        if os.path.exists(self.outputfiles_basename):
            shutil.copytree(
                self.outputfiles_basename, self.temporary_outputfiles_basename
            )

    def write_current_state(self):
        self._calculate_and_save_sampling_time()
        if self.use_temporary_directory:
            self._move_temporary_directory_to_proper_path()

    def _move_temporary_directory_to_proper_path(self):
        """
        Move the temporary back to the proper path

        Anything in the proper path at this point is removed including links
        """
        self._copy_temporary_directory_contents_to_proper_path()
        shutil.rmtree(self.temporary_outputfiles_basename)

    def _copy_temporary_directory_contents_to_proper_path(self):
        """
        Copy the temporary back to the proper path.
        Do not delete the temporary directory.
        """
        logger.info(
            f"Overwriting {self.outputfiles_basename} with {self.temporary_outputfiles_basename}"
        )
        outputfiles_basename_stripped = self.outputfiles_basename.rstrip("/")
        shutil.copytree(
            self.temporary_outputfiles_basename,
            outputfiles_basename_stripped,
            dirs_exist_ok=True,
        )

    def _setup_run_directory(self):
        """
        If using a temporary directory, the output directory is moved to the
        temporary directory.
        Used for Pymultinest, and Ultranest.
        """
        check_directory_exists_and_if_not_mkdir(self.outputfiles_basename)
        if self.use_temporary_directory:
            temporary_outputfiles_basename = tempfile.TemporaryDirectory().name
            self.temporary_outputfiles_basename = temporary_outputfiles_basename

            if os.path.exists(self.outputfiles_basename):
                shutil.copytree(
                    self.outputfiles_basename,
                    self.temporary_outputfiles_basename,
                    dirs_exist_ok=True,
                )
            check_directory_exists_and_if_not_mkdir(temporary_outputfiles_basename)

            self.kwargs["outputfiles_basename"] = self.temporary_outputfiles_basename
            logger.info(f"Using temporary file {temporary_outputfiles_basename}")
        else:
            self.kwargs["outputfiles_basename"] = self.outputfiles_basename
            logger.info(f"Using output file {self.outputfiles_basename}")
        self.time_file_path = self.kwargs["outputfiles_basename"] + "/sampling_time.dat"


class Error(Exception):
    """Base class for all exceptions raised by this module"""


class SamplerError(Error):
    """Base class for Error related to samplers in this module"""


class ResumeError(Error):
    """Class for errors arising from resuming runs"""


class SamplerNotInstalledError(SamplerError):
    """Base class for Error raised by not installed samplers"""


class IllegalSamplingSetError(Error):
    """Class for illegal sets of sampling parameters"""


class SamplingMarginalisedParameterError(IllegalSamplingSetError):
    """Class for errors that occur when sampling over marginalized parameters"""


class LikePriorEvaluator:
    """
    This class is copied and modified from ptemcee.LikePriorEvaluator, see
    https://github.com/willvousden/ptemcee for the original version

    We overwrite the logl and logp methods in order to improve the performance
    when using a MultiPool object: essentially reducing the amount of data
    transfer overhead.
    """

    def __init__(self):
        self.periodic_set = False

    def _setup_periodic(self):
        priors = sampling_convenience_dump.priors
        search_parameter_keys = sampling_convenience_dump.search_parameter_keys
        self._periodic = [
            priors[key].boundary == "periodic" for key in search_parameter_keys
        ]
        priors.sample()
        self._minima = np.array([priors[key].minimum for key in search_parameter_keys])
        self._range = (
            np.array([priors[key].maximum for key in search_parameter_keys])
            - self._minima
        )
        self.periodic_set = True

    def _wrap_periodic(self, array):
        if not self.periodic_set:
            self._setup_periodic()
        array[self._periodic] = (
            np.mod(
                array[self._periodic] - self._minima[self._periodic],
                self._range[self._periodic],
            )
            + self._minima[self._periodic]
        )
        return array

    def logl(self, v_array):
        priors = sampling_convenience_dump.priors
        likelihood = sampling_convenience_dump.likelihood
        search_parameter_keys = sampling_convenience_dump.search_parameter_keys
        parameters = sampling_convenience_dump.parameters.copy()
        parameters.update({key: v for key, v in zip(search_parameter_keys, v_array)})
        if priors.evaluate_constraints(parameters) == 0:
            return np.nan_to_num(-np.inf)
        elif sampling_convenience_dump.use_ratio:
            return likelihood.log_likelihood_ratio(parameters)
        else:
            return likelihood.log_likelihood(parameters)

    def logp(self, v_array):
        priors = sampling_convenience_dump.priors
        search_parameter_keys = sampling_convenience_dump.search_parameter_keys
        params = {key: t for key, t in zip(search_parameter_keys, v_array)}
        return priors.ln_prob(params)

    def call_emcee(self, theta):
        ll, lp = self.__call__(theta)
        return ll + lp, [ll, lp]

    def __call__(self, x):
        lp = self.logp(x)
        if np.isnan(lp):
            raise ValueError("Prior function returned NaN.")

        if lp == float("-inf"):
            # Can't return -inf, since this messes with beta=0 behaviour.
            ll = 0
        else:
            ll = self.logl(x)
            if np.isnan(ll).any():
                raise ValueError("Log likelihood function returned NaN.")

        return ll, lp
