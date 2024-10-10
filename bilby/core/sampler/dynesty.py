import datetime
import inspect
import os
import sys
import time
import warnings

import numpy as np
from pandas import DataFrame

from ..result import rejection_sample
from ..utils import (
    check_directory_exists_and_if_not_mkdir,
    latex_plot_format,
    logger,
    safe_file_dump,
)
from .base_sampler import NestedSampler, Sampler, _SamplingContainer, signal_wrapper


def _set_sampling_kwargs(args):
    nact, maxmcmc, proposals, naccept = args
    _SamplingContainer.nact = nact
    _SamplingContainer.maxmcmc = maxmcmc
    _SamplingContainer.proposals = proposals
    _SamplingContainer.naccept = naccept


def _prior_transform_wrapper(theta):
    """Wrapper to the prior transformation. Needed for multiprocessing."""
    from .base_sampler import _sampling_convenience_dump

    return _sampling_convenience_dump.priors.rescale(
        _sampling_convenience_dump.search_parameter_keys, theta
    )


def _log_likelihood_wrapper(theta):
    """Wrapper to the log likelihood. Needed for multiprocessing."""
    from .base_sampler import _sampling_convenience_dump

    if _sampling_convenience_dump.priors.evaluate_constraints(
        {
            key: theta[ii]
            for ii, key in enumerate(_sampling_convenience_dump.search_parameter_keys)
        }
    ):
        params = {
            key: t
            for key, t in zip(_sampling_convenience_dump.search_parameter_keys, theta)
        }
        _sampling_convenience_dump.likelihood.parameters.update(params)
        if _sampling_convenience_dump.use_ratio:
            return _sampling_convenience_dump.likelihood.log_likelihood_ratio()
        else:
            return _sampling_convenience_dump.likelihood.log_likelihood()
    else:
        return np.nan_to_num(-np.inf)


class Dynesty(NestedSampler):
    """
    bilby wrapper of `dynesty.NestedSampler`
    (https://dynesty.readthedocs.io/en/latest/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `dynesty.NestedSampler`, see
    documentation for that class for further help. Under Other Parameters below,
    we list commonly used kwargs and the Bilby defaults.

    Parameters
    ==========
    likelihood: likelihood.Likelihood
        A  object with a log_l method
    priors: bilby.core.prior.PriorDict, dict
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    outdir: str, optional
        Name of the output directory
    label: str, optional
        Naming scheme of the output files
    use_ratio: bool, optional
        Switch to set whether or not you want to use the log-likelihood ratio
        or just the log-likelihood
    plot: bool, optional
        Switch to set whether or not you want to create traceplots
    skip_import_verification: bool
        Skips the check if the sampler is installed if true. This is
        only advisable for testing environments
    print_method: str ('tqdm')
        The method to use for printing. The options are:
        - 'tqdm': use a `tqdm` `pbar`, this is the default.
        - 'interval-$TIME': print to `stdout` every `$TIME` seconds,
          e.g., 'interval-10' prints every ten seconds, this does not print every iteration
        - else: print to `stdout` at every iteration
    exit_code: int
        The code which the same exits on if it hasn't finished sampling
    check_point: bool,
        If true, use check pointing.
    check_point_plot: bool,
        If true, generate a trace plot along with the check-point
    check_point_delta_t: float (600)
        The minimum checkpoint period (in seconds). Should the run be
        interrupted, it can be resumed from the last checkpoint.
    n_check_point: int, optional (None)
        The number of steps to take before checking whether to check_point.
    resume: bool
        If true, resume run from checkpoint (if available)
    maxmcmc: int (5000)
        The maximum length of the MCMC exploration to find a new point
    nact: int (2)
        The number of autocorrelation lengths for MCMC exploration.
        For use with the :code:`act-walk` and :code:`rwalk` sample methods.
        See the dynesty guide in the Bilby docs for more details.
    naccept: int (60)
        The expected number of accepted steps for MCMC exploration when using
        the :code:`acceptance-walk` sampling method.
    rejection_sample_posterior: bool (True)
        Whether to form the posterior by rejection sampling the nested samples.
        If False, the nested samples are resampled with repetition. This was
        the default behaviour in :code:`Bilby<=1.4.1` and leads to
        non-independent samples being produced.
    proposals: iterable (None)
        The proposal methods to use during MCMC. This can be some combination
        of :code:`"diff", "volumetric"`. See the dynesty guide in the Bilby docs
        for more details. default=:code:`["diff"]`.
    rstate: numpy.random.Generator (None)
        Instance of a numpy random generator for generating random numbers.
        Also see :code:`seed` in 'Other Parameters'.

    Other Parameters
    ================
    nlive: int, (1000)
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points, npoints]
    bound: {'live', 'live-multi', 'none', 'single', 'multi', 'balls', 'cubes'}, ('live')
        Method used to select new points
    sample: {'act-walk', 'acceptance-walk', 'unif', 'rwalk', 'slice',
             'rslice', 'hslice', 'rwalk_dynesty'}, ('act-walk')
        Method used to sample uniformly within the likelihood constraints,
        conditioned on the provided bounds
    walks: int (100)
        Number of walks taken if using the dynesty implemented sample methods
        Note that the default `walks` in dynesty itself is 25, although using
        `ndim * 10` can be a reasonable rule of thumb for new problems.
        For :code:`sample="act-walk"` and :code:`sample="rwalk"` this parameter
        has no impact on the sampling.
    dlogz: float, (0.1)
        Stopping criteria
    seed: int (None)
        Use to seed the random number generator if :code:`rstate` is not
        specified.
    """

    sampler_name = "dynesty"
    sampling_seed_key = "seed"

    @property
    def _dynesty_init_kwargs(self):
        params = inspect.signature(self.sampler_init).parameters
        kwargs = {
            key: param.default
            for key, param in params.items()
            if param.default != param.empty
        }
        kwargs["sample"] = "act-walk"
        kwargs["bound"] = "live"
        kwargs["update_interval"] = 600
        kwargs["facc"] = 0.2
        return kwargs

    @property
    def _dynesty_sampler_kwargs(self):
        params = inspect.signature(self.sampler_class.run_nested).parameters
        kwargs = {
            key: param.default
            for key, param in params.items()
            if param.default != param.empty
        }
        kwargs["save_bounds"] = False
        if "dlogz" in kwargs:
            kwargs["dlogz"] = 0.1
        return kwargs

    @property
    def default_kwargs(self):
        kwargs = self._dynesty_init_kwargs
        kwargs.update(self._dynesty_sampler_kwargs)
        kwargs["seed"] = None
        return kwargs

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        use_ratio=False,
        plot=False,
        skip_import_verification=False,
        check_point=True,
        check_point_plot=True,
        n_check_point=None,
        check_point_delta_t=600,
        resume=True,
        nestcheck=False,
        exit_code=130,
        print_method="tqdm",
        maxmcmc=5000,
        nact=2,
        naccept=60,
        rejection_sample_posterior=True,
        proposals=None,
        **kwargs,
    ):
        self.nact = nact
        self.naccept = naccept
        self.maxmcmc = maxmcmc
        self.proposals = proposals
        self.print_method = print_method
        self._translate_kwargs(kwargs)
        super(Dynesty, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            exit_code=exit_code,
            **kwargs,
        )
        self.n_check_point = n_check_point
        self.check_point = check_point
        self.check_point_plot = check_point_plot
        self.resume = resume
        self.rejection_sample_posterior = rejection_sample_posterior
        self._apply_dynesty_boundaries("periodic")
        self._apply_dynesty_boundaries("reflective")

        self.nestcheck = nestcheck

        if self.n_check_point is None:
            self.n_check_point = (
                10
                if np.isnan(self._log_likelihood_eval_time)
                else max(
                    int(check_point_delta_t / self._log_likelihood_eval_time / 10), 10
                )
            )
        self.check_point_delta_t = check_point_delta_t
        logger.info(f"Checkpoint every check_point_delta_t = {check_point_delta_t}s")

        self.resume_file = f"{self.outdir}/{self.label}_resume.pickle"
        self.sampling_time = datetime.timedelta()
        self.pbar = None

    @property
    def sampler_function_kwargs(self):
        return {key: self.kwargs[key] for key in self._dynesty_sampler_kwargs}

    @property
    def sampler_init_kwargs(self):
        return {key: self.kwargs[key] for key in self._dynesty_init_kwargs}

    def _translate_kwargs(self, kwargs):
        kwargs = super()._translate_kwargs(kwargs)
        if "nlive" not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["nlive"] = kwargs.pop(equiv)
        if "print_progress" not in kwargs:
            if "verbose" in kwargs:
                kwargs["print_progress"] = kwargs.pop("verbose")
        if "walks" not in kwargs:
            for equiv in self.walks_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["walks"] = kwargs.pop(equiv)
        if "queue_size" not in kwargs:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["queue_size"] = kwargs.pop(equiv)
        if "seed" in kwargs:
            seed = kwargs.get("seed")
            if "rstate" not in kwargs:
                kwargs["rstate"] = np.random.default_rng(seed)
            else:
                logger.warning(
                    "Kwargs contain both 'rstate' and 'seed', ignoring 'seed'."
                )

    def _verify_kwargs_against_default_kwargs(self):
        if not self.kwargs["walks"]:
            self.kwargs["walks"] = 100
        if self.kwargs["print_func"] is None:
            self.kwargs["print_func"] = self._print_func
            if "interval" in self.print_method:
                self._last_print_time = datetime.datetime.now()
                self._print_interval = datetime.timedelta(
                    seconds=float(self.print_method.split("-")[1])
                )
        Sampler._verify_kwargs_against_default_kwargs(self)

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        """Get lists of the expected outputs directories and files.

        These are used by :code:`bilby_pipe` when transferring files via HTCondor.

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
            List of directory names. Will always be empty for dynesty.
        """
        filenames = []
        for kind in ["resume", "dynesty"]:
            filename = os.path.join(outdir, f"{label}_{kind}.pickle")
            filenames.append(filename)
        return filenames, []

    def _print_func(
        self,
        results,
        niter,
        ncall=None,
        dlogz=None,
        stop_val=None,
        nbatch=None,
        logl_min=-np.inf,
        logl_max=np.inf,
        *args,
        **kwargs,
    ):
        """Replacing status update for dynesty.result.print_func"""
        if "interval" in self.print_method:
            _time = datetime.datetime.now()
            if _time - self._last_print_time < self._print_interval:
                return
            else:
                self._last_print_time = _time

                # Add time in current run to overall sampling time
                total_time = self.sampling_time + _time - self.start_time

                # Remove fractional seconds
                total_time_str = str(total_time).split(".")[0]

        # Extract results at the current iteration.
        loglstar = results.loglstar
        delta_logz = results.delta_logz
        logz = results.logz
        logzvar = results.logzvar
        nc = results.nc
        bounditer = results.bounditer
        eff = results.eff

        # Adjusting outputs for printing.
        if delta_logz > 1e6:
            delta_logz = np.inf
        if 0.0 <= logzvar <= 1e6:
            logzerr = np.sqrt(logzvar)
        else:
            logzerr = np.nan
        if logz <= -1e6:
            logz = -np.inf
        if loglstar <= -1e6:
            loglstar = -np.inf

        if self.use_ratio:
            key = "logz-ratio"
        else:
            key = "logz"

        # Constructing output.
        string = list()
        string.append(f"bound:{bounditer:d}")
        string.append(f"nc:{nc:3d}")
        string.append(f"ncall:{ncall:.1e}")
        string.append(f"eff:{eff:0.1f}%")
        string.append(f"{key}={logz:0.2f}+/-{logzerr:0.2f}")
        if nbatch is not None:
            string.append(f"batch:{nbatch}")
        if logl_min > -np.inf:
            string.append(f"logl:{logl_min:.1f} < {loglstar:.1f} < {logl_max:.1f}")
        if dlogz is not None:
            string.append(f"dlogz:{delta_logz:0.3f}>{dlogz:0.2g}")
        else:
            string.append(f"stop:{stop_val:6.3f}")
        string = " ".join(string)

        if self.print_method == "tqdm":
            self.pbar.set_postfix_str(string, refresh=False)
            self.pbar.update(niter - self.pbar.n)
        else:
            print(f"{niter}it [{total_time_str} {string}]", file=sys.stdout, flush=True)

    def _apply_dynesty_boundaries(self, key):
        # The periodic kwargs passed into dynesty allows the parameters to
        # wander out of the bounds, this includes both periodic and reflective.
        # these are then handled in the prior_transform
        selected = list()
        for ii, param in enumerate(self.search_parameter_keys):
            if self.priors[param].boundary == key:
                logger.debug(f"Setting {key} boundary for {param}")
                selected.append(ii)
        if len(selected) == 0:
            selected = None
        self.kwargs[key] = selected

    def nestcheck_data(self, out_file):
        import nestcheck.data_processing

        ns_run = nestcheck.data_processing.process_dynesty_run(out_file)
        nestcheck_result = f"{self.outdir}/{self.label}_nestcheck.pickle"
        safe_file_dump(ns_run, nestcheck_result, "pickle")

    @property
    def nlive(self):
        return self.kwargs["nlive"]

    @property
    def sampler_init(self):
        from dynesty import NestedSampler

        return NestedSampler

    @property
    def sampler_class(self):
        from dynesty.sampler import Sampler

        return Sampler

    def _set_sampling_method(self):
        """
        Resolve the sampling method and sampler to use from the provided
        :code:`bound` and :code:`sample` arguments.

        This requires registering the :code:`bilby` specific methods in the
        appropriate locations within :code:`dynesty`.

        Additionally, some combinations of bound/sample/proposals are not
        compatible and so we either warn the user or raise an error.
        """
        import dynesty

        _set_sampling_kwargs((self.nact, self.maxmcmc, self.proposals, self.naccept))

        sample = self.kwargs["sample"]
        bound = self.kwargs["bound"]

        if sample not in ["rwalk", "act-walk", "acceptance-walk"] and bound in [
            "live",
            "live-multi",
        ]:
            logger.info(
                "Live-point based bound method requested with dynesty sample "
                f"'{sample}', overwriting to 'multi'"
            )
            self.kwargs["bound"] = "multi"
        elif bound == "live":
            from .dynesty_utils import LivePointSampler

            dynesty.dynamicsampler._SAMPLERS["live"] = LivePointSampler
        elif bound == "live-multi":
            from .dynesty_utils import MultiEllipsoidLivePointSampler

            dynesty.dynamicsampler._SAMPLERS[
                "live-multi"
            ] = MultiEllipsoidLivePointSampler
        elif sample == "acceptance-walk":
            raise DynestySetupError(
                "bound must be set to live or live-multi for sample=acceptance-walk"
            )
        elif self.proposals is None:
            logger.warning(
                "No proposals specified using dynesty sampling, defaulting "
                "to 'volumetric'."
            )
            self.proposals = ["volumetric"]
            _SamplingContainer.proposals = self.proposals
        elif "diff" in self.proposals:
            raise DynestySetupError(
                "bound must be set to live or live-multi to use differential "
                "evolution proposals"
            )

        if sample == "rwalk":
            logger.info(
                f"Using the bilby-implemented {sample} sample method with ACT estimated walks. "
                f"An average of {2 * self.nact} steps will be accepted up to chain length "
                f"{self.maxmcmc}."
            )
            from .dynesty_utils import AcceptanceTrackingRWalk

            if self.kwargs["walks"] > self.maxmcmc:
                raise DynestySetupError("You have maxmcmc < walks (minimum mcmc)")
            if self.nact < 1:
                raise DynestySetupError("Unable to run with nact < 1")
            AcceptanceTrackingRWalk.old_act = None
            dynesty.nestedsamplers._SAMPLING["rwalk"] = AcceptanceTrackingRWalk()
        elif sample == "acceptance-walk":
            logger.info(
                f"Using the bilby-implemented {sample} sampling with an average of "
                f"{self.naccept} accepted steps per MCMC and maximum length {self.maxmcmc}"
            )
            from .dynesty_utils import FixedRWalk

            dynesty.nestedsamplers._SAMPLING["acceptance-walk"] = FixedRWalk()
        elif sample == "act-walk":
            logger.info(
                f"Using the bilby-implemented {sample} sampling tracking the "
                f"autocorrelation function and thinning by "
                f"{self.nact} with maximum length {self.nact * self.maxmcmc}"
            )
            from .dynesty_utils import ACTTrackingRWalk

            ACTTrackingRWalk._cache = list()
            dynesty.nestedsamplers._SAMPLING["act-walk"] = ACTTrackingRWalk()
        elif sample == "rwalk_dynesty":
            sample = sample.strip("_dynesty")
            self.kwargs["sample"] = sample
            logger.info(f"Using the dynesty-implemented {sample} sample method")

    @signal_wrapper
    def run_sampler(self):
        import dynesty

        logger.info(f"Using dynesty version {dynesty.__version__}")

        self._set_sampling_method()
        self._setup_pool()

        if self.resume:
            self.resume = self.read_saved_state(continuing=True)

        if self.resume:
            logger.info("Resume file successfully loaded.")
        else:
            if self.kwargs["live_points"] is None:
                self.kwargs["live_points"] = self.get_initial_points_from_prior(
                    self.nlive
                )
                self.kwargs["live_points"] = (*self.kwargs["live_points"], None)
            self.sampler = self.sampler_init(
                loglikelihood=_log_likelihood_wrapper,
                prior_transform=_prior_transform_wrapper,
                ndim=self.ndim,
                **self.sampler_init_kwargs,
            )
        if self.print_method == "tqdm" and self.kwargs["print_progress"]:
            from tqdm.auto import tqdm

            self.pbar = tqdm(file=sys.stdout, initial=self.sampler.it)

        self.start_time = datetime.datetime.now()
        if self.check_point:
            out = self._run_external_sampler_with_checkpointing()
        else:
            out = self._run_external_sampler_without_checkpointing()
        self._update_sampling_time()

        self._close_pool()

        # Flushes the output to force a line break
        if self.pbar is not None:
            self.pbar = self.pbar.close()
            print("")

        check_directory_exists_and_if_not_mkdir(self.outdir)

        if self.nestcheck:
            self.nestcheck_data(out)

        dynesty_result = f"{self.outdir}/{self.label}_dynesty.pickle"
        safe_file_dump(out, dynesty_result, "dill")

        self._generate_result(out)
        self.result.sampling_time = self.sampling_time

        return self.result

    def _setup_pool(self):
        """
        In addition to the usual steps, we need to set the sampling kwargs on
        every process. To make sure we get every process, run the kwarg setting
        more times than we have processes.
        """
        super(Dynesty, self)._setup_pool()
        if self.pool is not None:
            args = (
                [(self.nact, self.maxmcmc, self.proposals, self.naccept)]
                * self.npool
                * 10
            )
            self.pool.map(_set_sampling_kwargs, args)

    def _generate_result(self, out):
        """
        Extract the information we need from the dynesty output. This includes
        the evidence, nested samples, run statistics. In addition, we generate
        the posterior samples from the nested samples.

        Parameters
        ==========
        out: dynesty.result.Result
            The dynesty output.
        """
        import dynesty
        from scipy.special import logsumexp

        from ..utils.random import rng

        logwts = out["logwt"]
        weights = np.exp(logwts - out["logz"][-1])
        nested_samples = DataFrame(out.samples, columns=self.search_parameter_keys)
        nested_samples["weights"] = weights
        nested_samples["log_likelihood"] = out.logl
        self.result.nested_samples = nested_samples
        if self.rejection_sample_posterior:
            keep = weights > rng.uniform(0, max(weights), len(weights))
            self.result.samples = out.samples[keep]
            self.result.log_likelihood_evaluations = out.logl[keep]
            logger.info(
                f"Rejection sampling nested samples to obtain {sum(keep)} posterior samples"
            )
        else:
            self.result.samples = dynesty.utils.resample_equal(out.samples, weights)
            self.result.log_likelihood_evaluations = self.reorder_loglikelihoods(
                unsorted_loglikelihoods=out.logl,
                unsorted_samples=out.samples,
                sorted_samples=self.result.samples,
            )
            logger.info("Resampling nested samples to posterior samples in place.")
        self.result.log_evidence = out.logz[-1]
        self.result.log_evidence_err = out.logzerr[-1]
        self.result.information_gain = out.information[-1]
        self.result.num_likelihood_evaluations = getattr(self.sampler, "ncall", 0)

        logneff = logsumexp(logwts) * 2 - logsumexp(logwts * 2)
        neffsamples = int(np.exp(logneff))
        self.result.meta_data["run_statistics"] = dict(
            nlikelihood=self.result.num_likelihood_evaluations,
            neffsamples=neffsamples,
            sampling_time_s=self.sampling_time.seconds,
            ncores=self.kwargs.get("queue_size", 1),
        )
        self.kwargs["rstate"] = None

    def _update_sampling_time(self):
        end_time = datetime.datetime.now()
        self.sampling_time += end_time - self.start_time
        self.start_time = end_time

    def _run_external_sampler_without_checkpointing(self):
        logger.debug("Running sampler without checkpointing")
        self.sampler.run_nested(**self.sampler_function_kwargs)
        return self.sampler.results

    def finalize_sampler_kwargs(self, sampler_kwargs):
        sampler_kwargs["maxcall"] = self.n_check_point
        sampler_kwargs["add_live"] = True

    def _run_external_sampler_with_checkpointing(self):
        """
        In order to access the checkpointing, we run the sampler for short
        periods of time (less than the checkpoint time) and if sufficient
        time has passed, write a checkpoint before continuing. To get the most
        informative checkpoint plots, the current live points are added to the
        chain of nested samples within dynesty and have to be removed before
        restarting the sampler.
        """

        logger.debug("Running sampler with checkpointing")

        old_ncall = self.sampler.ncall
        sampler_kwargs = self.sampler_function_kwargs.copy()
        warnings.filterwarnings(
            "ignore",
            message="The sampling was stopped short due to maxiter/maxcall limit*",
            category=UserWarning,
            module="dynesty.sampler",
        )
        while True:
            self.finalize_sampler_kwargs(sampler_kwargs)
            if getattr(self.sampler, "added_live", False):
                self.sampler._remove_live_points()
            self.sampler.run_nested(**sampler_kwargs)
            if self.sampler.ncall == old_ncall:
                break
            old_ncall = self.sampler.ncall

            if os.path.isfile(self.resume_file):
                last_checkpoint_s = time.time() - os.path.getmtime(self.resume_file)
            else:
                last_checkpoint_s = (
                    datetime.datetime.now() - self.start_time
                ).total_seconds()
            if last_checkpoint_s > self.check_point_delta_t:
                self.write_current_state()
                self.plot_current_state()
        if getattr(self.sampler, "added_live", False):
            self.sampler._remove_live_points()

        self.sampler.run_nested(**sampler_kwargs)
        self.write_current_state()
        self.plot_current_state()
        return self.sampler.results

    def _remove_checkpoint(self):
        """Remove checkpointed state"""
        if os.path.isfile(self.resume_file):
            os.remove(self.resume_file)

    def read_saved_state(self, continuing=False):
        """
        Read a pickled saved state of the sampler to disk.

        If the live points are present and the run is continuing
        they are removed.
        The random state must be reset, as this isn't saved by the pickle.
        `nqueue` is set to a negative number to trigger the queue to be
        refilled before the first iteration.
        The previous run time is set to self.

        Parameters
        ==========
        continuing: bool
            Whether the run is continuing or terminating, if True, the loaded
            state is mostly written back to disk.
        """
        import dill
        from dynesty import __version__ as dynesty_version

        from ... import __version__ as bilby_version

        versions = dict(bilby=bilby_version, dynesty=dynesty_version)
        if os.path.isfile(self.resume_file):
            logger.info(f"Reading resume file {self.resume_file}")
            with open(self.resume_file, "rb") as file:
                try:
                    sampler = dill.load(file)
                except EOFError:
                    sampler = None

                if not hasattr(sampler, "versions"):
                    logger.warning(
                        f"The resume file {self.resume_file} is corrupted or "
                        "the version of bilby has changed between runs. This "
                        "resume file will be ignored."
                    )
                    return False
                version_warning = (
                    "The {code} version has changed between runs. "
                    "This may cause unpredictable behaviour and/or failure. "
                    "Old version = {old}, new version = {new}."
                )
                for code in versions:
                    if not versions[code] == sampler.versions.get(code, None):
                        logger.warning(
                            version_warning.format(
                                code=code,
                                old=sampler.versions.get(code, "None"),
                                new=versions[code],
                            )
                        )
                del sampler.versions
                self.sampler = sampler
                if getattr(self.sampler, "added_live", False) and continuing:
                    self.sampler._remove_live_points()
                self.sampler.nqueue = -1
                self.start_time = self.sampler.kwargs.pop("start_time")
                self.sampling_time = self.sampler.kwargs.pop("sampling_time")
                self.sampler.queue_size = self.kwargs["queue_size"]
                self.sampler.pool = self.pool
                if self.pool is not None:
                    self.sampler.M = self.pool.map
                else:
                    self.sampler.M = map
            return True
        else:
            logger.info(f"Resume file {self.resume_file} does not exist.")
            return False

    def write_current_state_and_exit(self, signum=None, frame=None):
        if self.pbar is not None:
            self.pbar = self.pbar.close()
        super(Dynesty, self).write_current_state_and_exit(signum=signum, frame=frame)

    def write_current_state(self):
        """
        Write the current state of the sampler to disk.

        The sampler is pickle dumped using `dill`.
        The sampling time is also stored to get the full CPU time for the run.

        The check of whether the sampler is picklable is to catch an error
        when using pytest. Hopefully, this message won't be triggered during
        normal running.
        """

        import dill
        from dynesty import __version__ as dynesty_version

        from ... import __version__ as bilby_version

        if getattr(self, "sampler", None) is None:
            # Sampler not initialized, not able to write current state
            return

        check_directory_exists_and_if_not_mkdir(self.outdir)
        if hasattr(self, "start_time"):
            self._update_sampling_time()
            self.sampler.kwargs["sampling_time"] = self.sampling_time
            self.sampler.kwargs["start_time"] = self.start_time
        self.sampler.versions = dict(bilby=bilby_version, dynesty=dynesty_version)
        self.sampler.pool = None
        self.sampler.M = map
        if dill.pickles(self.sampler):
            safe_file_dump(self.sampler, self.resume_file, dill)
            logger.info(f"Written checkpoint file {self.resume_file}")
        else:
            logger.warning(
                "Cannot write pickle resume file! "
                "Job will not resume if interrupted."
            )
        self.sampler.pool = self.pool
        if self.sampler.pool is not None:
            self.sampler.M = self.sampler.pool.map

    def dump_samples_to_dat(self):
        """
        Save the current posterior samples to a space-separated plain-text
        file. These are unbiased posterior samples, however, there will not
        be many of them until the analysis is nearly over.
        """
        sampler = self.sampler
        ln_weights = sampler.saved_logwt - sampler.saved_logz[-1]

        weights = np.exp(ln_weights)
        samples = rejection_sample(np.array(sampler.saved_v), weights)
        nsamples = len(samples)

        # If we don't have enough samples, don't dump them
        if nsamples < 100:
            return

        filename = f"{self.outdir}/{self.label}_samples.dat"
        logger.info(f"Writing {nsamples} current samples to {filename}")

        df = DataFrame(samples, columns=self.search_parameter_keys)
        df.to_csv(filename, index=False, header=True, sep=" ")

    def plot_current_state(self):
        """
        Make diagonstic plots of the history and current state of the sampler.

        These plots are a mixture of :code:`dynesty` implemented run and trace
        plots and our custom stats plot. We also make a copy of the trace plot
        using the unit hypercube samples to reflect the internal state of the
        sampler.

        Any errors during plotting should be handled so that sampling can
        continue.
        """
        if self.check_point_plot:
            import dynesty.plotting as dyplot
            import matplotlib.pyplot as plt

            labels = [label.replace("_", " ") for label in self.search_parameter_keys]
            try:
                filename = f"{self.outdir}/{self.label}_checkpoint_trace.png"
                fig = dyplot.traceplot(self.sampler.results, labels=labels)[0]
                fig.tight_layout()
                fig.savefig(filename)
            except (
                RuntimeError,
                np.linalg.linalg.LinAlgError,
                ValueError,
                OverflowError,
            ) as e:
                logger.warning(e)
                logger.warning("Failed to create dynesty state plot at checkpoint")
            except Exception as e:
                logger.warning(
                    f"Unexpected error {e} in dynesty plotting. "
                    "Please report at github.com/bilby-dev/bilby/issues"
                )
            finally:
                plt.close("all")
            try:
                filename = f"{self.outdir}/{self.label}_checkpoint_trace_unit.png"
                from copy import deepcopy

                from dynesty.utils import results_substitute

                temp = deepcopy(self.sampler.results)
                temp = results_substitute(temp, dict(samples=temp["samples_u"]))
                fig = dyplot.traceplot(temp, labels=labels)[0]
                fig.tight_layout()
                fig.savefig(filename)
            except (
                RuntimeError,
                np.linalg.linalg.LinAlgError,
                ValueError,
                OverflowError,
            ) as e:
                logger.warning(e)
                logger.warning("Failed to create dynesty unit state plot at checkpoint")
            except Exception as e:
                logger.warning(
                    f"Unexpected error {e} in dynesty plotting. "
                    "Please report at github.com/bilby-dev/bilby/issues"
                )
            finally:
                plt.close("all")
            try:
                filename = f"{self.outdir}/{self.label}_checkpoint_run.png"
                fig, _ = dyplot.runplot(
                    self.sampler.results, logplot=False, use_math_text=False
                )
                fig.tight_layout()
                plt.savefig(filename)
            except (
                RuntimeError,
                np.linalg.linalg.LinAlgError,
                ValueError,
                OverflowError,
            ) as e:
                logger.warning(e)
                logger.warning("Failed to create dynesty run plot at checkpoint")
            except Exception as e:
                logger.warning(
                    f"Unexpected error {e} in dynesty plotting. "
                    "Please report at github.com/bilby-dev/bilby/issues"
                )
            finally:
                plt.close("all")
            try:
                filename = f"{self.outdir}/{self.label}_checkpoint_stats.png"
                fig, _ = dynesty_stats_plot(self.sampler)
                fig.tight_layout()
                plt.savefig(filename)
            except (RuntimeError, ValueError, OverflowError) as e:
                logger.warning(e)
                logger.warning("Failed to create dynesty stats plot at checkpoint")
            except DynestySetupError:
                logger.debug("Cannot create Dynesty stats plot with dynamic sampler.")
            except Exception as e:
                logger.warning(
                    f"Unexpected error {e} in dynesty plotting. "
                    "Please report at github.com/bilby-dev/bilby/issues"
                )
            finally:
                plt.close("all")

    def _run_test(self):
        """Run the sampler very briefly as a sanity test that it works."""
        import pandas as pd

        self._set_sampling_method()
        self._setup_pool()
        self.sampler = self.sampler_init(
            loglikelihood=_log_likelihood_wrapper,
            prior_transform=_prior_transform_wrapper,
            ndim=self.ndim,
            **self.sampler_init_kwargs,
        )
        sampler_kwargs = self.sampler_function_kwargs.copy()
        sampler_kwargs["maxiter"] = 2

        if self.print_method == "tqdm" and self.kwargs["print_progress"]:
            from tqdm.auto import tqdm

            self.pbar = tqdm(file=sys.stdout, initial=self.sampler.it)
        self.sampler.run_nested(**sampler_kwargs)
        self._close_pool()

        if self.pbar is not None:
            self.pbar = self.pbar.close()
            print("")
        N = 100
        self.result.samples = pd.DataFrame(self.priors.sample(N))[
            self.search_parameter_keys
        ].values
        self.result.nested_samples = self.result.samples
        self.result.log_likelihood_evaluations = np.ones(N)
        self.result.log_evidence = 1
        self.result.log_evidence_err = 0.1

        return self.result

    def prior_transform(self, theta):
        """Prior transform method that is passed into the external sampler.
        cube we map this back to [0, 1].

        Parameters
        ==========
        theta: list
            List of sampled values on a unit interval

        Returns
        =======
        list: Properly rescaled sampled values

        """
        return self.priors.rescale(self._search_parameter_keys, theta)


@latex_plot_format
def dynesty_stats_plot(sampler):
    """
    Plot diagnostic statistics from a dynesty run

    The plotted quantities per iteration are:

    - nc: the number of likelihood calls
    - scale: the number of accepted MCMC steps if using :code:`bound="live"`
      or :code:`bound="live-multi"`, otherwise, the scale applied to the MCMC
      steps
    - lifetime: the number of iterations a point stays in the live set

    There is also a histogram of the lifetime compared with the theoretical
    distribution. To avoid edge effects, we discard the first 6 * nlive

    Parameters
    ----------
    sampler: dynesty.sampler.Sampler
        The sampler object containing the run history.

    Returns
    -------
    fig: matplotlib.pyplot.figure.Figure
        Figure handle for the new plot
    axs: matplotlib.pyplot.axes.Axes
        Axes handles for the new plot

    """
    import matplotlib.pyplot as plt
    from scipy.stats import geom, ks_1samp

    fig, axs = plt.subplots(nrows=4, figsize=(8, 8))
    data = sampler.saved_run.D
    for ax, name in zip(axs, ["nc", "scale"]):
        ax.plot(data[name], color="blue")
        ax.set_ylabel(name.title())
    lifetimes = np.arange(len(data["it"])) - data["it"]
    axs[-2].set_ylabel("Lifetime")
    if not hasattr(sampler, "nlive"):
        raise DynestySetupError("Cannot make stats plot for dynamic sampler.")
    nlive = sampler.nlive
    burn = int(geom(p=1 / nlive).isf(1 / 2 / nlive))
    if len(data["it"]) > burn + sampler.nlive:
        axs[-2].plot(np.arange(0, burn), lifetimes[:burn], color="grey")
        axs[-2].plot(
            np.arange(burn, len(lifetimes) - nlive),
            lifetimes[burn:-nlive],
            color="blue",
        )
        axs[-2].plot(
            np.arange(len(lifetimes) - nlive, len(lifetimes)),
            lifetimes[-nlive:],
            color="red",
        )
        lifetimes = lifetimes[burn:-nlive]
        ks_result = ks_1samp(lifetimes, geom(p=1 / nlive).cdf)
        axs[-1].hist(
            lifetimes,
            bins=np.linspace(0, 6 * nlive, 60),
            histtype="step",
            density=True,
            color="blue",
            label=f"p value = {ks_result.pvalue:.3f}",
        )
        axs[-1].plot(
            np.arange(1, 6 * nlive),
            geom(p=1 / nlive).pmf(np.arange(1, 6 * nlive)),
            color="red",
        )
        axs[-1].set_xlim(0, 6 * nlive)
        axs[-1].legend()
        axs[-1].set_yscale("log")
    else:
        axs[-2].plot(
            np.arange(0, len(lifetimes) - nlive), lifetimes[:-nlive], color="grey"
        )
        axs[-2].plot(
            np.arange(len(lifetimes) - nlive, len(lifetimes)),
            lifetimes[-nlive:],
            color="red",
        )
    axs[-2].set_yscale("log")
    axs[-2].set_xlabel("Iteration")
    axs[-1].set_xlabel("Lifetime")
    return fig, axs


class DynestySetupError(Exception):
    pass
