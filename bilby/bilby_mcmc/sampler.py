import datetime
import os
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import differential_evolution

from ..core.result import rejection_sample
from ..core.sampler.base_sampler import (
    MCMCSampler,
    ResumeError,
    SamplerError,
    signal_wrapper,
)
from ..core.utils import (
    check_directory_exists_and_if_not_mkdir,
    logger,
    random,
    safe_file_dump,
)
from ..core.utils.parallel import sampling_convenience_dump
from . import proposals
from .chain import Chain, Sample
from .utils import LOGLKEY, LOGPKEY, ConvergenceInputs, ParallelTemperingInputs


class Bilby_MCMC(MCMCSampler):
    """The built-in Bilby MCMC sampler

    Parameters
    ----------
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
    skip_import_verification: bool
        Skips the check if the sampler is installed if true. This is
        only advisable for testing environments
    check_point_plot: bool
        If true, create plots at the check point
    check_point_delta_t: float
        The time in seconds afterwhich to checkpoint (defaults to 30 minutes)
    diagnostic: bool
        If true, create deep-diagnostic plots used for checking convergence
        problems.
    resume: bool
        If true, resume from any existing check point files
    exit_code: int
        The code on which to raise if exiting
    nsamples: int (1000)
        The number of samples to draw
    nensemble: int (1)
        The number of ensemble-chains to run (with periodic communication)
    pt_ensemble: bool (False)
        If true, each run a parallel-tempered set of chains for each
        ensemble-chain (in which case the total number of chains is
        nensemble * ntemps). Else, only the zero-ensemble chain is run with a
        parallel-tempering (in which case the total number of chains is
        nensemble + ntemps - 1).
    ntemps: int (1)
        The number of parallel-tempered chains to run
    Tmax: float, (None)
        If given, the maximum temperature to set the initial temperate-ladder
    Tmax_from_SNR: float (20)
        (Alternative to Tmax): The SNR to estimate an appropriate Tmax from.
    initial_betas: list (None)
        (Alternative to Tmax and Tmax_from_SNR): If given, an initial choice of
        the inverse temperature ladder.
    pt_rejection_sample: bool (False)
        If true, use rejection sampling to draw samples from the pt-chains.
    adapt, adapt_t0, adapt_nu: bool, float, float (True, 100, 10)
        Whether to use adaptation and the adaptation parameters.
        See arXiv:1501.05823 for a description of adapt_t0 and adapt_nu.
    burn_in_nact, thin_by_nact, fixed_discard: float, float, float (10, 1, 0)
        The number of auto-correlation times to discard for burn-in and to
        thin by. The fixed_discard is the number of steps discarded before
        automatic autocorrelation time analysis begins.
    autocorr_c: float (5)
        The step-size for the window search. See emcee.autocorr.integrated_time
        for additional details.
    L1steps: int
        The number of internal steps to take. Improves the scaling performance
        of multiprocessing. Note, all ACTs are calculated based on the saved
        steps. So, the total ACT (or number of steps) is L1steps * tau
        (or L1steps * position).
    L2steps: int
        The number of steps to take before swapping between parallel-tempered
        and ensemble chains.
    npool: int
        The number of multiprocessing cores to use. For efficiency, this must be
        matched to an integer number of the total number of chains.
    printdt: float
        Print an update on the progress every printdt s. Note, each print
        requires an evaluation of the ACT so short print times are unwise.
    min_tau: 1
        The minimum allowed ACT. Can be used to force a larger ACT.
    proposal_cycle: str, bilby.core.sampler.bilby_mcmc.proposals.ProposalCycle
        Either a string pointing to one of the built-in proposal cycles or,
        a proposal cycle.
    stop_after_convergence:
        If running with parallel-tempered chains. Stop updating the chains once
        they have congerged. After this time, random samples will be drawn at
        swap time.
    fixed_tau: int
        A fixed value for the ACT: used for testing purposes.
    tau_window: int, None
        Using tau', a previous estimates of tau, calculate the new tau using
        the last tau_window * tau' steps. If None, the entire chain is used.
    evidence_method: str, [stepping_stone, thermodynamic]
        The evidence calculation method to use. Defaults to stepping_stone, but
        the results of all available methods are stored in the ln_z_dict.
    initial_sample_method: str
        Method to draw the initial sample. Either "prior" (a random draw
        from the prior) or "maximize" (use an optimization approach to attempt
        to find the maximum posterior estimate).
    initial_sample_dict: dict
        A dictionary of the initial sample value. If incomplete, will overwrite
        the initial_sample drawn using initial_sample_method.
    normalize_prior: bool
        When False, disables calculation of constraint normalization factor
        during prior probability computation. Default value is True.
    verbose: bool
        Whether to print diagnostic output during the run.

    """

    default_kwargs = dict(
        nsamples=1000,
        nensemble=1,
        pt_ensemble=False,
        ntemps=1,
        Tmax=None,
        Tmax_from_SNR=20,
        initial_betas=None,
        adapt=True,
        adapt_t0=100,
        adapt_nu=10,
        pt_rejection_sample=False,
        burn_in_nact=10,
        thin_by_nact=1,
        fixed_discard=0,
        autocorr_c=5,
        L1steps=100,
        L2steps=3,
        printdt=60,
        check_point_delta_t=1800,
        min_tau=1,
        proposal_cycle="default",
        stop_after_convergence=False,
        fixed_tau=None,
        tau_window=None,
        evidence_method="stepping_stone",
        initial_sample_method="prior",
        initial_sample_dict=None,
    )

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        use_ratio=False,
        skip_import_verification=True,
        check_point_plot=True,
        diagnostic=False,
        resume=True,
        exit_code=130,
        verbose=True,
        normalize_prior=True,
        **kwargs,
    ):

        super(Bilby_MCMC, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            skip_import_verification=skip_import_verification,
            exit_code=exit_code,
            **kwargs,
        )

        self.check_point_plot = check_point_plot
        self.diagnostic = diagnostic
        self.kwargs["target_nsamples"] = self.kwargs["nsamples"]
        self.L1steps = self.kwargs["L1steps"]
        self.L2steps = self.kwargs["L2steps"]
        self.normalize_prior = normalize_prior
        self.pt_inputs = ParallelTemperingInputs(
            **{key: self.kwargs[key] for key in ParallelTemperingInputs._fields}
        )
        self.convergence_inputs = ConvergenceInputs(
            **{key: self.kwargs[key] for key in ConvergenceInputs._fields}
        )
        self.proposal_cycle = self.kwargs["proposal_cycle"]
        self.pt_rejection_sample = self.kwargs["pt_rejection_sample"]
        self.evidence_method = self.kwargs["evidence_method"]
        self.initial_sample_method = self.kwargs["initial_sample_method"]
        self.initial_sample_dict = self.kwargs["initial_sample_dict"]

        self.printdt = self.kwargs["printdt"]
        self.check_point_delta_t = self.kwargs["check_point_delta_t"]
        check_directory_exists_and_if_not_mkdir(self.outdir)
        self.resume = resume
        self.resume_file = "{}/{}_resume.pickle".format(self.outdir, self.label)

        self.verify_configuration()
        self.verbose = verbose

    def verify_configuration(self):
        if self.convergence_inputs.burn_in_nact / self.kwargs["target_nsamples"] > 0.1:
            logger.warning("Burn-in inefficiency fraction greater than 10%")

    def _translate_kwargs(self, kwargs):
        kwargs = super()._translate_kwargs(kwargs)
        if "printdt" not in kwargs:
            for equiv in ["print_dt", "print_update"]:
                if equiv in kwargs:
                    kwargs["printdt"] = kwargs.pop(equiv)
        if "npool" not in kwargs:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["npool"] = kwargs.pop(equiv)
        if "check_point_delta_t" not in kwargs:
            for equiv in self.check_point_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["check_point_delta_t"] = kwargs.pop(equiv)

    @property
    def target_nsamples(self):
        return self.kwargs["target_nsamples"]

    @signal_wrapper
    def run_sampler(self):
        self._setup_pool()
        self.setup_chain_set()
        self.start_time = datetime.datetime.now()
        self.draw()
        self._close_pool()
        self.check_point(ignore_time=True)

        self.result = self.add_data_to_result(
            result=self.result,
            ptsampler=self.ptsampler,
            outdir=self.outdir,
            label=self.label,
            make_plots=self.check_point_plot,
        )

        return self.result

    @staticmethod
    def add_data_to_result(result, ptsampler, outdir, label, make_plots):
        result.samples = ptsampler.samples
        result.log_likelihood_evaluations = result.samples[LOGLKEY].to_numpy()
        result.log_prior_evaluations = result.samples[LOGPKEY].to_numpy()
        ptsampler.compute_evidence(
            outdir=outdir,
            label=label,
            make_plots=make_plots,
        )
        result.log_evidence = ptsampler.ln_z
        result.log_evidence_err = ptsampler.ln_z_err
        result.sampling_time = datetime.timedelta(seconds=ptsampler.sampling_time)
        result.meta_data["bilby_mcmc"] = dict(
            tau=ptsampler.tau,
            convergence_inputs=ptsampler.convergence_inputs._asdict(),
            pt_inputs=ptsampler.pt_inputs._asdict(),
            total_steps=ptsampler.position,
            nsamples=ptsampler.nsamples,
        )
        if ptsampler.pool is not None and hasattr(ptsampler.pool, "_processes"):
            npool = ptsampler.pool._processes
        else:
            npool = 1
        result.meta_data["run_statistics"] = dict(
            nlikelihood=ptsampler.position * ptsampler.L1steps * ptsampler._nsamplers,
            neffsamples=ptsampler.nsamples * ptsampler.convergence_inputs.thin_by_nact,
            sampling_time_s=result.sampling_time.seconds,
            ncores=npool,
        )

        return result

    def setup_chain_set(self):
        if self.read_current_state() and self.resume is True:
            self.ptsampler.pool = self.pool
        else:
            self.init_ptsampler()

    def init_ptsampler(self):

        logger.info(f"Initializing BilbyPTMCMCSampler with:\n{self.get_setup_string()}")
        self.ptsampler = BilbyPTMCMCSampler(
            convergence_inputs=self.convergence_inputs,
            pt_inputs=self.pt_inputs,
            proposal_cycle=self.proposal_cycle,
            pt_rejection_sample=self.pt_rejection_sample,
            pool=self.pool,
            use_ratio=self.use_ratio,
            evidence_method=self.evidence_method,
            initial_sample_method=self.initial_sample_method,
            initial_sample_dict=self.initial_sample_dict,
            normalize_prior=self.normalize_prior,
        )

    def get_setup_string(self):
        string = (
            f"  Convergence settings: {self.convergence_inputs}\n"
            f"  Parallel-tempering settings: {self.pt_inputs}\n"
            f"  proposal_cycle: {self.proposal_cycle}\n"
            f"  pt_rejection_sample: {self.pt_rejection_sample}"
        )
        return string

    def draw(self):
        self._steps_since_last_print = 0
        self._time_since_last_print = 0
        logger.info(f"Drawing {self.target_nsamples} samples")
        logger.info(f"Checkpoint every check_point_delta_t={self.check_point_delta_t}s")
        logger.info(f"Print update every printdt={self.printdt}s")

        while True:
            t0 = datetime.datetime.now()
            self.ptsampler.step_all_chains()
            dt = (datetime.datetime.now() - t0).total_seconds()
            self.ptsampler.sampling_time += dt
            self._time_since_last_print += dt
            self._steps_since_last_print += self.ptsampler.L1steps

            if self._time_since_last_print > self.printdt:
                tp0 = datetime.datetime.now()
                self.print_progress()
                tp = datetime.datetime.now()
                ppt_frac = (tp - tp0).total_seconds() / self._time_since_last_print
                if ppt_frac > 0.01:
                    logger.warning(
                        f"Non-negligible print progress time (ppt_frac={ppt_frac:0.2f})"
                    )
                self._steps_since_last_print = 0
                self._time_since_last_print = 0

            self.check_point()

            if self.ptsampler.nsamples_last >= self.target_nsamples:
                # Perform a second check without cached values
                if self.ptsampler.nsamples_nocache >= self.target_nsamples:
                    logger.info("Reached convergence: exiting sampling")
                    break

    def check_point(self, ignore_time=False):
        tS = (datetime.datetime.now() - self.start_time).total_seconds()
        if os.path.isfile(self.resume_file):
            tR = time.time() - os.path.getmtime(self.resume_file)
        else:
            tR = np.inf

        if ignore_time or np.min([tS, tR]) > self.check_point_delta_t:
            logger.info("Checkpoint start")
            self.write_current_state()
            self.print_long_progress()
            logger.info("Checkpoint finished")

    def _remove_checkpoint(self):
        """Remove checkpointed state"""
        if os.path.isfile(self.resume_file):
            os.remove(self.resume_file)

    def read_current_state(self):
        """Read the existing resume file

        Returns
        -------
        success: boolean
            If true, resume file was successfully loaded, otherwise false

        """
        if os.path.isfile(self.resume_file) is False or not os.path.getsize(
            self.resume_file
        ):
            return False
        import dill

        with open(self.resume_file, "rb") as file:
            ptsampler = dill.load(file)
            if not isinstance(ptsampler, BilbyPTMCMCSampler):
                logger.debug("Malformed resume file, ignoring")
                return False
            self.ptsampler = ptsampler
            if self.ptsampler.pt_inputs != self.pt_inputs:
                msg = (
                    f"pt_inputs has changed: {self.ptsampler.pt_inputs} "
                    f"-> {self.pt_inputs}"
                )
                raise ResumeError(msg)
            self.ptsampler.set_convergence_inputs(self.convergence_inputs)
            self.ptsampler.pt_rejection_sample = self.pt_rejection_sample

        logger.info(
            f"Loaded resume file {self.resume_file} "
            f"with {self.ptsampler.position} steps "
            f"setup:\n{self.get_setup_string()}"
        )
        return True

    def write_current_state(self):
        import dill

        if not hasattr(self, "ptsampler"):
            logger.debug("Attempted checkpoint before initialization")
            return
        logger.debug("Check point")
        check_directory_exists_and_if_not_mkdir(self.outdir)

        _pool = self.ptsampler.pool
        self.ptsampler.pool = None
        if dill.pickles(self.ptsampler):
            safe_file_dump(self.ptsampler, self.resume_file, dill)
            logger.info("Written checkpoint file {}".format(self.resume_file))
        else:
            logger.warning(
                "Cannot write pickle resume file! Job may not resume if interrupted."
            )
            # Touch the file to postpone next check-point attempt
            Path(self.resume_file).touch(exist_ok=True)
        self.ptsampler.pool = _pool

    def print_long_progress(self):
        self.print_per_proposal()
        self.print_tau_dict()
        if self.ptsampler.ntemps > 1:
            self.print_pt_acceptance()
        if self.ptsampler.nensemble > 1:
            self.print_ensemble_acceptance()
        if self.check_point_plot:
            self.plot_progress(
                self.ptsampler, self.label, self.outdir, self.priors, self.diagnostic
            )
            self.ptsampler.compute_evidence(
                outdir=self.outdir, label=self.label, make_plots=True
            )

    def print_ensemble_acceptance(self):
        logger.info(f"Ensemble swaps = {self.ptsampler.swap_counter['ensemble']}")
        logger.info(self.ptsampler.ensemble_proposal_cycle)

    def print_progress(self):
        position = self.ptsampler.position

        # Total sampling time
        sampling_time = datetime.timedelta(seconds=self.ptsampler.sampling_time)
        time = str(sampling_time).split(".")[0]

        # Time for last evaluation set
        time_per_eval_ms = (
            1000 * self._time_since_last_print / self._steps_since_last_print
        )

        # Pull out progress summary
        tau = self.ptsampler.tau
        nsamples = self.ptsampler.nsamples
        minimum_index = self.ptsampler.primary_sampler.chain.minimum_index
        method = self.ptsampler.primary_sampler.chain.minimum_index_method
        mindex_str = f"{minimum_index:0.2e}({method})"
        alpha = self.ptsampler.primary_sampler.acceptance_ratio
        maxl = self.ptsampler.primary_sampler.chain.max_log_likelihood

        nlikelihood = position * self.L1steps * self.ptsampler._nsamplers
        eff = 100 * nsamples / nlikelihood

        # Estimated time til finish (ETF)
        if tau < np.inf:
            remaining_samples = self.target_nsamples - nsamples
            remaining_evals = (
                remaining_samples
                * self.convergence_inputs.thin_by_nact
                * tau
                * self.L1steps
            )
            remaining_time_s = time_per_eval_ms * 1e-3 * remaining_evals
            remaining_time_dt = datetime.timedelta(seconds=remaining_time_s)
            if remaining_samples > 0:
                remaining_time = str(remaining_time_dt).split(".")[0]
            else:
                remaining_time = "0"
        else:
            remaining_time = "-"

        msg = (
            f"{position:0.2e}|{time}|{mindex_str}|t={tau:0.0f}|"
            f"n={nsamples:0.0f}|a={alpha:0.2f}|e={eff:0.1e}%|"
            f"{time_per_eval_ms:0.2f}ms/ev|maxl={maxl:0.2f}|"
            f"ETF={remaining_time}"
        )

        if self.pt_rejection_sample:
            count = self.ptsampler.rejection_sampling_count
            rse = 100 * count / nsamples
            msg += f"|rse={rse:0.2f}%"

        if self.verbose:
            print(msg, flush=True)

    def print_per_proposal(self):
        logger.info("Zero-temperature proposals:")
        for prop in self.ptsampler[0].proposal_cycle.proposal_list:
            logger.info(prop)

    def print_pt_acceptance(self):
        logger.info(f"Temperature swaps = {self.ptsampler.swap_counter['temperature']}")
        for column in self.ptsampler.sampler_list_of_tempered_lists:
            for ii, sampler in enumerate(column):
                total = sampler.pt_accepted + sampler.pt_rejected
                beta = sampler.beta
                if total > 0:
                    ratio = f"{sampler.pt_accepted / total:0.2f}"
                else:
                    ratio = "-"
                logger.info(
                    f"Temp:{ii}<->{ii + 1}|"
                    f"beta={beta:0.4g}|"
                    f"hot-samp={sampler.nsamples}|"
                    f"swap={ratio}|"
                    f"conv={sampler.chain.converged}|"
                )

    def print_tau_dict(self):
        msg = f"Current taus={self.ptsampler.primary_sampler.chain.tau_dict}"
        logger.info(msg)

    @staticmethod
    def plot_progress(ptsampler, label, outdir, priors, diagnostic=False):
        logger.info("Creating diagnostic plots")
        for ii, row in ptsampler.sampler_dictionary.items():
            for jj, sampler in enumerate(row):
                plot_label = f"{label}_E{sampler.Eindex}_T{sampler.Tindex}"
                if diagnostic is True or sampler.beta == 1:
                    sampler.chain.plot(
                        outdir=outdir,
                        label=plot_label,
                        priors=priors,
                        all_samples=ptsampler.samples,
                    )

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
            List of directory names. Will always be empty for bilby_mcmc.
        """
        filenames = [os.path.join(outdir, f"{label}_resume.pickle")]
        return filenames, []


class BilbyPTMCMCSampler(object):
    def __init__(
        self,
        convergence_inputs,
        pt_inputs,
        proposal_cycle,
        pt_rejection_sample,
        pool,
        use_ratio,
        evidence_method,
        initial_sample_method,
        initial_sample_dict,
        normalize_prior=True,
    ):
        self.set_pt_inputs(pt_inputs)
        self.use_ratio = use_ratio
        self.initial_sample_method = initial_sample_method
        self.initial_sample_dict = initial_sample_dict
        self.normalize_prior = normalize_prior
        self.setup_sampler_dictionary(convergence_inputs, proposal_cycle)
        self.set_convergence_inputs(convergence_inputs)
        self.pt_rejection_sample = pt_rejection_sample
        self.pool = pool
        self.evidence_method = evidence_method

        # Initialize counters
        self.swap_counter = Counter()
        self.swap_counter["temperature"] = 0
        self.swap_counter["L2-temperature"] = 0
        self.swap_counter["ensemble"] = 0
        self.swap_counter["L2-ensemble"] = int(self.L2steps / 2) + 1

        self._nsamples_dict = {}
        self.ensemble_proposal_cycle = proposals.get_default_ensemble_proposal_cycle(
            sampling_convenience_dump.priors
        )
        self.sampling_time = 0
        self.ln_z_dict = dict()
        self.ln_z_err_dict = dict()

    def get_initial_betas(self):
        pt_inputs = self.pt_inputs
        if self.ntemps == 1:
            betas = np.array([1])
        elif pt_inputs.initial_betas is not None:
            betas = np.array(pt_inputs.initial_betas)
        elif pt_inputs.Tmax is not None:
            betas = np.logspace(0, -np.log10(pt_inputs.Tmax), pt_inputs.ntemps)
        elif pt_inputs.Tmax_from_SNR is not None:
            ndim = len(sampling_convenience_dump.priors.non_fixed_keys)
            target_hot_likelihood = ndim / 2
            Tmax = pt_inputs.Tmax_from_SNR**2 / (2 * target_hot_likelihood)
            betas = np.logspace(0, -np.log10(Tmax), pt_inputs.ntemps)
        else:
            raise SamplerError("Unable to set temperature ladder from inputs")

        if len(betas) != self.ntemps:
            raise SamplerError("Temperatures do not match ntemps")

        return betas

    def setup_sampler_dictionary(self, convergence_inputs, proposal_cycle):

        betas = self.get_initial_betas()
        logger.info(
            f"Initializing BilbyPTMCMCSampler with:"
            f"ntemps={self.ntemps}, "
            f"nensemble={self.nensemble}, "
            f"pt_ensemble={self.pt_ensemble}, "
            f"initial_betas={betas}, "
            f"initial_sample_method={self.initial_sample_method}, "
            f"initial_sample_dict={self.initial_sample_dict}\n"
        )
        self.sampler_dictionary = dict()
        for Tindex, beta in enumerate(betas):
            if beta == 1 or self.pt_ensemble:
                n = self.nensemble
            else:
                n = 1
            temp_sampler_list = [
                BilbyMCMCSampler(
                    beta=beta,
                    Tindex=Tindex,
                    Eindex=Eindex,
                    convergence_inputs=convergence_inputs,
                    proposal_cycle=proposal_cycle,
                    use_ratio=self.use_ratio,
                    initial_sample_method=self.initial_sample_method,
                    initial_sample_dict=self.initial_sample_dict,
                    normalize_prior=self.normalize_prior,
                )
                for Eindex in range(n)
            ]
            self.sampler_dictionary[Tindex] = temp_sampler_list

        # Store data
        self._nsamplers = len(self.sampler_list)

    @property
    def sampler_list(self):
        """A list of all individual samplers"""
        return [s for item in self.sampler_dictionary.values() for s in item]

    @sampler_list.setter
    def sampler_list(self, sampler_list):
        for sampler in sampler_list:
            self.sampler_dictionary[sampler.Tindex][sampler.Eindex] = sampler

    def sampler_list_by_column(self, column):
        return [row[column] for row in self.sampler_dictionary.values()]

    @property
    def sampler_list_of_tempered_lists(self):
        if self.pt_ensemble:
            return [self.sampler_list_by_column(ii) for ii in range(self.nensemble)]
        else:
            return [self.sampler_list_by_column(0)]

    @property
    def tempered_sampler_list(self):
        return [s for s in self.sampler_list if s.beta < 1]

    @property
    def zerotemp_sampler_list(self):
        return [s for s in self.sampler_list if s.beta == 1]

    @property
    def primary_sampler(self):
        return self.sampler_dictionary[0][0]

    def set_pt_inputs(self, pt_inputs):
        logger.info(f"Setting parallel tempering inputs={pt_inputs}")
        self.pt_inputs = pt_inputs

        # Pull out only what is needed
        self.ntemps = pt_inputs.ntemps
        self.nensemble = pt_inputs.nensemble
        self.pt_ensemble = pt_inputs.pt_ensemble
        self.adapt = pt_inputs.adapt
        self.adapt_t0 = pt_inputs.adapt_t0
        self.adapt_nu = pt_inputs.adapt_nu

    def set_convergence_inputs(self, convergence_inputs):
        logger.info(f"Setting convergence_inputs={convergence_inputs}")
        self.convergence_inputs = convergence_inputs
        self.L1steps = convergence_inputs.L1steps
        self.L2steps = convergence_inputs.L2steps
        for sampler in self.sampler_list:
            sampler.set_convergence_inputs(convergence_inputs)

    @property
    def tau(self):
        return self.primary_sampler.chain.tau

    @property
    def minimum_index(self):
        return self.primary_sampler.chain.minimum_index

    @property
    def nsamples(self):
        pos = self.primary_sampler.chain.position
        if hasattr(self, "_nsamples_dict") is False:
            self._nsamples_dict = {}
        if pos in self._nsamples_dict:
            return self._nsamples_dict[pos]
        logger.debug(f"Calculating nsamples at {pos}")
        self._nsamples_dict[pos] = self._calculate_nsamples()
        return self._nsamples_dict[pos]

    @property
    def nsamples_last(self):
        if len(self._nsamples_dict) > 0:
            return list(self._nsamples_dict.values())[-1]
        else:
            return 0

    @property
    def nsamples_nocache(self):
        for sampler in self.sampler_list:
            sampler.chain.tau_nocache
        pos = self.primary_sampler.chain.position
        self._nsamples_dict[pos] = self._calculate_nsamples()
        return self._nsamples_dict[pos]

    def _calculate_nsamples(self):
        nsamples_list = []
        for sampler in self.zerotemp_sampler_list:
            nsamples_list.append(sampler.nsamples)
        if self.pt_rejection_sample:
            for samp in self.sampler_list[1:]:
                nsamples_list.append(
                    len(samp.rejection_sample_zero_temperature_samples())
                )
        return sum(nsamples_list)

    @property
    def samples(self):
        cached_samples = getattr(self, "_cached_samples", (False,))
        if cached_samples[0] == self.position:
            return cached_samples[1]

        sample_list = []
        for sampler in self.zerotemp_sampler_list:
            sample_list.append(sampler.samples)
        if self.pt_rejection_sample:
            for sampler in self.tempered_sampler_list:
                sample_list.append(sampler.samples)
        samples = pd.concat(sample_list, ignore_index=True)
        self._cached_samples = (self.position, samples)
        return samples

    @property
    def position(self):
        return self.primary_sampler.chain.position

    @property
    def evaluations(self):
        return int(self.position * len(self.sampler_list))

    def __getitem__(self, index):
        return self.sampler_list[index]

    def step_all_chains(self):
        if self.pool:
            self.sampler_list = self.pool.map(call_step, self.sampler_list)
        else:
            for ii, sampler in enumerate(self.sampler_list):
                self.sampler_list[ii] = sampler.step()

        if self.nensemble > 1 and self.swap_counter["L2-ensemble"] >= self.L2steps:
            self.swap_counter["ensemble"] += 1
            self.swap_counter["L2-ensemble"] = 0
            self.ensemble_step()

        if self.ntemps > 1 and self.swap_counter["L2-temperature"] >= self.L2steps:
            self.swap_counter["temperature"] += 1
            self.swap_counter["L2-temperature"] = 0
            self.swap_tempered_chains()
            if self.position < self.adapt_t0 * 10:
                if self.adapt:
                    self.adapt_temperatures()
            elif self.adapt:
                logger.info(
                    f"Adaptation of temperature chains finished at step {self.position}"
                )
                self.adapt = False

        self.swap_counter["L2-ensemble"] += 1
        self.swap_counter["L2-temperature"] += 1

    @staticmethod
    def _get_sample_to_swap(sampler):
        if not (sampler.chain.converged and sampler.stop_after_convergence):
            v = sampler.chain[-1]
        else:
            v = sampler.chain.random_sample
        logl = v[LOGLKEY]
        return v, logl

    def swap_tempered_chains(self):
        if self.pt_ensemble:
            Eindexs = range(self.nensemble)
        else:
            Eindexs = [0]
        for Eindex in Eindexs:
            for Tindex in range(self.ntemps - 1):
                sampleri = self.sampler_dictionary[Tindex][Eindex]
                vi, logli = self._get_sample_to_swap(sampleri)
                betai = sampleri.beta

                samplerj = self.sampler_dictionary[Tindex + 1][Eindex]
                vj, loglj = self._get_sample_to_swap(samplerj)
                betaj = samplerj.beta

                dbeta = betaj - betai
                with np.errstate(over="ignore"):
                    alpha_swap = np.exp(dbeta * (logli - loglj))

                if random.rng.uniform(0, 1) <= alpha_swap:
                    sampleri.chain[-1] = vj
                    samplerj.chain[-1] = vi
                    self.sampler_dictionary[Tindex][Eindex] = sampleri
                    self.sampler_dictionary[Tindex + 1][Eindex] = samplerj
                    sampleri.pt_accepted += 1
                else:
                    sampleri.pt_rejected += 1

    def ensemble_step(self):
        for Tindex, sampler_list in self.sampler_dictionary.items():
            if len(sampler_list) > 1:
                for Eindex, sampler in enumerate(sampler_list):
                    curr = sampler.chain.current_sample
                    proposal = self.ensemble_proposal_cycle.get_proposal()
                    complement = [s.chain for s in sampler_list if s != sampler]
                    prop, log_factor = proposal(sampler.chain, complement)
                    logp = sampler.log_prior(prop)

                    if logp == -np.inf:
                        sampler.reject_proposal(curr, proposal)
                        self.sampler_dictionary[Tindex][Eindex] = sampler
                        continue

                    prop[LOGPKEY] = logp
                    prop[LOGLKEY] = sampler.log_likelihood(prop)
                    alpha = np.exp(
                        log_factor
                        + sampler.beta * prop[LOGLKEY]
                        + prop[LOGPKEY]
                        - sampler.beta * curr[LOGLKEY]
                        - curr[LOGPKEY]
                    )

                    if random.rng.uniform(0, 1) <= alpha:
                        sampler.accept_proposal(prop, proposal)
                    else:
                        sampler.reject_proposal(curr, proposal)
                    self.sampler_dictionary[Tindex][Eindex] = sampler

    def adapt_temperatures(self):
        """Adapt the temperature of the chains

        Using the dynamic temperature selection described in arXiv:1501.05823,
        adapt the chains to target a constant swap ratio. This method is based
        on github.com/willvousden/ptemcee/tree/master/ptemcee
        """

        self.primary_sampler.chain.minimum_index_adapt = self.position
        tt = self.swap_counter["temperature"]
        for sampler_list in self.sampler_list_of_tempered_lists:
            betas = np.array([s.beta for s in sampler_list])
            ratios = np.array([s.acceptance_ratio for s in sampler_list[:-1]])

            # Modulate temperature adjustments with a hyperbolic decay.
            decay = self.adapt_t0 / (tt + self.adapt_t0)
            kappa = decay / self.adapt_nu

            # Construct temperature adjustments.
            dSs = kappa * (ratios[:-1] - ratios[1:])

            # Compute new ladder (hottest and coldest chains don't move).
            deltaTs = np.diff(1 / betas[:-1])
            deltaTs *= np.exp(dSs)
            betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])
            for sampler, beta in zip(sampler_list, betas):
                sampler.beta = beta

    @property
    def ln_z(self):
        return self.ln_z_dict.get(self.evidence_method, np.nan)

    @property
    def ln_z_err(self):
        return self.ln_z_err_dict.get(self.evidence_method, np.nan)

    def compute_evidence(self, outdir, label, make_plots=True):
        if self.ntemps == 1:
            return
        kwargs = dict(outdir=outdir, label=label, make_plots=make_plots)
        methods = dict(
            thermodynamic=self.thermodynamic_integration_evidence,
            stepping_stone=self.stepping_stone_evidence,
        )
        for key, method in methods.items():
            ln_z, ln_z_err = self.compute_evidence_per_ensemble(method, kwargs)
            self.ln_z_dict[key] = ln_z
            self.ln_z_err_dict[key] = ln_z_err
            logger.debug(
                f"Log-evidence of {ln_z:0.2f}+/-{ln_z_err:0.2f} calculated using {key} method"
            )

    def compute_evidence_per_ensemble(self, method, kwargs):
        from scipy.special import logsumexp

        if self.ntemps == 1:
            return np.nan, np.nan

        lnZ_list = []
        lnZerr_list = []
        for index, ptchain in enumerate(self.sampler_list_of_tempered_lists):
            lnZ, lnZerr = method(ptchain, **kwargs)
            lnZ_list.append(lnZ)
            lnZerr_list.append(lnZerr)

        N = len(lnZ_list)

        # Average lnZ
        lnZ = logsumexp(lnZ_list, b=1.0 / N)

        # Propagate uncertainty in combined evidence
        lnZerr = 0.5 * logsumexp(2 * np.array(lnZerr_list), b=1.0 / N)

        return lnZ, lnZerr

    def thermodynamic_integration_evidence(
        self, ptchain, outdir, label, make_plots=True
    ):
        """Computes the evidence using thermodynamic integration

        We compute the evidence without the burnin samples, no thinning
        """
        from scipy.stats import sem

        betas = []
        mean_lnlikes = []
        sem_lnlikes = []
        for sampler in ptchain:
            lnlikes = sampler.chain.get_1d_array(LOGLKEY)
            mindex = sampler.chain.minimum_index
            lnlikes = lnlikes[mindex:]
            mean_lnlikes.append(np.mean(lnlikes))
            sem_lnlikes.append(sem(lnlikes))
            betas.append(sampler.beta)

        # Convert to array and re-order
        betas = np.array(betas)[::-1]
        mean_lnlikes = np.array(mean_lnlikes)[::-1]
        sem_lnlikes = np.array(sem_lnlikes)[::-1]

        lnZ, lnZerr = self._compute_evidence_from_mean_lnlikes(betas, mean_lnlikes)

        if make_plots:
            plot_label = f"{label}_E{ptchain[0].Eindex}"
            self._create_lnZ_plots(
                betas=betas,
                mean_lnlikes=mean_lnlikes,
                outdir=outdir,
                label=plot_label,
                sem_lnlikes=sem_lnlikes,
            )

        return lnZ, lnZerr

    def stepping_stone_evidence(self, ptchain, outdir, label, make_plots=True):
        """
        Compute the evidence using the stepping stone approximation.

        See https://arxiv.org/abs/1810.04488 and
        https://pubmed.ncbi.nlm.nih.gov/21187451/ for details.

        The uncertainty calculation is hopefully combining the evidence in each
        of the steps.

        Returns
        -------
        ln_z: float
            Estimate of the natural log evidence
        ln_z_err: float
            Estimate of the uncertainty in the evidence
        """
        # Order in increasing beta
        ptchain.reverse()

        # Get maximum usable set of samples across the ptchain
        min_index = max([samp.chain.minimum_index for samp in ptchain])
        max_index = min([len(samp.chain.get_1d_array(LOGLKEY)) for samp in ptchain])
        tau = self.tau

        if max_index - min_index <= 1 or np.isinf(tau):
            return np.nan, np.nan

        # Read in log likelihoods
        ln_likes = np.array(
            [samp.chain.get_1d_array(LOGLKEY)[min_index:max_index] for samp in ptchain]
        )[:-1].T

        # Thin to only independent samples
        ln_likes = ln_likes[:: int(self.tau), :]
        steps = ln_likes.shape[0]

        # Calculate delta betas
        betas = np.array([samp.beta for samp in ptchain])

        ln_z, ln_ratio = self._calculate_stepping_stone(betas, ln_likes)

        # Implementation of the bootstrap method described in Maturana-Russel
        # et. al. (2019) to estimate the evidence uncertainty.
        ll = 50  # Block length
        repeats = 100  # Repeats
        ln_z_realisations = []
        try:
            for _ in range(repeats):
                idxs = [random.rng.integers(i, i + ll) for i in range(steps - ll)]
                ln_z_realisations.append(
                    self._calculate_stepping_stone(betas, ln_likes[idxs, :])[0]
                )
            ln_z_err = np.std(ln_z_realisations)
        except ValueError:
            logger.info("Failed to estimate stepping stone uncertainty")
            ln_z_err = np.nan

        if make_plots:
            plot_label = f"{label}_E{ptchain[0].Eindex}"
            self._create_stepping_stone_plot(
                means=ln_ratio,
                outdir=outdir,
                label=plot_label,
            )

        return ln_z, ln_z_err

    @staticmethod
    def _calculate_stepping_stone(betas, ln_likes):
        from scipy.special import logsumexp

        n_samples = ln_likes.shape[0]
        d_betas = betas[1:] - betas[:-1]
        ln_ratio = logsumexp(d_betas * ln_likes, axis=0) - np.log(n_samples)
        return sum(ln_ratio), ln_ratio

    @staticmethod
    def _compute_evidence_from_mean_lnlikes(betas, mean_lnlikes):
        lnZ = trapezoid(mean_lnlikes, betas)
        z2 = trapezoid(mean_lnlikes[::-1][::2][::-1], betas[::-1][::2][::-1])
        lnZerr = np.abs(lnZ - z2)
        return lnZ, lnZerr

    def _create_lnZ_plots(self, betas, mean_lnlikes, outdir, label, sem_lnlikes=None):
        import matplotlib.pyplot as plt

        logger.debug("Creating thermodynamic evidence diagnostic plot")

        fig, ax1 = plt.subplots()
        if betas[-1] == 0:
            x, y = betas[:-1], mean_lnlikes[:-1]
        else:
            x, y = betas, mean_lnlikes
        if sem_lnlikes is not None:
            ax1.errorbar(x, y, sem_lnlikes, fmt="-")
        else:
            ax1.plot(x, y, "-o")
        ax1.set_xscale("log")
        ax1.set_xlabel(r"$\beta$")
        ax1.set_ylabel(r"$\langle \log(\mathcal{L}) \rangle$")

        plt.tight_layout()
        fig.savefig("{}/{}_beta_lnl.png".format(outdir, label))
        plt.close()

    def _create_stepping_stone_plot(self, means, outdir, label):
        import matplotlib.pyplot as plt

        logger.debug("Creating stepping stone evidence diagnostic plot")

        n_steps = len(means)

        fig, axes = plt.subplots(nrows=2, figsize=(8, 10))

        ax = axes[0]
        ax.plot(np.arange(1, n_steps + 1), means)
        ax.set_xlabel("$k$")
        ax.set_ylabel("$r_{k}$")

        ax = axes[1]
        ax.plot(np.arange(1, n_steps + 1), np.cumsum(means[::1])[::1])
        ax.set_xlabel("$k$")
        ax.set_ylabel("Cumulative $\\ln Z$")

        plt.tight_layout()
        fig.savefig("{}/{}_stepping_stone.png".format(outdir, label))
        plt.close()

    @property
    def rejection_sampling_count(self):
        if self.pt_rejection_sample:
            counts = 0
            for column in self.sampler_list_of_tempered_lists:
                for sampler in column:
                    counts += sampler.rejection_sampling_count
            return counts
        else:
            return None


class BilbyMCMCSampler(object):
    def __init__(
        self,
        convergence_inputs,
        proposal_cycle=None,
        beta=1,
        Tindex=0,
        Eindex=0,
        use_ratio=False,
        initial_sample_method="prior",
        initial_sample_dict=None,
        normalize_prior=True,
    ):
        self.beta = beta
        self.Tindex = Tindex
        self.Eindex = Eindex
        self.use_ratio = use_ratio
        self.normalize_prior = normalize_prior
        self.parameters = sampling_convenience_dump.priors.non_fixed_keys
        self.ndim = len(self.parameters)

        if initial_sample_method.lower() == "prior":
            full_sample_dict = sampling_convenience_dump.priors.sample()
            initial_sample = {
                k: v
                for k, v in full_sample_dict.items()
                if k in sampling_convenience_dump.priors.non_fixed_keys
            }
        elif initial_sample_method.lower() in ["maximize", "maximise", "maximum"]:
            initial_sample = get_initial_maximimum_posterior_sample(self.beta)
        else:
            raise ValueError(
                f"initial sample method {initial_sample_method} not understood"
            )

        if initial_sample_dict is not None:
            initial_sample.update(initial_sample_dict)

        if self.beta == 1:
            logger.info(f"Using initial sample {initial_sample}")

        initial_sample = Sample(initial_sample)
        initial_sample[LOGLKEY] = self.log_likelihood(initial_sample)
        initial_sample[LOGPKEY] = self.log_prior(initial_sample)

        self.chain = Chain(initial_sample=initial_sample)
        self.set_convergence_inputs(convergence_inputs)

        self.accepted = 0
        self.rejected = 0
        self.pt_accepted = 0
        self.pt_rejected = 0
        self.rejection_sampling_count = 0

        if isinstance(proposal_cycle, str):
            # Only print warnings for the primary sampler
            if Tindex == 0 and Eindex == 0:
                warn = True
            else:
                warn = False

            self.proposal_cycle = proposals.get_proposal_cycle(
                proposal_cycle,
                sampling_convenience_dump.priors,
                L1steps=self.chain.L1steps,
                warn=warn,
            )
        elif isinstance(proposal_cycle, proposals.ProposalCycle):
            self.proposal_cycle = proposal_cycle
        else:
            raise SamplerError("Proposal cycle not understood")

        if self.Tindex == 0 and self.Eindex == 0:
            logger.info(f"Using {self.proposal_cycle}")

    def set_convergence_inputs(self, convergence_inputs):
        for key, val in convergence_inputs._asdict().items():
            setattr(self.chain, key, val)
        self.target_nsamples = convergence_inputs.target_nsamples
        self.stop_after_convergence = convergence_inputs.stop_after_convergence

    def log_likelihood(self, sample):
        params = deepcopy(sampling_convenience_dump.parameters)
        params.update(sample.sample_dict)

        if self.use_ratio:
            return sampling_convenience_dump.likelihood.log_likelihood_ratio(params)
        else:
            return sampling_convenience_dump.likelihood.log_likelihood(params)

    def log_prior(self, sample):
        return sampling_convenience_dump.priors.ln_prob(
            sample.parameter_only_dict,
            normalized=self.normalize_prior,
        )

    def accept_proposal(self, prop, proposal):
        self.chain.append(prop)
        self.accepted += 1
        proposal.accepted += 1

    def reject_proposal(self, curr, proposal):
        self.chain.append(curr)
        self.rejected += 1
        proposal.rejected += 1

    def step(self):
        if self.stop_after_convergence and self.chain.converged:
            return self

        internal_steps = 0
        internal_accepted = 0
        internal_rejected = 0
        curr = self.chain.current_sample.copy()
        while internal_steps < self.chain.L1steps:
            internal_steps += 1
            proposal = self.proposal_cycle.get_proposal()
            prop, log_factor = proposal(
                self.chain,
                likelihood=sampling_convenience_dump.likelihood,
                priors=sampling_convenience_dump.priors,
            )
            logp = self.log_prior(prop)

            if np.isinf(logp) or np.isnan(logp):
                internal_rejected += 1
                proposal.rejected += 1
                continue

            prop[LOGPKEY] = logp
            prop[LOGLKEY] = self.log_likelihood(prop)

            if np.isinf(prop[LOGLKEY]) or np.isnan(prop[LOGLKEY]):
                internal_rejected += 1
                proposal.rejected += 1
                continue

            with np.errstate(over="ignore"):
                alpha = np.exp(
                    log_factor
                    + self.beta * prop[LOGLKEY]
                    + prop[LOGPKEY]
                    - self.beta * curr[LOGLKEY]
                    - curr[LOGPKEY]
                )

            if random.rng.uniform(0, 1) <= alpha:
                internal_accepted += 1
                proposal.accepted += 1
                curr = prop
                self.chain.current_sample = curr
            else:
                internal_rejected += 1
                proposal.rejected += 1

        self.chain.append(curr)
        self.rejected += internal_rejected
        self.accepted += internal_accepted
        return self

    @property
    def nsamples(self):
        nsamples = self.chain.nsamples
        if nsamples > self.target_nsamples and self.chain.converged is False:
            logger.debug(f"Temperature {self.Tindex} chain reached convergence")
            self.chain.converged = True
        return nsamples

    @property
    def acceptance_ratio(self):
        return self.accepted / (self.accepted + self.rejected)

    @property
    def samples(self):
        if self.beta == 1:
            return self.chain.samples
        else:
            return self.rejection_sample_zero_temperature_samples(print_message=True)

    def rejection_sample_zero_temperature_samples(self, print_message=False):
        beta = self.beta
        chain = self.chain
        hot_samples = pd.DataFrame(
            chain._chain_array[chain.minimum_index : chain.position], columns=chain.keys
        )
        if len(hot_samples) == 0:
            logger.debug(
                f"Rejection sampling for Temp {self.Tindex} failed: "
                "no usable hot samples"
            )
            return hot_samples

        # Pull out log likelihood
        zerotemp_logl = hot_samples[LOGLKEY]

        # Revert to true likelihood if needed
        if sampling_convenience_dump.use_ratio:
            zerotemp_logl += sampling_convenience_dump.likelihood.noise_log_likelihood()

        # Calculate normalised weights
        log_weights = (1 - beta) * zerotemp_logl
        max_weight = np.max(log_weights)
        unnormalised_weights = np.exp(log_weights - max_weight)
        weights = unnormalised_weights / np.sum(unnormalised_weights)

        # Rejection sample
        samples = rejection_sample(hot_samples, weights)

        # Logging
        self.rejection_sampling_count = len(samples)

        if print_message:
            logger.info(
                f"Rejection sampling Temp {self.Tindex}, beta={beta:0.2f} "
                f"yielded {len(samples)} samples"
            )
        return samples


def get_initial_maximimum_posterior_sample(beta):
    """A method to attempt optimization of the maximum likelihood

    This uses a simple scipy optimization approach, starting from a number
    of draws from the prior to avoid problems with local optimization.

    """
    logger.info("Finding initial maximum posterior estimate")
    likelihood = sampling_convenience_dump.likelihood
    priors = sampling_convenience_dump.priors
    search_parameter_keys = sampling_convenience_dump.search_parameter_keys

    bounds = []
    for key in search_parameter_keys:
        bounds.append((priors[key].minimum, priors[key].maximum))

    def neg_log_post(x):
        sample = {key: val for key, val in zip(search_parameter_keys, x)}
        ln_prior = priors.ln_prob(sample)

        if np.isinf(ln_prior):
            return -np.inf

        parameters = deepcopy(sampling_convenience_dump.parameters)
        parameters.update(sample)

        return -beta * likelihood.log_likelihood(parameters) - ln_prior

    res = differential_evolution(neg_log_post, bounds, popsize=100, init="sobol")
    if res.success:
        sample = {key: val for key, val in zip(search_parameter_keys, res.x)}
        logger.info(f"Initial maximum posterior estimate {sample}")
        return sample
    else:
        raise ValueError("Failed to find initial maximum posterior estimate")


# Methods used to aid parallelisation:


def call_step(sampler):
    sampler = sampler.step()
    return sampler
