import os
import sys
import warnings

import numpy as np
from pandas import DataFrame
from scipy.special import logsumexp

from ..utils import check_directory_exists_and_if_not_mkdir, load_json, logger
from .base_sampler import NestedSampler, signal_wrapper


class Nessai(NestedSampler):
    """bilby wrapper of nessai (https://github.com/mj-will/nessai)

    .. warning::
        The nessai sampler interface in bilby is deprecated and will be
        removed in future release. Please use the :code`nessai-bilby`
        sampler plugin instead: https://github.com/bilby-dev/nessai-bilby

    All positional and keyword arguments passed to `run_sampler` are propagated
    to `nessai.flowsampler.FlowSampler`

    See the documentation for an explanation of the different kwargs.

    Documentation: https://nessai.readthedocs.io/
    """

    sampler_name = "nessai"
    _default_kwargs = None
    _run_kwargs_list = None
    sampling_seed_key = "seed"

    msg = (
        "The nessai sampler interface in bilby is deprecated and will"
        " be removed in future release. Please use the `nessai-bilby`"
        "sampler plugin instead: https://github.com/bilby-dev/nessai-bilby."
    )
    warnings.warn(msg, FutureWarning)

    @property
    def run_kwargs_list(self):
        """List of kwargs used in the run method of :code:`FlowSampler`"""
        if not self._run_kwargs_list:
            from nessai.utils.bilbyutils import get_run_kwargs_list

            self._run_kwargs_list = get_run_kwargs_list()
            ignored_kwargs = ["save"]
            for ik in ignored_kwargs:
                if ik in self._run_kwargs_list:
                    self._run_kwargs_list.remove(ik)
        return self._run_kwargs_list

    @property
    def default_kwargs(self):
        """Default kwargs for nessai.

        Retrieves default values from nessai directly and then includes any
        bilby specific defaults. This avoids the need to update bilby when the
        defaults change or new kwargs are added to nessai.

        Includes the following kwargs that are specific to bilby:

        - :code:`nessai_log_level`: allows setting the logging level in nessai
        - :code:`nessai_logging_stream`: allows setting the logging stream
        - :code:`nessai_plot`: allows toggling the plotting in FlowSampler.run
        """
        if not self._default_kwargs:
            from nessai.utils.bilbyutils import get_all_kwargs

            kwargs = get_all_kwargs()

            # Defaults for bilby that will override nessai defaults
            bilby_defaults = dict(
                output=None,
                exit_code=self.exit_code,
                nessai_log_level=None,
                nessai_logging_stream="stdout",
                nessai_plot=True,
                plot_posterior=False,  # bilby already produces a posterior plot
                log_on_iteration=False,  # Use periodic logging by default
                logging_interval=60,  # Log every 60 seconds
            )
            kwargs.update(bilby_defaults)
            # Kwargs that cannot be set in bilby
            remove = [
                "save",
                "signal_handling",
            ]
            for k in remove:
                if k in kwargs:
                    kwargs.pop(k)
            self._default_kwargs = kwargs
        return self._default_kwargs

    def log_prior(self, theta):
        """

        Parameters
        ----------
        theta: list
            List of sampled values on a unit interval

        Returns
        -------
        float: Joint ln prior probability of theta

        """
        return self.priors.ln_prob(theta, axis=0)

    def get_nessai_model(self):
        """Get the model for nessai."""
        from nessai.livepoint import dict_to_live_points
        from nessai.model import Model as BaseModel

        class Model(BaseModel):
            """A wrapper class to pass our log_likelihood and priors into nessai

            Parameters
            ----------
            names : list of str
                List of parameters to sample
            priors : :obj:`bilby.core.prior.PriorDict`
                Priors to use for sampling. Needed for the bounds and the
                `sample` method.
            """

            def __init__(self, names, priors):
                self.names = names
                self.priors = priors
                self._update_bounds()

            @staticmethod
            def log_likelihood(x, **kwargs):
                """Compute the log likelihood"""
                theta = [x[n].item() for n in self.search_parameter_keys]
                return self.log_likelihood(theta)

            @staticmethod
            def log_prior(x, **kwargs):
                """Compute the log prior"""
                theta = {n: x[n] for n in self._search_parameter_keys}
                return self.log_prior(theta)

            def _update_bounds(self):
                self.bounds = {
                    key: [self.priors[key].minimum, self.priors[key].maximum]
                    for key in self.names
                }

            def new_point(self, N=1):
                """Draw a point from the prior"""
                prior_samples = self.priors.sample(size=N)
                samples = {n: prior_samples[n] for n in self.names}
                return dict_to_live_points(samples)

            def new_point_log_prob(self, x):
                """Proposal probability for new the point"""
                return self.log_prior(x)

            @staticmethod
            def from_unit_hypercube(x):
                """Map samples from the unit hypercube to the prior."""
                theta = {}
                for n in self._search_parameter_keys:
                    theta[n] = self.priors[n].rescale(x[n])
                return dict_to_live_points(theta)

            @staticmethod
            def to_unit_hypercube(x):
                """Map samples from the prior to the unit hypercube."""
                theta = {n: x[n] for n in self._search_parameter_keys}
                return dict_to_live_points(self.priors.cdf(theta))

        model = Model(self.search_parameter_keys, self.priors)
        return model

    def split_kwargs(self):
        """Split kwargs into configuration and run time kwargs"""
        kwargs = self.kwargs.copy()
        run_kwargs = {}
        for k in self.run_kwargs_list:
            run_kwargs[k] = kwargs.pop(k)
        run_kwargs["plot"] = kwargs.pop("nessai_plot")
        return kwargs, run_kwargs

    def get_posterior_weights(self):
        """Get the posterior weights for the nested samples"""
        from nessai.posterior import compute_weights

        _, log_weights = compute_weights(
            np.array(self.fs.nested_samples["logL"]),
            np.array(self.fs.ns.state.nlive),
        )
        w = np.exp(log_weights - logsumexp(log_weights))
        return w

    def get_nested_samples(self):
        """Get the nested samples dataframe"""
        ns = DataFrame(self.fs.nested_samples)
        ns.rename(
            columns=dict(logL="log_likelihood", logP="log_prior", it="iteration"),
            inplace=True,
        )
        return ns

    def update_result(self):
        """Update the result object."""
        from nessai.livepoint import live_points_to_array

        # Manually set likelihood evaluations because parallelisation breaks the counter
        self.result.num_likelihood_evaluations = self.fs.ns.total_likelihood_evaluations

        self.result.sampling_time = self.fs.ns.sampling_time
        self.result.samples = live_points_to_array(
            self.fs.posterior_samples, self.search_parameter_keys
        )
        self.result.log_likelihood_evaluations = self.fs.posterior_samples["logL"]
        self.result.nested_samples = self.get_nested_samples()
        self.result.nested_samples["weights"] = self.get_posterior_weights()
        self.result.log_evidence = self.fs.log_evidence
        self.result.log_evidence_err = self.fs.log_evidence_error

    @signal_wrapper
    def run_sampler(self):
        """Run the sampler.

        Nessai is designed to be ran in two stages, initialise the sampler
        and then call the run method with additional configuration. This means
        there are effectively two sets of keyword arguments: one for
        initializing the sampler and the other for the run function.
        """
        from nessai.flowsampler import FlowSampler
        from nessai.utils import setup_logger

        kwargs, run_kwargs = self.split_kwargs()

        # Setup the logger for nessai, use nessai_log_level if specified, else use
        # the level of the bilby logger.
        nessai_log_level = kwargs.pop("nessai_log_level")
        if nessai_log_level is None or nessai_log_level == "bilby":
            nessai_log_level = logger.getEffectiveLevel()
        nessai_logging_stream = kwargs.pop("nessai_logging_stream")

        setup_logger(
            self.outdir,
            label=self.label,
            log_level=nessai_log_level,
            stream=nessai_logging_stream,
        )

        # Get the nessai model
        model = self.get_nessai_model()

        # Configure the sampler
        self.fs = FlowSampler(
            model,
            signal_handling=False,  # Disable signal handling so it can be handled by bilby
            **kwargs,
        )
        # Run the sampler
        self.fs.run(**run_kwargs)

        # Update the result
        self.update_result()

        return self.result

    def _translate_kwargs(self, kwargs):
        """Translate the keyword arguments"""
        super()._translate_kwargs(kwargs)
        if "nlive" not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["nlive"] = kwargs.pop(equiv)
        if "n_pool" not in kwargs:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["n_pool"] = kwargs.pop(equiv)
            if "n_pool" not in kwargs:
                kwargs["n_pool"] = self._npool

    def _verify_kwargs_against_default_kwargs(self):
        """Verify the keyword arguments"""
        if "config_file" in self.kwargs:
            d = load_json(self.kwargs["config_file"], None)
            self.kwargs.update(d)
            self.kwargs.pop("config_file")

        if not self.kwargs["plot"]:
            self.kwargs["plot"] = self.plot

        if not self.kwargs["output"]:
            self.kwargs["output"] = os.path.join(
                self.outdir, f"{self.label}_nessai", ""
            )

        check_directory_exists_and_if_not_mkdir(self.kwargs["output"])
        NestedSampler._verify_kwargs_against_default_kwargs(self)

    def write_current_state(self):
        """Write the current state of the sampler"""
        self.fs.ns.checkpoint()

    def write_current_state_and_exit(self, signum=None, frame=None):
        """
        Overwrites the base class to make sure that :code:`Nessai` terminates
        properly.
        """
        if hasattr(self, "fs"):
            self.fs.terminate_run(code=signum)
        else:
            logger.warning("Sampler is not initialized")
        self._log_interruption(signum=signum)
        sys.exit(self.exit_code)

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
            List of file names. This will be empty for nessai.
        list
            List of directory names.
        """
        dirs = [os.path.join(outdir, f"{label}_{cls.sampler_name}", "")]
        dirs += [os.path.join(dirs[0], d, "") for d in ["proposal", "diagnostics"]]
        filenames = []
        return filenames, dirs

    def _setup_pool(self):
        pass
