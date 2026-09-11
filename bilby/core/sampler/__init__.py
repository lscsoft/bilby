import datetime
import inspect
import sys

from ..prior import DeltaFunction, PriorDict
from ..utils import (
    command_line_args,
    env_package_list,
    get_entry_points,
    global_meta_data,
    loaded_modules_dict,
    logger,
)
from ..utils.parallel import bilby_pool
from . import proposal
from .base_sampler import Sampler, SamplingMarginalisedParameterError


class ImplementedSamplers:
    """Dictionary-like object that contains implemented samplers.

    This class is singleton and only one instance can exist.
    """

    _instance = None

    _samplers = get_entry_points("bilby.samplers")

    def keys(self):
        """Iterator of available samplers by name.

        Reduces the list to its simplest. This includes removing the 'bilby.'
        prefix from native samplers if a corresponding plugin is not available.
        """
        keys = []
        for key in self._samplers.keys():
            name = key.replace("bilby.", "")
            if name in self._samplers.keys():
                keys.append(key)
            else:
                keys.append(name)
        return iter(keys)

    def values(self):
        """Iterator of sampler classes.

        Note: the classes need to loaded using :code:`.load()` before being
        called.
        """
        return iter(self._samplers.values())

    def items(self):
        """Iterator of tuples containing keys (sampler names) and classes.

        Note: the classes need to loaded using :code:`.load()` before being
        called.
        """
        return iter(((k, v) for k, v in zip(self.keys(), self.values())))

    def valid_keys(self):
        """All valid keys including bilby.<sampler name>."""
        keys = set(self._samplers.keys())
        return iter(keys.union({k.replace("bilby.", "") for k in keys}))

    def native_keys(self):
        """Iterator of native sampler names (without the `bilby.` prefix).

        This excludes any samplers that are only available through plugins.
        """
        return iter(
            {
                k.replace("bilby.", "")
                for k in self._samplers.keys()
                if k.startswith("bilby.")
            }
        )

    def __getitem__(self, key):
        if key in self._samplers:
            return self._samplers[key]
        elif f"bilby.{key}" in self._samplers:
            return self._samplers[f"bilby.{key}"]
        else:
            raise ValueError(
                f"Sampler {key} is not implemented! "
                f"Available samplers are: {list(self.keys())}"
            )

    def __contains__(self, value):
        return value in self.valid_keys()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


IMPLEMENTED_SAMPLERS = ImplementedSamplers()


def get_implemented_samplers():
    """Get a list of the names of the implemented samplers.

    This includes natively supported samplers (e.g. dynesty) and any additional
    samplers that are supported through the sampler plugins.

    Returns
    -------
    list
        The list of implemented samplers.
    """
    return list(IMPLEMENTED_SAMPLERS.keys())


def get_sampler_class(sampler):
    """Get the class for a sampler from its name.

    This includes natively supported samplers (e.g. dynesty) and any additional
    samplers that are supported through the sampler plugins.

    Parameters
    ----------
    sampler : str
        The name of the sampler.

    Returns
    -------
    Sampler
        The sampler class.

    Raises
    ------
    ValueError
        Raised if the sampler is not implemented.
    """
    return IMPLEMENTED_SAMPLERS[sampler.lower()].load()


if command_line_args.sampler_help:
    sampler = command_line_args.sampler_help
    if sampler in IMPLEMENTED_SAMPLERS:
        sampler_class = IMPLEMENTED_SAMPLERS[sampler].load()
        print(f'Help for sampler "{sampler}":')
        print(sampler_class.__doc__)
    else:
        if sampler == "None":
            print(
                "For help with a specific sampler, call sampler-help with "
                "the name of the sampler"
            )
        else:
            print(f"Requested sampler {sampler} not implemented")
        print(f"Available samplers = {get_implemented_samplers()}")

    sys.exit()


def run_sampler(
    likelihood,
    priors=None,
    label="label",
    outdir="outdir",
    sampler="dynesty",
    use_ratio=None,
    injection_parameters=None,
    conversion_function=None,
    plot=False,
    default_priors_file=None,
    clean=None,
    meta_data=None,
    save=True,
    gzip=False,
    result_class=None,
    npool=1,
    pool=None,
    **kwargs,
):
    """
    The primary interface to easy parameter estimation

    Parameters
    ==========
    likelihood: `bilby.Likelihood`
        A `Likelihood` instance
    priors: `bilby.PriorDict`
        A PriorDict/dictionary of the priors for each parameter - missing
        parameters will use default priors, if None, all priors will be default
    label: str
        Name for the run, used in output files
    outdir: str
        A string used in defining output files
    sampler: str, Sampler
        The name of the sampler to use - see
        `bilby.sampler.get_implemented_samplers()` for a list of available
        samplers.
        Alternatively a Sampler object can be passed
    use_ratio: bool (None)
        If True, use the likelihood's `log_likelihood_ratio`, rather than just
        the `log_likelihood`. If left as the default value of `None`, and the
        likelihood has a valid `log_likelihood_ratio` method, then that method
        will also be used, i.e., the likelihood ratio will be calculated. To
        ensure that the `log_likelihood` method is used for the calculation,
        rather than the likelihood ratio, this argument should be set to False.
    injection_parameters: dict
        A dictionary of injection parameters used in creating the data (if
        using simulated data). Appended to the result object and saved.
    plot: bool
        If true, generate a corner plot and, if applicable diagnostic plots
    conversion_function: function, optional
        Function to apply to posterior to generate additional parameters.
        This function should take one positional argument, a dictionary or
        pandas dataframe and three optional arguments: the likelihood, prior
        dict, and an integer :code:`npool` to allow parallelisation.
    default_priors_file: str
        If given, a file containing the default priors; otherwise defaults to
        the bilby defaults for a binary black hole.
    clean: bool
        If given, override the command line interface `clean` option.
    meta_data: dict
        If given, adds the key-value pairs to the 'results' object before
        saving. For example, if `meta_data={dtype: 'signal'}`. Warning: in case
        of conflict with keys saved by bilby, the meta_data keys will be
        overwritten.
    save: bool, str
        If true, save the priors and results to disk.
        If hdf5, save as an hdf5 file instead of json.
        If pickle or pkl, save as an pickle file instead of json.
    gzip: bool
        If true, and save is true, gzip the saved results file.
    result_class: bilby.core.result.Result, or child of
        The result class to use. By default, `bilby.core.result.Result` is used,
        but objects which inherit from this class can be given providing
        additional methods.
    npool: int
        An integer specifying the available CPUs to create pool objects for
        parallelization.
    **kwargs:
        All kwargs are passed directly to the samplers `run` function

    Returns
    =======
    result: bilby.core.result.Result
        An object containing the results
    """

    logger.info(f"Running for label '{label}', output will be saved to '{outdir}'")

    if clean:
        command_line_args.clean = clean
    if command_line_args.clean:
        kwargs["resume"] = False

    from . import IMPLEMENTED_SAMPLERS

    if priors is None:
        priors = dict()

    _check_marginalized_parameters_not_sampled(likelihood, priors)

    if type(priors) == dict:
        priors = PriorDict(priors)
    elif isinstance(priors, PriorDict):
        pass
    else:
        raise ValueError("Input priors not understood should be dict or PriorDict")

    priors.fill_priors()

    # Generate the meta-data if not given and append the likelihood meta_data
    if meta_data is None:
        meta_data = dict()
    likelihood.label = label
    likelihood.outdir = outdir
    meta_data["likelihood"] = likelihood.meta_data
    meta_data["loaded_modules"] = loaded_modules_dict()
    meta_data["environment_packages"] = env_package_list(as_dataframe=True)
    meta_data["global_meta_data"] = global_meta_data

    if command_line_args.bilby_zero_likelihood_mode:
        from bilby.core.likelihood import ZeroLikelihood

        likelihood = ZeroLikelihood(likelihood)

    common_kwargs = dict(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        label=label,
        injection_parameters=injection_parameters,
        meta_data=meta_data,
        use_ratio=use_ratio,
        plot=plot,
        result_class=result_class,
        npool=npool,
        pool=pool,
    )

    if isinstance(sampler, Sampler):
        pass
    elif isinstance(sampler, str):
        sampler_class = get_sampler_class(sampler)
        sampler = sampler_class(**common_kwargs, **kwargs)
    elif inspect.isclass(sampler):
        sampler = sampler.__init__(**common_kwargs, **kwargs)
    else:
        raise ValueError(
            "Provided sampler should be a Sampler object or name of a known "
            f"sampler: {get_implemented_samplers()}."
        )

    if sampler.cached_result:
        logger.warning("Using cached result")
        result = sampler.cached_result
        result = apply_conversion_function(
            result=result,
            likelihood=likelihood,
            conversion_function=conversion_function,
            npool=npool,
            pool=pool,
        )
    else:
        # Run the sampler
        with bilby_pool(
            likelihood,
            priors,
            use_ratio=sampler.use_ratio,
            search_parameter_keys=sampler.search_parameter_keys,
            npool=npool,
            pool=pool,
            parameters=priors.sample(),
        ) as _pool:
            start_time = datetime.datetime.now()
            sampler.pool = _pool
            if command_line_args.bilby_test_mode:
                result = sampler._run_test()
            else:
                result = sampler.run_sampler()
            end_time = datetime.datetime.now()
            result = finalize_result(
                result=result,
                likelihood=likelihood,
                use_ratio=sampler.use_ratio,
                start_time=start_time,
                end_time=end_time,
            )

            # Initial save of the sampler in case of failure in samples_to_posterior
            if save:
                result.save_to_file(extension=save, gzip=gzip, outdir=outdir)

            result = apply_conversion_function(
                result=result,
                likelihood=likelihood,
                conversion_function=conversion_function,
                npool=npool,
                pool=_pool,
            )

    if save:
        # The overwrite here ensures we overwrite the initially stored data
        result.save_to_file(overwrite=True, extension=save, gzip=gzip, outdir=outdir)

    if plot:
        result.plot_corner()
    logger.info(f"Summary of results:\n{result}")
    return result


def apply_conversion_function(
    result, likelihood, conversion_function, npool=None, pool=None
):
    """
    Apply the conversion function to the injected parameters and posterior if the
    posterior has not already been created from the stored samples.

    Parameters
    ----------
    result : bilby.core.result.Result
        The result object from the sampler.
    likelihood : bilby.Likelihood
        The likelihood used during sampling.
    conversion_function : function
        The conversion function to apply.
    npool : int, optional
        The number of processes to use in a processing pool.
    pool : multiprocessing.Pool, schwimmbad.MPIPool, optional
        The pool to use for parallelisation, this overrides the :code:`npool` argument.
    """
    if None not in [result.injection_parameters, conversion_function]:
        result.injection_parameters = conversion_function(
            result.injection_parameters,
            likelihood=likelihood,
        )

    # Check if the posterior has already been created
    if getattr(result, "_posterior", None) is None:
        result.samples_to_posterior(
            likelihood=likelihood,
            priors=result.priors,
            conversion_function=conversion_function,
            npool=npool,
            pool=pool,
        )
    return result


def finalize_result(result, likelihood, use_ratio, start_time=None, end_time=None):
    # Some samplers calculate the sampling time internally
    if result.sampling_time is None and None not in [start_time, end_time]:
        result.sampling_time = end_time - start_time
    elif isinstance(result.sampling_time, (float, int)):
        result.sampling_time = datetime.timedelta(result.sampling_time)

    logger.info(f"Sampling time: {result.sampling_time}")
    # Convert sampling time into seconds
    result.sampling_time = result.sampling_time.total_seconds()

    if use_ratio:
        result.log_noise_evidence = likelihood.noise_log_likelihood()
        result.log_bayes_factor = result.log_evidence
        result.log_evidence = result.log_bayes_factor + result.log_noise_evidence
    else:
        result.log_noise_evidence = likelihood.noise_log_likelihood()
        result.log_bayes_factor = result.log_evidence - result.log_noise_evidence

    return result


def _check_marginalized_parameters_not_sampled(likelihood, priors):
    for key in likelihood.marginalized_parameters:
        if key in priors:
            if not isinstance(priors[key], (float, DeltaFunction)):
                raise SamplingMarginalisedParameterError(
                    f"Likelihood is {key} marginalized but you are trying to sample in {key}. "
                )
