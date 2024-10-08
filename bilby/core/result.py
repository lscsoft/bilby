import datetime
import inspect
import json
import os
from collections import namedtuple
from copy import copy
from importlib import import_module
from itertools import product
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import scipy.stats

from . import utils
from .utils import (
    logger, infer_parameters_from_function,
    check_directory_exists_and_if_not_mkdir,
    latex_plot_format, safe_save_figure,
    BilbyJsonEncoder, load_json,
    move_old_file, get_version_information,
    decode_bilby_json, docstring,
    recursively_save_dict_contents_to_group,
    recursively_load_dict_contents_from_group,
    recursively_decode_bilby_json,
    safe_file_dump,
    random,
)
from .prior import Prior, PriorDict, DeltaFunction, ConditionalDeltaFunction


EXTENSIONS = ["json", "hdf5", "h5", "pickle", "pkl"]


def __eval_l(likelihood, params):
    likelihood.parameters.update(params)
    return likelihood.log_likelihood()


def result_file_name(outdir, label, extension='json', gzip=False):
    """ Returns the standard filename used for a result file

    Parameters
    ==========
    outdir: str
        Name of the output directory
    label: str
        Naming scheme of the output file
    extension: str, optional
        Whether to save as `hdf5`, `json`, or `pickle`
    gzip: bool, optional
        Set to True to append `.gz` to the extension for saving in gzipped format

    Returns
    =======
    str: File name of the output file
    """
    if extension == 'pickle':
        extension = 'pkl'
    if extension in ['json', 'hdf5', 'pkl']:
        if extension == 'json' and gzip:
            return os.path.join(outdir, '{}_result.{}.gz'.format(label, extension))
        else:
            return os.path.join(outdir, '{}_result.{}'.format(label, extension))
    else:
        raise ValueError("Extension type {} not understood".format(extension))


def _determine_file_name(filename, outdir, label, extension, gzip):
    """ Helper method to determine the filename """
    if filename is not None:
        if isinstance(filename, os.PathLike):
            # convert PathLike object to string
            return str(filename)
        else:
            return filename
    else:
        if (outdir is None) and (label is None):
            raise ValueError("No information given to load file")
        else:
            return result_file_name(outdir, label, extension, gzip)


def read_in_result(filename=None, outdir=None, label=None, extension='json', gzip=False, result_class=None):
    """ Reads in a stored bilby result object

    Parameters
    ==========
    filename: str
        Path to the file to be read (alternative to giving the outdir and label)
    outdir, label, extension: str
        Name of the output directory, label and extension used for the default
        naming scheme.
    result_class: bilby.core.result.Result, or child of
        The result class to use. By default, `bilby.core.result.Result` is used,
        but objects which inherit from this class can be given providing
        additional methods.
    """
    filename = _determine_file_name(filename, outdir, label, extension, gzip)

    if result_class is None:
        result_class = Result
    elif not issubclass(result_class, Result):
        raise ValueError(f"Input result_class={result_class} not understood")

    # Get the actual extension (may differ from the default extension if the filename is given)
    extension = os.path.splitext(filename)[1].lstrip('.')
    if extension == 'gz':  # gzipped file
        extension = os.path.splitext(os.path.splitext(filename)[0])[1].lstrip('.')

    if 'json' in extension:
        result = result_class.from_json(filename=filename)
    elif ('hdf5' in extension) or ('h5' in extension):
        result = result_class.from_hdf5(filename=filename)
    elif ("pkl" in extension) or ("pickle" in extension):
        result = result_class.from_pickle(filename=filename)
    elif extension is None:
        raise ValueError("No filetype extension provided")
    else:
        raise ValueError("Filetype {} not understood".format(extension))
    return result


def read_in_result_list(filename_list, invalid="warning"):
    """ Read in a set of results

    Parameters
    ==========
    filename_list: list
        A list of filename paths
    invalid: str (ignore, warning, error)
        Behaviour if a file in filename_list is not a valid bilby result

    Returns
    -------
    result_list: ResultList
        A list of results
    """
    results_list = []
    for filename in filename_list:
        if (
            not os.path.exists(filename)
            and os.path.exists(f"{os.path.splitext(filename)[0]}.pkl")
        ):
            pickle_path = f"{os.path.splitext(filename)[0]}.pkl"
            logger.warning(
                f"Result file {filename} doesn't exist but {pickle_path} does. "
                f"Using {pickle_path}."
            )
            filename = pickle_path
        try:
            results_list.append(read_in_result(filename=filename))
        except Exception as e:
            msg = f"Failed to read in file {filename} due to exception {e}"
            if invalid == "error":
                raise ResultListError(msg)
            elif invalid == "warning":
                logger.warning(msg)
    return ResultList(results_list)


def get_weights_for_reweighting(
        result, new_likelihood=None, new_prior=None, old_likelihood=None,
        old_prior=None, resume_file=None, n_checkpoint=5000, npool=1):
    """ Calculate the weights for reweight()

    See bilby.core.result.reweight() for help with the inputs

    Returns
    =======
    ln_weights: array
        An array of the natural-log weights
    new_log_likelihood_array: array
        An array of the natural-log likelihoods from the new likelihood
    new_log_prior_array: array
        An array of the natural-log priors
    old_log_likelihood_array: array
        An array of the natural-log likelihoods from the old likelihood
    old_log_prior_array: array
        An array of the natural-log priors
    resume_file: string
        filepath for the resume file which stores the weights
    n_checkpoint: int
        Number of samples to reweight before writing a resume file
    """
    from tqdm.auto import tqdm

    nposterior = len(result.posterior)

    if (resume_file is not None) and os.path.exists(resume_file):
        old_log_likelihood_array, old_log_prior_array, new_log_likelihood_array, new_log_prior_array = \
            np.genfromtxt(resume_file)

        starting_index = np.argmin(np.abs(old_log_likelihood_array))
        logger.info(f'Checkpoint resuming from {starting_index}.')

    else:
        old_log_likelihood_array = np.zeros(nposterior)
        old_log_prior_array = np.zeros(nposterior)
        new_log_likelihood_array = np.zeros(nposterior)
        new_log_prior_array = np.zeros(nposterior)

        starting_index = 0

    dict_samples = [{key: sample[key] for key in result.posterior}
                    for _, sample in result.posterior.iterrows()]
    n = len(dict_samples) - starting_index

    # Helper function to compute likelihoods in parallel
    def eval_pool(this_logl):
        with multiprocessing.Pool(processes=npool) as pool:
            chunksize = max(100, n // (2 * npool))
            return list(tqdm(
                pool.imap(partial(__eval_l, this_logl),
                        dict_samples[starting_index:], chunksize=chunksize),
                desc='Computing likelihoods',
                total=n)
            )

    if old_likelihood is None:
        old_log_likelihood_array[starting_index:] = \
            result.posterior["log_likelihood"][starting_index:].to_numpy()
    else:
        old_log_likelihood_array[starting_index:] = eval_pool(old_likelihood)

    if new_likelihood is None:
        # Don't perform likelihood reweighting (i.e. likelihood isn't updated)
        new_log_likelihood_array[starting_index:] = old_log_likelihood_array[starting_index:]
    else:
        new_log_likelihood_array[starting_index:] = eval_pool(new_likelihood)

    # Compute priors
    for ii, sample in enumerate(tqdm(dict_samples[starting_index:],
                                     desc='Computing priors',
                                     total=n),
                                start=starting_index):
        # prior calculation needs to not have prior or likelihood keys
        ln_prior = sample.pop("log_prior", np.nan)
        if "log_likelihood" in sample:
            del sample["log_likelihood"]

        if old_prior is not None:
            old_log_prior_array[ii] = old_prior.ln_prob(sample)
        else:
            old_log_prior_array[ii] = ln_prior

        if new_prior is not None:
            new_log_prior_array[ii] = new_prior.ln_prob(sample)
        else:
            # Don't perform prior reweighting (i.e. prior isn't updated)
            new_log_prior_array[ii] = old_log_prior_array[ii]

        if (ii % (n_checkpoint) == 0) and (resume_file is not None):
            checkpointed_index = np.argmin(np.abs(old_log_likelihood_array))
            logger.info(f'Checkpointing with {checkpointed_index} samples')
            np.savetxt(
                resume_file,
                [old_log_likelihood_array, old_log_prior_array, new_log_likelihood_array, new_log_prior_array])

    ln_weights = (
        new_log_likelihood_array + new_log_prior_array - old_log_likelihood_array - old_log_prior_array)

    return ln_weights, new_log_likelihood_array, new_log_prior_array, old_log_likelihood_array, old_log_prior_array


def rejection_sample(posterior, weights):
    """ Perform rejection sampling on a posterior using weights

    Parameters
    ==========
    posterior: pd.DataFrame or np.ndarray of shape (nsamples, nparameters)
        The dataframe or array containing posterior samples
    weights: np.ndarray
        An array of weights

    Returns
    =======
    reweighted_posterior: pd.DataFrame
        The posterior resampled using rejection sampling

    """
    keep = weights > random.rng.uniform(0, max(weights), weights.shape)
    return posterior[keep]


def reweight(result, label=None, new_likelihood=None, new_prior=None,
             old_likelihood=None, old_prior=None, conversion_function=None, npool=1,
             verbose_output=False, resume_file=None, n_checkpoint=5000,
             use_nested_samples=False):
    """ Reweight a result to a new likelihood/prior using rejection sampling

    Parameters
    ==========
    label: str, optional
        An updated label to apply to the result object
    new_likelihood: bilby.core.likelood.Likelihood, (optional)
        If given, the new likelihood to reweight too. If not given, likelihood
        reweighting is not applied
    new_prior: bilby.core.prior.PriorDict, (optional)
        If given, the new prior to reweight too. If not given, prior
        reweighting is not applied
    old_likelihood: bilby.core.likelihood.Likelihood, (optional)
        If given, calculate the old likelihoods from this object. If not given,
        the values stored in the posterior are used.
    old_prior: bilby.core.prior.PriorDict, (optional)
        If given, calculate the old prior from this object. If not given,
        the values stored in the posterior are used.
    conversion_function: function, optional
        Function which adds in extra parameters to the data frame,
        should take the data_frame, likelihood and prior as arguments.
    npool: int, optional
        Number of threads with which to execute the conversion function
    verbose_output: bool, optional
        Flag determining whether the weight array and associated prior and
        likelihood evaluations are output as well as the result file
    resume_file: string, optional
        filepath for the resume file which stores the weights
    n_checkpoint: int, optional
        Number of samples to reweight before writing a resume file
    use_nested_samples: bool, optional
        If true reweight the nested samples instead. This can greatly improve reweighting efficiency, especially if the
        target distribution has support beyond the proposal posterior distribution.

    Returns
    =======
    result: bilby.core.result.Result
        A copy of the result object with a reweighted posterior
    new_log_likelihood_array: array, optional (if verbose_output=True)
        An array of the natural-log likelihoods from the new likelihood
    new_log_prior_array: array, optional (if verbose_output=True)
        An array of the natural-log priors from the new likelihood
    old_log_likelihood_array: array, optional (if verbose_output=True)
        An array of the natural-log likelihoods from the old likelihood
    old_log_prior_array: array, optional (if verbose_output=True)
        An array of the natural-log priors from the old likelihood

    """
    from scipy.special import logsumexp

    result = copy(result)

    if use_nested_samples:
        result.posterior = result.nested_samples

    nposterior = len(result.posterior)
    logger.info("Reweighting posterior with {} samples".format(nposterior))

    ln_weights, new_log_likelihood_array, new_log_prior_array, old_log_likelihood_array, old_log_prior_array =\
        get_weights_for_reweighting(
            result, new_likelihood=new_likelihood, new_prior=new_prior,
            old_likelihood=old_likelihood, old_prior=old_prior,
            resume_file=resume_file, n_checkpoint=n_checkpoint, npool=npool)

    if use_nested_samples:
        ln_weights += np.log(result.posterior["weights"])

    weights = np.exp(ln_weights)

    # Overwrite the likelihood and prior evaluations
    result.posterior["log_likelihood"] = new_log_likelihood_array
    result.posterior["log_prior"] = new_log_prior_array

    result.posterior = rejection_sample(result.posterior, weights=weights)
    result.posterior = result.posterior.reset_index(drop=True)
    logger.info("Rejection sampling resulted in {} samples".format(len(result.posterior)))
    result.meta_data["reweighted_using_rejection_sampling"] = True

    if use_nested_samples:
        result.log_evidence += np.log(np.sum(weights))
    else:
        result.log_evidence += logsumexp(ln_weights) - np.log(nposterior)

    if new_prior is not None:
        for key, prior in new_prior.items():
            result.priors[key] = prior

    if conversion_function is not None:
        data_frame = result.posterior
        if "npool" in inspect.signature(conversion_function).parameters:
            data_frame = conversion_function(data_frame, new_likelihood, new_prior, npool=npool)
        else:
            data_frame = conversion_function(data_frame, new_likelihood, new_prior)
        result.posterior = data_frame

    if label:
        result.label = label
    else:
        result.label += "_reweighted"

    if verbose_output:
        return result, weights, new_log_likelihood_array, \
            new_log_prior_array, old_log_likelihood_array, old_log_prior_array
    else:
        return result


class Result(object):
    def __init__(self, label='no_label', outdir='.', sampler=None,
                 search_parameter_keys=None, fixed_parameter_keys=None,
                 constraint_parameter_keys=None, priors=None,
                 sampler_kwargs=None, injection_parameters=None,
                 meta_data=None, posterior=None, samples=None,
                 nested_samples=None, log_evidence=np.nan,
                 log_evidence_err=np.nan, information_gain=np.nan,
                 log_noise_evidence=np.nan, log_bayes_factor=np.nan,
                 log_likelihood_evaluations=None,
                 log_prior_evaluations=None, sampling_time=None, nburn=None,
                 num_likelihood_evaluations=None, walkers=None,
                 max_autocorrelation_time=None, use_ratio=None,
                 parameter_labels=None, parameter_labels_with_unit=None,
                 version=None):
        """ A class to store the results of the sampling run

        Parameters
        ==========
        label, outdir, sampler: str
            The label, output directory, and sampler used
        search_parameter_keys, fixed_parameter_keys, constraint_parameter_keys: list
            Lists of the search, constraint, and fixed parameter keys.
            Elements of the list should be of type `str` and match the keys
            of the `prior`
        priors: dict, bilby.core.prior.PriorDict
            A dictionary of the priors used in the run
        sampler_kwargs: dict
            Key word arguments passed to the sampler
        injection_parameters: dict
            A dictionary of the injection parameters
        meta_data: dict
            A dictionary of meta data to store about the run
        posterior: pandas.DataFrame
            A pandas data frame of the posterior
        samples, nested_samples: array_like
            An array of the output posterior samples and the unweighted samples
        log_evidence, log_evidence_err, log_noise_evidence, log_bayes_factor: float
            Natural log evidences
        information_gain: float
            The Kullback-Leibler divergence
        log_likelihood_evaluations: array_like
            The evaluations of the likelihood for each sample point
        num_likelihood_evaluations: int
            The number of times the likelihood function is called
        log_prior_evaluations: array_like
            The evaluations of the prior for each sample point
        sampling_time: datetime.timedelta, float
            The time taken to complete the sampling
        nburn: int
            The number of burn-in steps discarded for MCMC samplers
        walkers: array_like
            The samplers taken by a ensemble MCMC samplers
        max_autocorrelation_time: float
            The estimated maximum autocorrelation time for MCMC samplers
        use_ratio: bool
            A boolean stating whether the likelihood ratio, as opposed to the
            likelihood was used during sampling
        parameter_labels, parameter_labels_with_unit: list
            Lists of the latex-formatted parameter labels
        version: str,
            Version information for software used to generate the result. Note,
            this information is generated when the result object is initialized

        Notes
        =====
        All sampling output parameters, e.g. the samples themselves are
        typically not given at initialisation, but set at a later stage.

        """

        self.label = label
        self.outdir = os.path.abspath(outdir)
        self.sampler = sampler
        self.search_parameter_keys = search_parameter_keys
        self.fixed_parameter_keys = fixed_parameter_keys
        self.constraint_parameter_keys = constraint_parameter_keys
        self.parameter_labels = parameter_labels
        self.parameter_labels_with_unit = parameter_labels_with_unit
        self.priors = priors
        self.sampler_kwargs = sampler_kwargs
        self.meta_data = meta_data
        self.injection_parameters = injection_parameters
        self.posterior = posterior
        self.samples = samples
        if isinstance(nested_samples, dict):
            nested_samples = pd.DataFrame(nested_samples)
        self.nested_samples = nested_samples
        self.walkers = walkers
        self.nburn = nburn
        self.use_ratio = use_ratio
        self.log_evidence = log_evidence
        self.log_evidence_err = log_evidence_err
        self.information_gain = information_gain
        self.log_noise_evidence = log_noise_evidence
        self.log_bayes_factor = log_bayes_factor
        self.log_likelihood_evaluations = log_likelihood_evaluations
        self.log_prior_evaluations = log_prior_evaluations
        self.num_likelihood_evaluations = num_likelihood_evaluations
        if isinstance(sampling_time, float):
            sampling_time = datetime.timedelta(seconds=sampling_time)
        self.sampling_time = sampling_time
        self.version = version
        self.max_autocorrelation_time = max_autocorrelation_time

        self.prior_values = None
        self._kde = None

    _load_doctstring = """ Read in a saved .{format} data file

    Parameters
    ==========
    filename: str
        If given, try to load from this filename
    outdir, label: str
        If given, use the default naming convention for saved results file

    Returns
    =======
    result: bilby.core.result.Result

    Raises
    ======
    ValueError: If no filename is given and either outdir or label is None
                If no bilby.core.result.Result is found in the path

    """

    @staticmethod
    @docstring(_load_doctstring.format(format="pickle"))
    def from_pickle(filename=None, outdir=None, label=None):
        filename = _determine_file_name(filename, outdir, label, 'hdf5', False)
        import dill
        with open(filename, "rb") as ff:
            return dill.load(ff)

    @classmethod
    @docstring(_load_doctstring.format(format="hdf5"))
    def from_hdf5(cls, filename=None, outdir=None, label=None):
        import h5py
        filename = _determine_file_name(filename, outdir, label, 'hdf5', False)
        with h5py.File(filename, "r") as ff:
            data = recursively_load_dict_contents_from_group(ff, '/')
        data["posterior"] = pd.DataFrame(data["posterior"])
        data["priors"] = PriorDict._get_from_json_dict(
            json.loads(data["priors"], object_hook=decode_bilby_json)
        )
        try:
            cls = getattr(import_module(data['__module__']), data['__name__'])
        except ImportError:
            logger.debug(
                "Module {}.{} not found".format(data["__module__"], data["__name__"])
            )
        except KeyError:
            logger.debug("No class specified, using base Result.")
        for key in ["__module__", "__name__"]:
            if key in data:
                del data[key]
        return cls(**data)

    @classmethod
    @docstring(_load_doctstring.format(format="json"))
    def from_json(cls, filename=None, outdir=None, label=None, gzip=False):
        from json.decoder import JSONDecodeError

        filename = _determine_file_name(filename, outdir, label, 'json', gzip)

        if os.path.isfile(filename):
            try:
                dictionary = load_json(filename, gzip)
            except JSONDecodeError as e:
                raise IOError(
                    "JSON failed to decode {} with message {}".format(filename, e)
                )
            try:
                return cls(**dictionary)
            except TypeError as e:
                raise IOError("Unable to load dictionary, error={}".format(e))
        else:
            raise IOError("No result '{}' found".format(filename))

    def __str__(self):
        """Print a summary """
        if getattr(self, 'posterior', None) is not None:
            if getattr(self, 'log_noise_evidence', None) is not None:
                return ("nsamples: {:d}\n"
                        "ln_noise_evidence: {:6.3f}\n"
                        "ln_evidence: {:6.3f} +/- {:6.3f}\n"
                        "ln_bayes_factor: {:6.3f} +/- {:6.3f}\n"
                        .format(len(self.posterior), self.log_noise_evidence, self.log_evidence,
                                self.log_evidence_err, self.log_bayes_factor,
                                self.log_evidence_err))
            else:
                return ("nsamples: {:d}\n"
                        "ln_evidence: {:6.3f} +/- {:6.3f}\n"
                        .format(len(self.posterior), self.log_evidence, self.log_evidence_err))
        else:
            return ''

    @property
    def meta_data(self):
        return self._meta_data

    @meta_data.setter
    def meta_data(self, meta_data):
        if meta_data is None:
            meta_data = dict()
        meta_data = recursively_decode_bilby_json(meta_data)
        self._meta_data = meta_data

    @property
    def priors(self):
        if self._priors is not None:
            return self._priors
        else:
            raise ValueError('Result object has no priors')

    @priors.setter
    def priors(self, priors):
        if isinstance(priors, dict):
            if isinstance(priors, PriorDict):
                self._priors = priors
            else:
                self._priors = PriorDict(priors)
            if self.parameter_labels is None:
                self.parameter_labels = [self.priors[k].latex_label for k in
                                         self.search_parameter_keys]
            if self.parameter_labels_with_unit is None:
                self.parameter_labels_with_unit = [
                    self.priors[k].latex_label_with_unit for k in
                    self.search_parameter_keys]
        elif priors is None:
            self._priors = priors
            self.parameter_labels = self.search_parameter_keys
            self.parameter_labels_with_unit = self.search_parameter_keys
        else:
            raise ValueError("Input priors not understood")

    @property
    def samples(self):
        """ An array of samples """
        if self._samples is not None:
            return self._samples
        else:
            raise ValueError("Result object has no stored samples")

    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @property
    def num_likelihood_evaluations(self):
        """ number of likelihood evaluations """
        if self._num_likelihood_evaluations is not None:
            return self._num_likelihood_evaluations
        else:
            raise ValueError("Result object has no stored likelihood evaluations")

    @num_likelihood_evaluations.setter
    def num_likelihood_evaluations(self, num_likelihood_evaluations):
        self._num_likelihood_evaluations = num_likelihood_evaluations

    @property
    def nested_samples(self):
        """" An array of unweighted samples """
        if self._nested_samples is not None:
            return self._nested_samples
        else:
            raise ValueError("Result object has no stored nested samples")

    @nested_samples.setter
    def nested_samples(self, nested_samples):
        self._nested_samples = nested_samples

    @property
    def walkers(self):
        """" An array of the ensemble walkers """
        if self._walkers is not None:
            return self._walkers
        else:
            raise ValueError("Result object has no stored walkers")

    @walkers.setter
    def walkers(self, walkers):
        self._walkers = walkers

    @property
    def nburn(self):
        """" An array of the ensemble walkers """
        if self._nburn is not None:
            return self._nburn
        else:
            raise ValueError("Result object has no stored nburn")

    @nburn.setter
    def nburn(self, nburn):
        self._nburn = nburn

    @property
    def posterior(self):
        """ A pandas data frame of the posterior """
        if self._posterior is not None:
            return self._posterior
        else:
            raise ValueError("Result object has no stored posterior")

    @posterior.setter
    def posterior(self, posterior):
        self._posterior = posterior

    @property
    def log_10_bayes_factor(self):
        return self.log_bayes_factor / np.log(10)

    @property
    def log_10_evidence(self):
        return self.log_evidence / np.log(10)

    @property
    def log_10_evidence_err(self):
        return self.log_evidence_err / np.log(10)

    @property
    def log_10_noise_evidence(self):
        return self.log_noise_evidence / np.log(10)

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, version):
        if version is None:
            self._version = 'bilby={}'.format(utils.get_version_information())
        else:
            self._version = version

    def _get_save_data_dictionary(self):
        # This list defines all the parameters saved in the result object
        save_attrs = [
            'label', 'outdir', 'sampler', 'log_evidence', 'log_evidence_err',
            'log_noise_evidence', 'log_bayes_factor', 'priors', 'posterior',
            'injection_parameters', 'meta_data', 'search_parameter_keys',
            'fixed_parameter_keys', 'constraint_parameter_keys',
            'sampling_time', 'sampler_kwargs', 'use_ratio', 'information_gain',
            'log_likelihood_evaluations', 'log_prior_evaluations',
            'num_likelihood_evaluations', 'samples', 'nested_samples',
            'walkers', 'nburn', 'parameter_labels', 'parameter_labels_with_unit',
            'version']
        dictionary = dict()
        for attr in save_attrs:
            try:
                dictionary[attr] = getattr(self, attr)
            except ValueError as e:
                logger.debug("Unable to save {}, message: {}".format(attr, e))
                pass
        return dictionary

    def save_to_file(self, filename=None, overwrite=False, outdir=None,
                     extension='json', gzip=False):
        """

        Writes the Result to a file.

        Supported formats are: `json`, `hdf5`, `pickle`

        Parameters
        ==========
        filename: optional,
            Filename to write to (overwrites the default)
        overwrite: bool, optional
            Whether or not to overwrite an existing result file.
            default=False
        outdir: str, optional
            Path to the outdir. Default is the one stored in the result object.
        extension: str, optional {json, hdf5, pkl, pickle, True}
            Determines the method to use to store the data (if True defaults
            to json)
        gzip: bool, optional
            If true, and outputting to a json file, this will gzip the resulting
            file and add '.gz' to the file extension.
        """

        if extension is True:
            extension = "json"

        outdir = self._safe_outdir_creation(outdir, self.save_to_file)
        if filename is None:
            filename = result_file_name(outdir, self.label, extension, gzip)

        move_old_file(filename, overwrite)

        # Convert the prior to a string representation for saving on disk
        dictionary = self._get_save_data_dictionary()

        # Convert callable sampler_kwargs to strings
        if dictionary.get('sampler_kwargs', None) is not None:
            for key in dictionary['sampler_kwargs']:
                if hasattr(dictionary['sampler_kwargs'][key], '__call__'):
                    dictionary['sampler_kwargs'][key] = str(dictionary['sampler_kwargs'])

        try:
            # convert priors to JSON dictionary for both JSON and hdf5 files
            if extension == 'json':
                dictionary["priors"] = dictionary["priors"]._get_json_dict()
                if gzip:
                    import gzip
                    # encode to a string
                    json_str = json.dumps(dictionary, cls=BilbyJsonEncoder).encode('utf-8')
                    with gzip.GzipFile(filename, 'w') as file:
                        file.write(json_str)
                else:
                    with open(filename, 'w') as file:
                        json.dump(dictionary, file, indent=2, cls=BilbyJsonEncoder)
            elif extension == 'hdf5':
                import h5py
                dictionary["__module__"] = self.__module__
                dictionary["__name__"] = self.__class__.__name__
                with h5py.File(filename, 'w') as h5file:
                    recursively_save_dict_contents_to_group(h5file, '/', dictionary)
            elif extension == 'pkl':
                safe_file_dump(self, filename, "dill")
            else:
                raise ValueError("Extension type {} not understood".format(extension))
        except Exception as e:
            filename = ".".join(filename.split(".")[:-1]) + ".pkl"
            safe_file_dump(self, filename, "dill")
            logger.error(
                "\n\nSaving the data has failed with the following message:\n"
                "{}\nData has been dumped to {}.\n\n".format(e, filename)
            )

    def save_posterior_samples(self, filename=None, outdir=None, label=None):
        """ Saves posterior samples to a file

        Generates a .dat file containing the posterior samples and auxiliary
        data saved in the posterior. Note, strings in the posterior are
        removed while complex numbers will be given as absolute values with
        abs appended to the column name

        Parameters
        ==========
        filename: str
            Alternative filename to use. Defaults to
            outdir/label_posterior_samples.dat
        outdir, label: str
            Alternative outdir and label to use

        """
        if filename is None:
            if label is None:
                label = self.label
            outdir = self._safe_outdir_creation(outdir, self.save_posterior_samples)
            filename = '{}/{}_posterior_samples.dat'.format(outdir, label)
        else:
            outdir = os.path.dirname(filename)
            self._safe_outdir_creation(outdir, self.save_posterior_samples)

        # Drop non-numeric columns
        df = self.posterior.select_dtypes([np.number]).copy()

        # Convert complex columns to abs
        for key in df.keys():
            if np.any(np.iscomplex(df[key])):
                complex_term = df.pop(key)
                df.loc[:, key + "_abs"] = np.abs(complex_term)
                df.loc[:, key + "_angle"] = np.angle(complex_term)

        logger.info("Writing samples file to {}".format(filename))
        df.to_csv(filename, index=False, header=True, sep=' ')

    def get_latex_labels_from_parameter_keys(self, keys):
        """ Returns a list of latex_labels corresponding to the given keys

        Parameters
        ==========
        keys: list
            List of strings corresponding to the desired latex_labels

        Returns
        =======
        list: The desired latex_labels

        """
        latex_labels = []
        for key in keys:
            if key in self.search_parameter_keys:
                idx = self.search_parameter_keys.index(key)
                label = self.parameter_labels_with_unit[idx]
            elif key in self.parameter_labels:
                label = key
            else:
                label = None
                logger.debug(
                    'key {} not a parameter label or latex label'.format(key)
                )
            if label is None:
                label = key.replace("_", " ")
            latex_labels.append(label)
        return latex_labels

    @property
    def covariance_matrix(self):
        """ The covariance matrix of the samples the posterior """
        samples = self.posterior[self.search_parameter_keys].values
        return np.cov(samples.T)

    @property
    def posterior_volume(self):
        """ The posterior volume """
        if self.covariance_matrix.ndim == 0:
            return np.sqrt(self.covariance_matrix)
        else:
            return 1 / np.sqrt(np.abs(np.linalg.det(
                1 / self.covariance_matrix)))

    @staticmethod
    def prior_volume(priors):
        """ The prior volume, given a set of priors """
        return np.prod([priors[k].maximum - priors[k].minimum for k in priors])

    def occam_factor(self, priors):
        """ The Occam factor,

        See Chapter 28, `Mackay "Information Theory, Inference, and Learning
        Algorithms" <http://www.inference.org.uk/itprnn/book.html>`_ Cambridge
        University Press (2003).

        """
        return self.posterior_volume / self.prior_volume(priors)

    @property
    def bayesian_model_dimensionality(self):
        """ Characterises how many parameters are effectively constraint by the data

        See <https://arxiv.org/abs/1903.06682>

        Returns
        =======
        float: The model dimensionality
        """
        return 2 * (np.mean(self.posterior['log_likelihood']**2) -
                    np.mean(self.posterior['log_likelihood'])**2)

    def get_one_dimensional_median_and_error_bar(self, key, fmt='.2f',
                                                 quantiles=(0.16, 0.84)):
        """ Calculate the median and error bar for a given key

        Parameters
        ==========
        key: str
            The parameter key for which to calculate the median and error bar
        fmt: str, ('.2f')
            A format string
        quantiles: list, tuple
            A length-2 tuple of the lower and upper-quantiles to calculate
            the errors bars for.

        Returns
        =======
        summary: namedtuple
            An object with attributes, median, lower, upper and string

        """
        summary = namedtuple('summary', ['median', 'lower', 'upper', 'string'])

        if len(quantiles) != 2:
            raise ValueError("quantiles must be of length 2")

        quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
        quants = np.percentile(self.posterior[key], quants_to_compute * 100)
        summary.median = quants[1]
        summary.plus = quants[2] - summary.median
        summary.minus = summary.median - quants[0]

        fmt = "{{0:{0}}}".format(fmt).format
        string_template = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        summary.string = string_template.format(
            fmt(summary.median), fmt(summary.minus), fmt(summary.plus))
        return summary

    @latex_plot_format
    def plot_single_density(self, key, prior=None, cumulative=False,
                            title=None, truth=None, save=True,
                            file_base_name=None, bins=50, label_fontsize=16,
                            title_fontsize=16, quantiles=(0.16, 0.84), dpi=300):
        """ Plot a 1D marginal density, either probability or cumulative.

        Parameters
        ==========
        key: str
            Name of the parameter to plot
        prior: {bool (True), bilby.core.prior.Prior}
            If true, add the stored prior probability density function to the
            one-dimensional marginal distributions. If instead a Prior
            is provided, this will be plotted.
        cumulative: bool
            If true plot the CDF
        title: bool
            If true, add 1D title of the median and (by default 1-sigma)
            error bars. To change the error bars, pass in the quantiles kwarg.
            See method `get_one_dimensional_median_and_error_bar` for further
            details). If `quantiles=None` is passed in, no title is added.
        truth: {bool, float}
            If true, plot self.injection_parameters[parameter].
            If float, plot this value.
        save: bool:
            If true, save plot to disk.
        file_base_name: str, optional
            If given, the base file name to use (by default `outdir/label_` is
            used)
        bins: int
            The number of histogram bins
        label_fontsize, title_fontsize: int
            The fontsizes for the labels and titles
        quantiles: tuple
            A length-2 tuple of the lower and upper-quantiles to calculate
            the errors bars for.
        dpi: int
            Dots per inch resolution of the plot

        Returns
        =======
        figure: matplotlib.pyplot.figure
            A matplotlib figure object
        """
        import matplotlib.pyplot as plt
        logger.info('Plotting {} marginal distribution'.format(key))
        label = self.get_latex_labels_from_parameter_keys([key])[0]
        fig, ax = plt.subplots()
        try:
            ax.hist(self.posterior[key].values, bins=bins, density=True,
                    histtype='step', cumulative=cumulative)
        except ValueError as e:
            logger.info(
                'Failed to generate 1d plot for {}, error message: {}'
                .format(key, e))
            return
        ax.set_xlabel(label, fontsize=label_fontsize)
        if truth is not None:
            ax.axvline(truth, ls='-', color='orange')

        summary = self.get_one_dimensional_median_and_error_bar(
            key, quantiles=quantiles)
        ax.axvline(summary.median - summary.minus, ls='--', color='C0')
        ax.axvline(summary.median + summary.plus, ls='--', color='C0')
        if title:
            ax.set_title(summary.string, fontsize=title_fontsize)

        if isinstance(prior, Prior):
            theta = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 300)
            if cumulative is False:
                ax.plot(theta, prior.prob(theta), color='C2')
            else:
                ax.plot(theta, prior.cdf(theta), color='C2')

        if save:
            fig.tight_layout()
            if cumulative:
                file_name = file_base_name + key + '_cdf'
            else:
                file_name = file_base_name + key + '_pdf'
            safe_save_figure(fig=fig, filename=file_name, dpi=dpi)
            plt.close(fig)
        else:
            return fig

    def plot_marginals(self, parameters=None, priors=None, titles=True,
                       file_base_name=None, bins=50, label_fontsize=16,
                       title_fontsize=16, quantiles=(0.16, 0.84), dpi=300,
                       outdir=None):
        """ Plot 1D marginal distributions

        Parameters
        ==========
        parameters: (list, dict), optional
            If given, either a list of the parameter names to include, or a
            dictionary of parameter names and their "true" values to plot.
        priors: {bool (False), bilby.core.prior.PriorDict}
            If true, add the stored prior probability density functions to the
            one-dimensional marginal distributions. If instead a PriorDict
            is provided, this will be plotted.
        titles: bool
            If true, add 1D titles of the median and (by default 1-sigma)
            error bars. To change the error bars, pass in the quantiles kwarg.
            See method `get_one_dimensional_median_and_error_bar` for further
            details). If `quantiles=None` is passed in, no title is added.
        file_base_name: str, optional
            If given, the base file name to use (by default `outdir/label_` is
            used)
        bins: int
            The number of histogram bins
        label_fontsize, title_fontsize: int
            The font sizes for the labels and titles
        quantiles: tuple
            A length-2 tuple of the lower and upper-quantiles to calculate
            the errors bars for.
        dpi: int
            Dots per inch resolution of the plot
        outdir: str, optional
            Path to the outdir. Default is the one store in the result object.

        Returns
        =======
        """
        if isinstance(parameters, dict):
            plot_parameter_keys = list(parameters.keys())
            truths = parameters
        elif parameters is None:
            plot_parameter_keys = self.posterior.keys()
            if self.injection_parameters is None:
                truths = dict()
            else:
                truths = self.injection_parameters
        else:
            plot_parameter_keys = list(parameters)
            if self.injection_parameters is None:
                truths = dict()
            else:
                truths = self.injection_parameters

        if file_base_name is None:
            outdir = self._safe_outdir_creation(outdir, self.plot_marginals)
            file_base_name = '{}/{}_1d/'.format(outdir, self.label)
            check_directory_exists_and_if_not_mkdir(file_base_name)

        if priors is True:
            priors = getattr(self, 'priors', dict())
        elif isinstance(priors, dict):
            pass
        elif priors in [False, None]:
            priors = dict()
        else:
            raise ValueError('Input priors={} not understood'.format(priors))

        for i, key in enumerate(plot_parameter_keys):
            if not isinstance(self.posterior[key].values[0], float):
                continue
            prior = priors.get(key, None)
            truth = truths.get(key, None)
            for cumulative in [False, True]:
                self.plot_single_density(
                    key, prior=prior, cumulative=cumulative, title=titles,
                    truth=truth, save=True, file_base_name=file_base_name,
                    bins=bins, label_fontsize=label_fontsize, dpi=dpi,
                    title_fontsize=title_fontsize, quantiles=quantiles)

    @latex_plot_format
    def plot_corner(self, parameters=None, priors=None, titles=True, save=True,
                    filename=None, dpi=300, **kwargs):
        """ Plot a corner-plot

        Parameters
        ==========
        parameters: (list, dict), optional
            If given, either a list of the parameter names to include, or a
            dictionary of parameter names and their "true" values to plot.
        priors: {bool (False), bilby.core.prior.PriorDict}
            If true, add the stored prior probability density functions to the
            one-dimensional marginal distributions. If instead a PriorDict
            is provided, this will be plotted.
        titles: bool
            If true, add 1D titles of the median and (by default 1-sigma)
            error bars. To change the error bars, pass in the quantiles kwarg.
            See method `get_one_dimensional_median_and_error_bar` for further
            details). If `quantiles=None` is passed in, no title is added.
        save: bool, optional
            If true, save the image using the given label and outdir
        filename: str, optional
            If given, overwrite the default filename
        dpi: int, optional
            Dots per inch resolution of the plot
        **kwargs:
            Other keyword arguments are passed to `corner.corner`. We set some
            defaults to improve the basic look and feel, but these can all be
            overridden. Also optional an 'outdir' argument which can be used
            to override the outdir set by the absolute path of the result object.

        Notes
        =====
            The generation of the corner plot themselves is done by the corner
            python module, see https://corner.readthedocs.io for more
            information.

            Truth-lines can be passed in in several ways. Either as the values
            of the parameters dict, or a list via the `truths` kwarg. If
            injection_parameters where given to run_sampler, these will auto-
            matically be added to the plot. This behaviour can be stopped by
            adding truths=False.

        Returns
        =======
        fig:
            A matplotlib figure instance

        """
        import corner
        import matplotlib.pyplot as plt

        # If in testing mode, not corner plots are generated
        if utils.command_line_args.bilby_test_mode:
            return

        defaults_kwargs = dict(
            bins=50, smooth=0.9,
            title_kwargs=dict(fontsize=16), color='#0072C1',
            truth_color='tab:orange', quantiles=[0.16, 0.84],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, plot_datapoints=True, fill_contours=True,
            max_n_ticks=3)

        if 'lionize' in kwargs and kwargs['lionize'] is True:
            defaults_kwargs['truth_color'] = 'tab:blue'
            defaults_kwargs['color'] = '#FF8C00'

        label_kwargs_defaults = dict(fontsize=16)
        hist_kwargs_defaults = dict(density=True)

        label_kwargs_input = kwargs.get("label_kwargs", dict())
        hist_kwargs_input = kwargs.get("hist_kwargs", dict())

        label_kwargs_defaults.update(label_kwargs_input)
        hist_kwargs_defaults.update(hist_kwargs_input)

        defaults_kwargs.update(kwargs)
        kwargs = defaults_kwargs

        kwargs["label_kwargs"] = label_kwargs_defaults
        kwargs["hist_kwargs"] = hist_kwargs_defaults

        # Handle if truths was passed in
        if 'truth' in kwargs:
            kwargs['truths'] = kwargs.pop('truth')
        if "truths" in kwargs:
            truths = kwargs.get('truths')
            if isinstance(parameters, list) and isinstance(truths, list):
                if len(parameters) != len(truths):
                    raise ValueError(
                        "Length of parameters and truths don't match")
            elif isinstance(truths, dict) and parameters is None:
                parameters = kwargs.pop('truths')
            elif isinstance(truths, bool):
                pass
            elif truths is None:
                kwargs["truths"] = False
            else:
                raise ValueError(
                    "Combination of parameters and truths not understood")

        # If injection parameters where stored, use these as parameter values
        # but do not overwrite input parameters (or truths)
        cond1 = getattr(self, 'injection_parameters', None) is not None
        cond2 = parameters is None
        cond3 = bool(kwargs.get("truths", True))
        if cond1 and cond2 and cond3:
            parameters = {
                key: self.injection_parameters.get(key, np.nan)
                for key in self.search_parameter_keys
            }

        # If parameters is a dictionary, use the keys to determine which
        # parameters to plot and the values as truths.
        if isinstance(parameters, dict):
            plot_parameter_keys = list(parameters.keys())
            kwargs['truths'] = list(parameters.values())
        elif parameters is None:
            plot_parameter_keys = self.search_parameter_keys
        else:
            plot_parameter_keys = list(parameters)

        # Get latex formatted strings for the plot labels
        kwargs['labels'] = kwargs.get(
            'labels', self.get_latex_labels_from_parameter_keys(
                plot_parameter_keys))

        kwargs["labels"] = sanity_check_labels(kwargs["labels"])

        # Unless already set, set the range to include all samples
        # This prevents ValueErrors being raised for parameters with no range
        kwargs['range'] = kwargs.get('range', [1] * len(plot_parameter_keys))

        # Remove truths if it is a bool
        if isinstance(kwargs.get('truths'), bool):
            kwargs.pop('truths')

        # Create the data array to plot and pass everything to corner
        xs = self.posterior[plot_parameter_keys].values
        if len(plot_parameter_keys) > 1:
            fig = corner.corner(xs, **kwargs)
        else:
            ax = kwargs.get("ax", plt.subplot())
            ax.hist(xs, bins=kwargs["bins"], color=kwargs["color"],
                    histtype="step", **kwargs["hist_kwargs"])
            ax.set_xlabel(kwargs["labels"][0])
            fig = plt.gcf()

        axes = fig.get_axes()

        #  Add the titles
        if titles and kwargs.get('quantiles', None) is not None:
            for i, par in enumerate(plot_parameter_keys):
                ax = axes[i + i * len(plot_parameter_keys)]
                if ax.title.get_text() == '':
                    ax.set_title(self.get_one_dimensional_median_and_error_bar(
                        par, quantiles=kwargs['quantiles']).string,
                        **kwargs['title_kwargs'])

        #  Add priors to the 1D plots
        if priors is True:
            priors = getattr(self, 'priors', False)
        if isinstance(priors, dict):
            for i, par in enumerate(plot_parameter_keys):
                ax = axes[i + i * len(plot_parameter_keys)]
                theta = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 300)
                ax.plot(theta, priors[par].prob(theta), color='C2')
        elif priors in [False, None]:
            pass
        else:
            raise ValueError('Input priors={} not understood'.format(priors))

        if save:
            if filename is None:
                outdir = self._safe_outdir_creation(kwargs.get('outdir'), self.plot_corner)
                filename = '{}/{}_corner.png'.format(outdir, self.label)
            logger.debug('Saving corner plot to {}'.format(filename))
            safe_save_figure(fig=fig, filename=filename, dpi=dpi)
            plt.close(fig)

        return fig

    @latex_plot_format
    def plot_walkers(self, **kwargs):
        """ Method to plot the trace of the walkers in an ensemble MCMC plot """
        import matplotlib.pyplot as plt
        if hasattr(self, 'walkers') is False:
            logger.warning("Cannot plot_walkers as no walkers are saved")
            return

        if utils.command_line_args.bilby_test_mode:
            return

        nwalkers, nsteps, ndim = self.walkers.shape
        idxs = np.arange(nsteps)
        fig, axes = plt.subplots(nrows=ndim, figsize=(6, 3 * ndim))
        walkers = self.walkers[:, :, :]
        parameter_labels = sanity_check_labels(self.parameter_labels)
        for i, ax in enumerate(axes):
            ax.plot(idxs[:self.nburn + 1], walkers[:, :self.nburn + 1, i].T,
                    lw=0.1, color='r')
            ax.set_ylabel(parameter_labels[i])

        for i, ax in enumerate(axes):
            ax.plot(idxs[self.nburn:], walkers[:, self.nburn:, i].T, lw=0.1,
                    color='k')
            ax.set_ylabel(parameter_labels[i])

        fig.tight_layout()
        outdir = self._safe_outdir_creation(kwargs.get('outdir'), self.plot_walkers)
        filename = '{}/{}_walkers.png'.format(outdir, self.label)
        logger.debug('Saving walkers plot to {}'.format('filename'))
        safe_save_figure(fig=fig, filename=filename)
        plt.close(fig)

    @latex_plot_format
    def plot_with_data(self, model, x, y, ndraws=1000, npoints=1000,
                       xlabel=None, ylabel=None, data_label='data',
                       data_fmt='o', draws_label=None, filename=None,
                       maxl_label='max likelihood', dpi=300, outdir=None):
        """ Generate a figure showing the data and fits to the data

        Parameters
        ==========
        model: function
            A python function which when called as `model(x, **kwargs)` returns
            the model prediction (here `kwargs` is a dictionary of key-value
            pairs of the model parameters.
        x, y: np.ndarray
            The independent and dependent data to plot
        ndraws: int
            Number of draws from the posterior to plot
        npoints: int
            Number of points used to plot the smoothed fit to the data
        xlabel, ylabel: str
            Labels for the axes
        data_label, draws_label, maxl_label: str
            Label for the data, draws, and max likelihood legend
        data_fmt: str
            Matpltolib fmt code, defaults to `'-o'`
        dpi: int
            Passed to `plt.savefig`
        filename: str
            If given, the filename to use. Otherwise, the filename is generated
            from the outdir and label attributes.
        outdir: str, optional
            Path to the outdir. Default is the one store in the result object.

        """
        import matplotlib.pyplot as plt

        # Determine model_posterior, the subset of the full posterior which
        # should be passed into the model
        model_keys = infer_parameters_from_function(model)
        model_posterior = self.posterior[model_keys]

        xsmooth = np.linspace(np.min(x), np.max(x), npoints)
        fig, ax = plt.subplots()
        logger.info('Plotting {} draws'.format(ndraws))
        for _ in range(ndraws):
            s = model_posterior.sample().to_dict('records')[0]
            ax.plot(xsmooth, model(xsmooth, **s), alpha=0.25, lw=0.1, color='r',
                    label=draws_label)
        try:
            if all(~np.isnan(self.posterior.log_likelihood)):
                logger.info('Plotting maximum likelihood')
                s = model_posterior.iloc[self.posterior.log_likelihood.idxmax()]
                ax.plot(xsmooth, model(xsmooth, **s), lw=1, color='k',
                        label=maxl_label)
        except (AttributeError, TypeError):
            logger.debug(
                "No log likelihood values stored, unable to plot max")

        ax.plot(x, y, data_fmt, markersize=2, label=data_label)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        ax.legend(numpoints=3)
        fig.tight_layout()
        if filename is None:
            outdir = self._safe_outdir_creation(outdir, self.plot_with_data)
            filename = '{}/{}_plot_with_data'.format(outdir, self.label)
        safe_save_figure(fig=fig, filename=filename, dpi=dpi)
        plt.close(fig)

    @staticmethod
    def _add_prior_fixed_values_to_posterior(posterior, priors):
        if priors is None:
            return posterior
        for key in priors:
            if isinstance(priors[key], DeltaFunction) and \
                    not isinstance(priors[key], ConditionalDeltaFunction):
                posterior[key] = priors[key].peak
            elif isinstance(priors[key], float):
                posterior[key] = priors[key]
        return posterior

    def samples_to_posterior(self, likelihood=None, priors=None,
                             conversion_function=None, npool=1):
        """
        Convert array of samples to posterior (a Pandas data frame)

        Also applies the conversion function to any stored posterior

        Parameters
        ==========
        likelihood: bilby.likelihood.GravitationalWaveTransient, optional
            GravitationalWaveTransient likelihood used for sampling.
        priors: bilby.prior.PriorDict, optional
            Dictionary of prior object, used to fill in delta function priors.
        conversion_function: function, optional
            Function which adds in extra parameters to the data frame,
            should take the data_frame, likelihood and prior as arguments.
        """

        data_frame = pd.DataFrame(
            self.samples, columns=self.search_parameter_keys)
        data_frame = self._add_prior_fixed_values_to_posterior(
            data_frame, priors)
        data_frame['log_likelihood'] = getattr(
            self, 'log_likelihood_evaluations', np.nan)
        if self.log_prior_evaluations is None and priors is not None:
            data_frame['log_prior'] = priors.ln_prob(
                dict(data_frame[self.search_parameter_keys]), axis=0)
        else:
            data_frame['log_prior'] = self.log_prior_evaluations

        if conversion_function is not None:
            if "npool" in inspect.signature(conversion_function).parameters:
                data_frame = conversion_function(data_frame, likelihood, priors, npool=npool)
            else:
                data_frame = conversion_function(data_frame, likelihood, priors)
        self.posterior = data_frame

    def calculate_prior_values(self, priors):
        """
        Evaluate prior probability for each parameter for each sample.

        Parameters
        ==========
        priors: dict, PriorDict
            Prior distributions
        """
        self.prior_values = pd.DataFrame()
        for key in priors:
            if key in self.posterior.keys():
                if isinstance(priors[key], DeltaFunction):
                    continue
                else:
                    self.prior_values[key]\
                        = priors[key].prob(self.posterior[key].values)

    def get_all_injection_credible_levels(self, keys=None, weights=None):
        """
        Get credible levels for all parameters

        Parameters
        ==========
        keys: list, optional
            A list of keys for which return the credible levels, if None,
            defaults to search_parameter_keys
        weights: array, optional
            A list of weights for the posterior samples to calculate a set of
            weighted credible intervals.
            If None, assumes equal weights between samples.

        Returns
        =======
        credible_levels: dict
            The credible levels at which the injected parameters are found.
        """
        if keys is None:
            keys = self.search_parameter_keys
        if self.injection_parameters is None:
            raise (
                TypeError,
                "Result object has no 'injection_parameters'. "
                "Cannot compute credible levels."
            )
        credible_levels = {key: self.get_injection_credible_level(key, weights=weights)
                           for key in keys
                           if isinstance(self.injection_parameters.get(key, None), float)}
        return credible_levels

    def get_injection_credible_level(self, parameter, weights=None):
        """
        Get the credible level of the injected parameter

        Calculated as CDF(injection value)

        Parameters
        ==========
        parameter: str
            Parameter to get credible level for
        weights: array, optional
            A list of weights for the posterior samples to calculate a
            weighted credible interval.
            If None, assumes equal weights between samples.

        Returns
        =======
        float: credible level
        """
        if self.injection_parameters is None:
            raise (
                TypeError,
                "Result object has no 'injection_parameters'. "
                "Cannot copmute credible levels."
            )

        if weights is None:
            weights = np.ones(len(self.posterior))

        if parameter in self.posterior and\
                parameter in self.injection_parameters:
            credible_level =\
                sum(np.array(self.posterior[parameter].values <
                    self.injection_parameters[parameter]) * weights) / (sum(weights))
            return credible_level
        else:
            return np.nan

    def _check_attribute_match_to_other_object(self, name, other_object):
        """ Check attribute name exists in other_object and is the same

        Parameters
        ==========
        name: str
            Name of the attribute in this instance
        other_object: object
            Other object with attributes to compare with

        Returns
        =======
        bool: True if attribute name matches with an attribute of other_object, False otherwise

        """
        a = getattr(self, name, False)
        b = getattr(other_object, name, False)
        logger.debug('Checking {} value: {}=={}'.format(name, a, b))
        if (a is not False) and (b is not False):
            type_a = type(a)
            type_b = type(b)
            if type_a == type_b:
                if type_a in [str, float, int, dict, list]:
                    try:
                        return a == b
                    except ValueError:
                        return False
                elif type_a in [np.ndarray]:
                    return np.all(a == b)
        return False

    @property
    def kde(self):
        """ Kernel density estimate built from the stored posterior

        Uses `scipy.stats.gaussian_kde` to generate the kernel density
        """
        if self._kde:
            return self._kde
        else:
            self._kde = scipy.stats.gaussian_kde(
                self.posterior[self.search_parameter_keys].values.T)
            return self._kde

    def posterior_probability(self, sample):
        """ Calculate the posterior probability for a new sample

        This queries a Kernel Density Estimate of the posterior to calculate
        the posterior probability density for the new sample.

        Parameters
        ==========
        sample: dict, or list of dictionaries
            A dictionary containing all the keys from
            self.search_parameter_keys and corresponding values at which to
            calculate the posterior probability

        Returns
        =======
        p: array-like,
            The posterior probability of the sample

        """
        if isinstance(sample, dict):
            sample = [sample]
        ordered_sample = [[s[key] for key in self.search_parameter_keys]
                          for s in sample]
        return self.kde(ordered_sample)

    def _safe_outdir_creation(self, outdir=None, caller_func=None):
        if outdir is None:
            outdir = self.outdir
        try:
            utils.check_directory_exists_and_if_not_mkdir(outdir)
        except PermissionError:
            raise FileMovedError("Can not write in the out directory.\n"
                                 "Did you move the here file from another system?\n"
                                 "Try calling " + caller_func.__name__ + " with the 'outdir' "
                                 "keyword argument, e.g. " + caller_func.__name__ + "(outdir='.')")
        return outdir

    def get_weights_by_new_prior(self, old_prior, new_prior, prior_names=None):
        """ Calculate a list of sample weights based on the ratio of new to old priors

            Parameters
            ==========
            old_prior: PriorDict,
                The prior used in the generation of the original samples.

            new_prior: PriorDict,
                The prior to use to reweight the samples.

            prior_names: list
                A list of the priors to include in the ratio during reweighting.

            Returns
            =======
            weights: array-like,
                A list of sample weights.

                """
        weights = []

        # Shared priors - these will form a ratio
        if prior_names is not None:
            shared_parameters = {key: self.posterior[key] for key in new_prior if
                                 key in old_prior and key in prior_names}
        else:
            shared_parameters = {key: self.posterior[key] for key in new_prior if key in old_prior}
        parameters = [{key: self.posterior[key][i] for key in shared_parameters.keys()}
                      for i in range(len(self.posterior))]

        for i in range(len(self.posterior)):
            weight = 1
            for prior_key in shared_parameters.keys():
                val = self.posterior[prior_key][i]
                weight *= new_prior.evaluate_constraints(parameters[i])
                weight *= new_prior[prior_key].prob(val) / old_prior[prior_key].prob(val)

            weights.append(weight)

        return weights

    def to_arviz(self, prior=None):
        """ Convert the Result object to an ArviZ InferenceData object.

            Parameters
            ==========
            prior: int
                If a positive integer is given then that number of prior
                samples will be drawn and stored in the ArviZ InferenceData
                object.

            Returns
            =======
            azdata: InferenceData
                The ArviZ InferenceData object.

            Raises
            ======
            RuntimeError: If ArviZ is not installed.
        """

        try:
            import arviz as az
        except ImportError:
            raise ResultError(
                "ArviZ is not installed, so cannot convert to InferenceData."
            )

        posdict = {}
        for key in self.posterior:
            posdict[key] = self.posterior[key].values

        if "log_likelihood" in posdict:
            loglikedict = {
                "log_likelihood": posdict.pop("log_likelihood")
            }
        else:
            if self.log_likelihood_evaluations is not None:
                loglikedict = {
                    "log_likelihood": self.log_likelihood_evaluations
                }
            else:
                loglikedict = None

        priorsamples = None
        if prior is not None:
            if self.priors is None:
                logger.warning(
                    "No priors are in the Result object, so prior samples "
                    "will not be included in the output."
                )
            else:
                priorsamples = self.priors.sample(size=prior)

        azdata = az.from_dict(
            posterior=posdict,
            log_likelihood=loglikedict,
            prior=priorsamples,
        )

        # add attributes
        version = {
            "inference_library": "bilby: {}".format(self.sampler),
            "inference_library_version": get_version_information()
        }

        azdata.posterior.attrs.update(version)
        if "log_likelihood" in azdata._groups:
            azdata.log_likelihood.attrs.update(version)
        if "prior" in azdata._groups:
            azdata.prior.attrs.update(version)

        return azdata


class ResultList(list):

    def __init__(self, results=None, consistency_level="warning"):
        """ A class to store a list of :class:`bilby.core.result.Result` objects
        from equivalent runs on the same data. This provides methods for
        outputting combined results.

        Parameters
        ==========
        results: list
            A list of `:class:`bilby.core.result.Result`.
        consistency_level: str, [ignore, warning, error]
            If warning, print a warning if inconsistencies are discovered
            between the results. If error, raise an error if inconsistencies
            are discovered between the results before combining. If ignore, do
            nothing.

        """
        super(ResultList, self).__init__()
        self.consistency_level = consistency_level
        for result in results:
            self.append(result)

    def append(self, result):
        """
        Append a :class:`bilby.core.result.Result`, or set of results, to the
        list.

        Parameters
        ==========
        result: :class:`bilby.core.result.Result` or filename
            pointing to a result object, to append to the list.
        """

        if isinstance(result, Result):
            super(ResultList, self).append(result)
        elif isinstance(result, (str, os.PathLike)):
            super(ResultList, self).append(read_in_result(result))
        else:
            raise TypeError("Could not append a non-Result type")

    def combine(self, shuffle=False, consistency_level="error"):
        """
        Return the combined results in a :class:bilby.core.result.Result`
        object.

        Parameters
        ----------
        shuffle: bool
            If true, shuffle the samples when combining, otherwise they are concatenated.
        consistency_level: str, [ignore, warning, error]
            Overwrite the class level consistency_level. If warning, print a
            warning if inconsistencies are discovered between the results. If
            error, raise an error if inconsistencies are discovered between
            the results before combining. If ignore, do nothing.

        Returns
        -------
        result: bilby.core.result.Result
            The combined result file

        """

        self.consistency_level = consistency_level

        if len(self) == 0:
            return Result()
        elif len(self) == 1:
            return copy(self[0])
        else:
            result = copy(self[0])

        if result.label is not None:
            result.label += '_combined'

        self.check_consistent_sampler()
        self.check_consistent_data()
        self.check_consistent_parameters()
        self.check_consistent_priors()

        # check which kind of sampler was used: MCMC or Nested Sampling
        if result._nested_samples is not None:
            posteriors, result = self._combine_nested_sampled_runs(result)
        elif result.sampler in ["bilby_mcmc", "bilbymcmc"]:
            posteriors, result = self._combine_mcmc_sampled_runs(result)
        else:
            posteriors = [res.posterior for res in self]

        combined_posteriors = pd.concat(posteriors, ignore_index=True)

        if shuffle:
            result.posterior = combined_posteriors.sample(len(combined_posteriors))
        else:
            result.posterior = combined_posteriors

        logger.info(f"Combined results have {len(result.posterior)} samples")

        return result

    def _combine_nested_sampled_runs(self, result):
        """
        Combine multiple nested sampling runs.

        Currently this keeps posterior samples from each run in proportion with
        the evidence for each individual run

        Parameters
        ==========
        result: bilby.core.result.Result
            The result object to put the new samples in.

        Returns
        =======
        posteriors: list
            A list of pandas DataFrames containing the reduced sample set from
            each run.
        result: bilby.core.result.Result
            The result object with the combined evidences.
        """
        from scipy.special import logsumexp
        self.check_nested_samples()

        # Combine evidences
        log_evidences = np.array([res.log_evidence for res in self])
        result.log_evidence = logsumexp(log_evidences, b=1. / len(self))
        result.log_bayes_factor = result.log_evidence - result.log_noise_evidence

        # Propagate uncertainty in combined evidence
        log_errs = [res.log_evidence_err for res in self if np.isfinite(res.log_evidence_err)]
        if len(log_errs) > 0:
            result.log_evidence_err = 0.5 * logsumexp(2 * np.array(log_errs), b=1. / len(self))
        else:
            result.log_evidence_err = np.nan

        # Combined posteriors with a weighting
        result_weights = np.exp(log_evidences - np.max(log_evidences))
        posteriors = list()
        for res, frac in zip(self, result_weights):
            selected_samples = (random.rng.uniform(size=len(res.posterior)) < frac)
            posteriors.append(res.posterior[selected_samples])

        # remove original nested_samples
        result.nested_samples = None
        result.sampler_kwargs = None
        return posteriors, result

    def _combine_mcmc_sampled_runs(self, result):
        """
        Combine multiple MCMC sampling runs.

        Currently this keeps all posterior samples from each run

        Parameters
        ----------
        result: bilby.core.result.Result
            The result object to put the new samples in.

        Returns
        -------
        posteriors: list
            A list of pandas DataFrames containing the reduced sample set from
            each run.
        result: bilby.core.result.Result
            The result object with the combined evidences.
        """

        from scipy.special import logsumexp

        # Combine evidences
        log_evidences = np.array([res.log_evidence for res in self])
        result.log_evidence = logsumexp(log_evidences, b=1. / len(self))
        result.log_bayes_factor = result.log_evidence - result.log_noise_evidence

        # Propagate uncertainty in combined evidence
        log_errs = [res.log_evidence_err for res in self if np.isfinite(res.log_evidence_err)]
        if len(log_errs) > 0:
            result.log_evidence_err = 0.5 * logsumexp(2 * np.array(log_errs), b=1. / len(self))
        else:
            result.log_evidence_err = np.nan

        # Combined posteriors with a weighting
        posteriors = list()
        for res in self:
            posteriors.append(res.posterior)

        return posteriors, result

    def check_nested_samples(self):
        for res in self:
            try:
                res.nested_samples
            except ValueError:
                raise ResultListError("Not all results contain nested samples")

    def _error_or_warning_consistency(self, msg):
        if self.consistency_level == "error":
            raise ResultListError(msg)
        elif self.consistency_level == "warning":
            logger.warning(msg)
        elif self.consistency_level == "ignore":
            pass
        else:
            raise ValueError(f"Input consistency_level {self.consistency_level} not understood")

    def check_consistent_priors(self):
        for res in self:
            for p in self[0].priors.keys():
                if not self[0].priors[p] == res.priors[p] or len(self[0].priors) != len(res.priors):
                    msg = "Inconsistent priors between results"
                    self._error_or_warning_consistency(msg)

    def check_consistent_parameters(self):
        if not np.all([set(self[0].search_parameter_keys) == set(res.search_parameter_keys) for res in self]):
            msg = "Inconsistent parameters between results"
            self._error_or_warning_consistency(msg)

    def check_consistent_data(self):
        if not np.allclose(
            [res.log_noise_evidence for res in self],
            self[0].log_noise_evidence,
            atol=1e-8,
            rtol=0.0,
            equal_nan=True,
        ):
            msg = "Inconsistent data between results"
            self._error_or_warning_consistency(msg)

    def check_consistent_sampler(self):
        if not np.all([res.sampler == self[0].sampler for res in self]):
            msg = "Inconsistent samplers between results"
            self._error_or_warning_consistency(msg)


@latex_plot_format
def plot_multiple(results, filename=None, labels=None, colours=None,
                  save=True, evidences=False, corner_labels=None, linestyles=None,
                  **kwargs):
    """ Generate a corner plot overlaying two sets of results

    Parameters
    ==========
    results: list
        A list of `bilby.core.result.Result` objects containing the samples to
        plot.
    filename: str
        File name to save the figure to. If None (default), a filename is
        constructed from the outdir of the first element of results and then
        the labels for all the result files.
    labels: list
        List of strings to use when generating a legend. If None (default), the
        `label` attribute of each result in `results` is used.
    colours: list
        The colours for each result. If None, default styles are applied.
    save: bool
        If true, save the figure
    kwargs: dict
        All other keyword arguments are passed to `result.plot_corner` (except
        for the keyword `labels` for which you should use the dedicated
        `corner_labels` input).
        However, `show_titles` and `truths` are ignored since they would be
        ambiguous on such a plot.
    evidences: bool, optional
        Add the log-evidence calculations to the legend. If available, the
        Bayes factor will be used instead.
    corner_labels: list, optional
        List of strings to be passed to the input `labels` to `result.plot_corner`.

    Returns
    =======
    fig:
        A matplotlib figure instance

    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mpllines

    kwargs['show_titles'] = False
    kwargs['truths'] = None
    if corner_labels is not None:
        kwargs['labels'] = corner_labels

    fig = results[0].plot_corner(save=False, **kwargs)
    default_filename = '{}/{}'.format(results[0].outdir, 'combined')
    lines = []
    default_labels = []
    for i, result in enumerate(results):
        if colours:
            c = colours[i]
        else:
            c = 'C{}'.format(i)
        if linestyles is not None:
            linestyle = linestyles[i]
        else:
            linestyle = 'solid'
        hist_kwargs = kwargs.get('hist_kwargs', dict())
        hist_kwargs['color'] = c
        hist_kwargs["linestyle"] = linestyle
        kwargs["hist_kwargs"] = hist_kwargs
        fig = result.plot_corner(fig=fig, save=False, color=c, contour_kwargs={"linestyles": linestyle}, **kwargs)
        default_filename += '_{}'.format(result.label)
        lines.append(mpllines.Line2D([0], [0], color=c, linestyle=linestyle))
        default_labels.append(result.label)

    # Rescale the axes
    for i, ax in enumerate(fig.axes):
        ax.autoscale()
    plt.draw()

    if labels is None:
        labels = default_labels

    labels = sanity_check_labels(labels)

    if evidences:
        if np.isnan(results[0].log_bayes_factor):
            template = r'{label} $\mathrm{{ln}}(Z)={lnz:1.3g}$'
        else:
            template = r'{label} $\mathrm{{ln}}(B)={lnbf:1.3g}$'
        labels = [
            template.format(
                label=label,
                lnz=result.log_evidence,
                lnbf=result.log_bayes_factor,
            )
            for label, result in zip(labels, results)
        ]

    axes = fig.get_axes()
    ndim = int(np.sqrt(len(axes)))
    axes[ndim - 1].legend(lines, labels)

    if filename is None:
        filename = default_filename

    if save:
        safe_save_figure(fig=fig, filename=filename)
    return fig


@latex_plot_format
def make_pp_plot(results, filename=None, save=True, confidence_interval=[0.68, 0.95, 0.997],
                 lines=None, legend_fontsize='x-small', keys=None, title=True,
                 confidence_interval_alpha=0.1, weight_list=None,
                 **kwargs):
    """
    Make a P-P plot for a set of runs with injected signals.

    Parameters
    ==========
    results: list
        A list of Result objects, each of these should have injected_parameters
    filename: str, optional
        The name of the file to save, the default is "outdir/pp.png"
    save: bool, optional
        Whether to save the file, default=True
    confidence_interval: (float, list), optional
        The confidence interval to be plotted, defaulting to 1-2-3 sigma
    lines: list
        If given, a list of matplotlib line formats to use, must be greater
        than the number of parameters.
    legend_fontsize: float
        The font size for the legend
    keys: list
        A list of keys to use, if None defaults to search_parameter_keys
    title: bool
        Whether to add the number of results and total p-value as a plot title
    confidence_interval_alpha: float, list, optional
        The transparency for the background condifence interval
    weight_list: list, optional
        List of the weight arrays for each set of posterior samples.
    kwargs:
        Additional kwargs to pass to matplotlib.pyplot.plot

    Returns
    =======
    fig, pvals:
        matplotlib figure and a NamedTuple with attributes `combined_pvalue`,
        `pvalues`, and `names`.
    """
    import matplotlib.pyplot as plt

    if keys is None:
        keys = results[0].search_parameter_keys

    if weight_list is None:
        weight_list = [None] * len(results)

    credible_levels = list()
    for i, result in enumerate(results):
        credible_levels.append(
            result.get_all_injection_credible_levels(keys, weights=weight_list[i])
        )
    credible_levels = pd.DataFrame(credible_levels)

    if lines is None:
        colors = ["C{}".format(i) for i in range(8)]
        linestyles = ["-", "--", ":"]
        lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    if len(lines) < len(credible_levels.keys()):
        raise ValueError("Larger number of parameters than unique linestyles")

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)
    fig, ax = plt.subplots()

    if isinstance(confidence_interval, float):
        confidence_interval = [confidence_interval]
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(confidence_interval)
    elif len(confidence_interval_alpha) != len(confidence_interval):
        raise ValueError(
            "confidence_interval_alpha must have the same length as confidence_interval")

    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')

    pvalues = []
    logger.info("Key: KS-test p-value")
    for ii, key in enumerate(credible_levels):
        pp = np.array([sum(credible_levels[key].values < xx) /
                       len(credible_levels) for xx in x_values])
        pvalue = scipy.stats.kstest(credible_levels[key], 'uniform').pvalue
        pvalues.append(pvalue)
        logger.info("{}: {}".format(key, pvalue))

        try:
            name = results[0].priors[key].latex_label
        except (AttributeError, KeyError):
            name = key
        label = "{} ({:2.3f})".format(name, pvalue)
        plt.plot(x_values, pp, lines[ii], label=label, **kwargs)

    Pvals = namedtuple('pvals', ['combined_pvalue', 'pvalues', 'names'])
    pvals = Pvals(combined_pvalue=scipy.stats.combine_pvalues(pvalues)[1],
                  pvalues=pvalues,
                  names=list(credible_levels.keys()))
    logger.info(
        "Combined p-value: {}".format(pvals.combined_pvalue))

    if title:
        ax.set_title("N={}, p-value={:2.4f}".format(
            len(results), pvals.combined_pvalue))
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    ax.legend(handlelength=2, labelspacing=0.25, fontsize=legend_fontsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if save:
        if filename is None:
            filename = 'outdir/pp.png'
        safe_save_figure(fig=fig, filename=filename, dpi=500)

    return fig, pvals


def sanity_check_labels(labels):
    """ Check labels for plotting to remove matplotlib errors """
    for ii, lab in enumerate(labels):
        if "_" in lab and "$" not in lab:
            labels[ii] = lab.replace("_", "-")
    return labels


class ResultError(Exception):
    """ Base exception for all Result related errors """


class ResultListError(ResultError):
    """ For Errors occurring during combining results. """


class FileMovedError(ResultError):
    """ Exceptions that occur when files have been moved """
