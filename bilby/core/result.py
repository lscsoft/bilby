from __future__ import division

import os
from distutils.version import LooseVersion
from collections import OrderedDict, namedtuple
from itertools import product

import numpy as np
import pandas as pd
import corner
import json
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import lines as mpllines

from . import utils
from .utils import (logger, infer_parameters_from_function,
                    check_directory_exists_and_if_not_mkdir,
                    BilbyJsonEncoder, decode_bilby_json)
from .prior import Prior, PriorDict, DeltaFunction


def result_file_name(outdir, label, extension='json', gzip=False):
    """ Returns the standard filename used for a result file

    Parameters
    ----------
    outdir: str
        Name of the output directory
    label: str
        Naming scheme of the output file
    extension: str, optional
        Whether to save as `hdf5` or `json`
    gzip: bool, optional
        Set to True to append `.gz` to the extension for saving in gzipped format

    Returns
    -------
    str: File name of the output file
    """
    if extension in ['json', 'hdf5']:
        if extension == 'json' and gzip:
            return '{}/{}_result.{}.gz'.format(outdir, label, extension)
        else:
            return '{}/{}_result.{}'.format(outdir, label, extension)
    else:
        raise ValueError("Extension type {} not understood".format(extension))


def _determine_file_name(filename, outdir, label, extension, gzip):
    """ Helper method to determine the filename """
    if filename is not None:
        return filename
    else:
        if (outdir is None) and (label is None):
            raise ValueError("No information given to load file")
        else:
            return result_file_name(outdir, label, extension, gzip)


def read_in_result(filename=None, outdir=None, label=None, extension='json', gzip=False):
    """ Reads in a stored bilby result object

    Parameters
    ----------
    filename: str
        Path to the file to be read (alternative to giving the outdir and label)
    outdir, label, extension: str
        Name of the output directory, label and extension used for the default
        naming scheme.

    """
    filename = _determine_file_name(filename, outdir, label, extension, gzip)

    # Get the actual extension (may differ from the default extension if the filename is given)
    extension = os.path.splitext(filename)[1].lstrip('.')
    if extension == 'gz':  # gzipped file
        extension = os.path.splitext(os.path.splitext(filename)[0])[1].lstrip('.')

    if 'json' in extension:
        result = Result.from_json(filename=filename)
    elif ('hdf5' in extension) or ('h5' in extension):
        result = Result.from_hdf5(filename=filename)
    elif extension is None:
        raise ValueError("No filetype extension provided")
    else:
        raise ValueError("Filetype {} not understood".format(extension))
    return result


class Result(object):
    def __init__(self, label='no_label', outdir='.', sampler=None,
                 search_parameter_keys=None, fixed_parameter_keys=None,
                 constraint_parameter_keys=None, priors=None,
                 sampler_kwargs=None, injection_parameters=None,
                 meta_data=None, posterior=None, samples=None,
                 nested_samples=None, log_evidence=np.nan,
                 log_evidence_err=np.nan, log_noise_evidence=np.nan,
                 log_bayes_factor=np.nan, log_likelihood_evaluations=None,
                 log_prior_evaluations=None, sampling_time=None, nburn=None,
                 walkers=None, max_autocorrelation_time=None,
                 parameter_labels=None, parameter_labels_with_unit=None,
                 gzip=False, version=None):
        """ A class to store the results of the sampling run

        Parameters
        ----------
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
        log_likelihood_evaluations: array_like
            The evaluations of the likelihood for each sample point
        log_prior_evaluations: array_like
            The evaluations of the prior for each sample point
        sampling_time: float
            The time taken to complete the sampling
        nburn: int
            The number of burn-in steps discarded for MCMC samplers
        walkers: array_like
            The samplers taken by a ensemble MCMC samplers
        max_autocorrelation_time: float
            The estimated maximum autocorrelation time for MCMC samplers
        parameter_labels, parameter_labels_with_unit: list
            Lists of the latex-formatted parameter labels
        gzip: bool
            Set to True to gzip the results file (if using json format)
        version: str,
            Version information for software used to generate the result. Note,
            this information is generated when the result object is initialized

        Note
        ---------
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
        self.nested_samples = nested_samples
        self.walkers = walkers
        self.nburn = nburn
        self.log_evidence = log_evidence
        self.log_evidence_err = log_evidence_err
        self.log_noise_evidence = log_noise_evidence
        self.log_bayes_factor = log_bayes_factor
        self.log_likelihood_evaluations = log_likelihood_evaluations
        self.log_prior_evaluations = log_prior_evaluations
        self.sampling_time = sampling_time
        self.version = version
        self.max_autocorrelation_time = max_autocorrelation_time

        self.prior_values = None
        self._kde = None

    @classmethod
    def from_hdf5(cls, filename=None, outdir=None, label=None):
        """ Read in a saved .h5 data file

        Parameters
        ----------
        filename: str
            If given, try to load from this filename
        outdir, label: str
            If given, use the default naming convention for saved results file

        Returns
        -------
        result: bilby.core.result.Result

        Raises
        -------
        ValueError: If no filename is given and either outdir or label is None
                    If no bilby.core.result.Result is found in the path

        """
        import deepdish
        filename = _determine_file_name(filename, outdir, label, 'hdf5', False)

        if os.path.isfile(filename):
            dictionary = deepdish.io.load(filename)
            # Some versions of deepdish/pytables return the dictionanary as
            # a dictionary with a key 'data'
            if len(dictionary) == 1 and 'data' in dictionary:
                dictionary = dictionary['data']
            try:
                if isinstance(dictionary.get('posterior', None), dict):
                    dictionary['posterior'] = pd.DataFrame(dictionary['posterior'])
                return cls(**dictionary)
            except TypeError as e:
                raise IOError("Unable to load dictionary, error={}".format(e))
        else:
            raise IOError("No result '{}' found".format(filename))

    @classmethod
    def from_json(cls, filename=None, outdir=None, label=None, gzip=False):
        """ Read in a saved .json data file

        Parameters
        ----------
        filename: str
            If given, try to load from this filename
        outdir, label: str
            If given, use the default naming convention for saved results file

        Returns
        -------
        result: bilby.core.result.Result

        Raises
        -------
        ValueError: If no filename is given and either outdir or label is None
                    If no bilby.core.result.Result is found in the path

        """
        filename = _determine_file_name(filename, outdir, label, 'json', gzip)

        if os.path.isfile(filename):
            if gzip or os.path.splitext(filename)[1].lstrip('.') == 'gz':
                import gzip
                with gzip.GzipFile(filename, 'r') as file:
                    json_str = file.read().decode('utf-8')
                dictionary = json.loads(json_str, object_hook=decode_bilby_json)
            else:
                with open(filename, 'r') as file:
                    dictionary = json.load(file, object_hook=decode_bilby_json)
            for key in dictionary.keys():
                # Convert the loaded priors to bilby prior type
                if key == 'priors':
                    for param in dictionary[key].keys():
                        dictionary[key][param] = str(dictionary[key][param])
                    dictionary[key] = PriorDict(dictionary[key])
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
                        "log_noise_evidence: {:6.3f}\n"
                        "log_evidence: {:6.3f} +/- {:6.3f}\n"
                        "log_bayes_factor: {:6.3f} +/- {:6.3f}\n"
                        .format(len(self.posterior), self.log_noise_evidence, self.log_evidence,
                                self.log_evidence_err, self.log_bayes_factor,
                                self.log_evidence_err))
            else:
                return ("nsamples: {:d}\n"
                        "log_evidence: {:6.3f} +/- {:6.3f}\n"
                        .format(len(self.posterior), self.log_evidence, self.log_evidence_err))
        else:
            return ''

    @property
    def priors(self):
        if self._priors is not None:
            return self._priors
        else:
            raise ValueError('Result object has no priors')

    @priors.setter
    def priors(self, priors):
        if isinstance(priors, dict):
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
            'sampling_time', 'sampler_kwargs',
            'log_likelihood_evaluations', 'log_prior_evaluations', 'samples',
            'nested_samples', 'walkers', 'nburn', 'parameter_labels',
            'parameter_labels_with_unit', 'version']
        dictionary = OrderedDict()
        for attr in save_attrs:
            try:
                dictionary[attr] = getattr(self, attr)
            except ValueError as e:
                logger.debug("Unable to save {}, message: {}".format(attr, e))
                pass
        return dictionary

    def save_to_file(self, overwrite=False, outdir=None, extension='json', gzip=False):
        """
        Writes the Result to a json or deepdish h5 file

        Parameters
        ----------
        overwrite: bool, optional
            Whether or not to overwrite an existing result file.
            default=False
        outdir: str, optional
            Path to the outdir. Default is the one stored in the result object.
        extension: str, optional {json, hdf5, True}
            Determines the method to use to store the data (if True defaults
            to json)
        gzip: bool, optional
            If true, and outputing to a json file, this will gzip the resulting
            file and add '.gz' to the file extension.
        """

        if extension is True:
            extension = "json"

        outdir = self._safe_outdir_creation(outdir, self.save_to_file)
        file_name = result_file_name(outdir, self.label, extension, gzip)

        if os.path.isfile(file_name):
            if overwrite:
                logger.debug('Removing existing file {}'.format(file_name))
                os.remove(file_name)
            else:
                logger.debug(
                    'Renaming existing file {} to {}.old'.format(file_name,
                                                                 file_name))
                os.rename(file_name, file_name + '.old')

        logger.debug("Saving result to {}".format(file_name))

        # Convert the prior to a string representation for saving on disk
        dictionary = self._get_save_data_dictionary()
        if dictionary.get('priors', False):
            dictionary['priors'] = {key: str(self.priors[key]) for key in self.priors}

        # Convert callable sampler_kwargs to strings
        if dictionary.get('sampler_kwargs', None) is not None:
            for key in dictionary['sampler_kwargs']:
                if hasattr(dictionary['sampler_kwargs'][key], '__call__'):
                    dictionary['sampler_kwargs'][key] = str(dictionary['sampler_kwargs'])

        try:
            if extension == 'json':
                if gzip:
                    import gzip
                    # encode to a string
                    json_str = json.dumps(dictionary, cls=BilbyJsonEncoder).encode('utf-8')
                    with gzip.GzipFile(file_name, 'w') as file:
                        file.write(json_str)
                else:
                    with open(file_name, 'w') as file:
                        json.dump(dictionary, file, indent=2, cls=BilbyJsonEncoder)
            elif extension == 'hdf5':
                import deepdish
                for key in dictionary:
                    if isinstance(dictionary[key], pd.DataFrame):
                        dictionary[key] = dictionary[key].to_dict()
                deepdish.io.save(file_name, dictionary)
            else:
                raise ValueError("Extension type {} not understood".format(extension))
        except Exception as e:
            logger.error("\n\n Saving the data has failed with the "
                         "following message:\n {} \n\n".format(e))

    def save_posterior_samples(self, outdir=None):
        """Saves posterior samples to a file"""
        outdir = self._safe_outdir_creation(outdir, self.save_posterior_samples)
        filename = '{}/{}_posterior_samples.txt'.format(outdir, self.label)
        self.posterior.to_csv(filename, index=False, header=True)

    def get_latex_labels_from_parameter_keys(self, keys):
        """ Returns a list of latex_labels corresponding to the given keys

        Parameters
        ----------
        keys: list
            List of strings corresponding to the desired latex_labels

        Returns
        -------
        list: The desired latex_labels

        """
        latex_labels = []
        for k in keys:
            if k in self.search_parameter_keys:
                idx = self.search_parameter_keys.index(k)
                latex_labels.append(self.parameter_labels_with_unit[idx])
            elif k in self.parameter_labels:
                latex_labels.append(k)
            else:
                logger.debug(
                    'key {} not a parameter label or latex label'.format(k))
                latex_labels.append(' '.join(k.split('_')))
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

    def get_one_dimensional_median_and_error_bar(self, key, fmt='.2f',
                                                 quantiles=(0.16, 0.84)):
        """ Calculate the median and error bar for a given key

        Parameters
        ----------
        key: str
            The parameter key for which to calculate the median and error bar
        fmt: str, ('.2f')
            A format string
        quantiles: list, tuple
            A length-2 tuple of the lower and upper-quantiles to calculate
            the errors bars for.

        Returns
        -------
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

    def plot_single_density(self, key, prior=None, cumulative=False,
                            title=None, truth=None, save=True,
                            file_base_name=None, bins=50, label_fontsize=16,
                            title_fontsize=16, quantiles=(0.16, 0.84), dpi=300):
        """ Plot a 1D marginal density, either probability or cumulative.

        Parameters
        ----------
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
        -------
        figure: matplotlib.pyplot.figure
            A matplotlib figure object
        """
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
            ax.plot(theta, prior.prob(theta), color='C2')

        if save:
            fig.tight_layout()
            if cumulative:
                file_name = file_base_name + key + '_cdf'
            else:
                file_name = file_base_name + key + '_pdf'
            fig.savefig(file_name, dpi=dpi)
            plt.close(fig)
        else:
            return fig

    def plot_marginals(self, parameters=None, priors=None, titles=True,
                       file_base_name=None, bins=50, label_fontsize=16,
                       title_fontsize=16, quantiles=(0.16, 0.84), dpi=300,
                       outdir=None):
        """ Plot 1D marginal distributions

        Parameters
        ----------
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
        -------
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

    def plot_corner(self, parameters=None, priors=None, titles=True, save=True,
                    filename=None, dpi=300, **kwargs):
        """ Plot a corner-plot

        Parameters
        ----------
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
        -----
            The generation of the corner plot themselves is done by the corner
            python module, see https://corner.readthedocs.io for more
            information.

        Returns
        -------
        fig:
            A matplotlib figure instance

        """

        # If in testing mode, not corner plots are generated
        if utils.command_line_args.test:
            return

        # bilby default corner kwargs. Overwritten by anything passed to kwargs
        defaults_kwargs = dict(
            bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
            title_kwargs=dict(fontsize=16), color='#0072C1',
            truth_color='tab:orange', quantiles=[0.16, 0.84],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, plot_datapoints=True, fill_contours=True,
            max_n_ticks=3)

        if LooseVersion(matplotlib.__version__) < "2.1":
            defaults_kwargs['hist_kwargs'] = dict(normed=True)
        else:
            defaults_kwargs['hist_kwargs'] = dict(density=True)

        if 'lionize' in kwargs and kwargs['lionize'] is True:
            defaults_kwargs['truth_color'] = 'tab:blue'
            defaults_kwargs['color'] = '#FF8C00'

        defaults_kwargs.update(kwargs)
        kwargs = defaults_kwargs

        # Handle if truths was passed in
        if 'truth' in kwargs:
            kwargs['truths'] = kwargs.pop('truth')
        if kwargs.get('truths'):
            truths = kwargs.get('truths')
            if isinstance(parameters, list) and isinstance(truths, list):
                if len(parameters) != len(truths):
                    raise ValueError(
                        "Length of parameters and truths don't match")
            elif isinstance(truths, dict) and parameters is None:
                parameters = kwargs.pop('truths')
            else:
                raise ValueError(
                    "Combination of parameters and truths not understood")

        # If injection parameters where stored, use these as parameter values
        # but do not overwrite input parameters (or truths)
        cond1 = getattr(self, 'injection_parameters', None) is not None
        cond2 = parameters is None
        if cond1 and cond2:
            parameters = {key: self.injection_parameters[key] for key in
                          self.search_parameter_keys}

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

        # Unless already set, set the range to include all samples
        # This prevents ValueErrors being raised for parameters with no range
        kwargs['range'] = kwargs.get('range', [1] * len(plot_parameter_keys))

        # Create the data array to plot and pass everything to corner
        xs = self.posterior[plot_parameter_keys].values
        fig = corner.corner(xs, **kwargs)
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
            fig.savefig(filename, dpi=dpi)
            plt.close(fig)

        return fig

    def plot_walkers(self, **kwargs):
        """ Method to plot the trace of the walkers in an ensemble MCMC plot """
        if hasattr(self, 'walkers') is False:
            logger.warning("Cannot plot_walkers as no walkers are saved")
            return

        if utils.command_line_args.test:
            return

        nwalkers, nsteps, ndim = self.walkers.shape
        idxs = np.arange(nsteps)
        fig, axes = plt.subplots(nrows=ndim, figsize=(6, 3 * ndim))
        walkers = self.walkers[:, :, :]
        for i, ax in enumerate(axes):
            ax.plot(idxs[:self.nburn + 1], walkers[:, :self.nburn + 1, i].T,
                    lw=0.1, color='r')
            ax.set_ylabel(self.parameter_labels[i])

        for i, ax in enumerate(axes):
            ax.plot(idxs[self.nburn:], walkers[:, self.nburn:, i].T, lw=0.1,
                    color='k')
            ax.set_ylabel(self.parameter_labels[i])

        fig.tight_layout()
        outdir = self._safe_outdir_creation(kwargs.get('outdir'), self.plot_walkers)
        filename = '{}/{}_walkers.png'.format(outdir, self.label)
        logger.debug('Saving walkers plot to {}'.format('filename'))
        fig.savefig(filename)
        plt.close(fig)

    def plot_with_data(self, model, x, y, ndraws=1000, npoints=1000,
                       xlabel=None, ylabel=None, data_label='data',
                       data_fmt='o', draws_label=None, filename=None,
                       maxl_label='max likelihood', dpi=300, outdir=None):
        """ Generate a figure showing the data and fits to the data

        Parameters
        ----------
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
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        ax.legend(numpoints=3)
        fig.tight_layout()
        if filename is None:
            outdir = self._safe_outdir_creation(outdir, self.plot_with_data)
            filename = '{}/{}_plot_with_data'.format(outdir, self.label)
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)

    @staticmethod
    def _add_prior_fixed_values_to_posterior(posterior, priors):
        if priors is None:
            return posterior
        for key in priors:
            if isinstance(priors[key], DeltaFunction):
                posterior[key] = priors[key].peak
            elif isinstance(priors[key], float):
                posterior[key] = priors[key]
        return posterior

    def samples_to_posterior(self, likelihood=None, priors=None,
                             conversion_function=None):
        """
        Convert array of samples to posterior (a Pandas data frame)

        Also applies the conversion function to any stored posterior

        Parameters
        ----------
        likelihood: bilby.likelihood.GravitationalWaveTransient, optional
            GravitationalWaveTransient likelihood used for sampling.
        priors: dict, optional
            Dictionary of prior object, used to fill in delta function priors.
        conversion_function: function, optional
            Function which adds in extra parameters to the data frame,
            should take the data_frame, likelihood and prior as arguments.
        """
        try:
            data_frame = self.posterior
        except ValueError:
            data_frame = pd.DataFrame(
                self.samples, columns=self.search_parameter_keys)
            data_frame = self._add_prior_fixed_values_to_posterior(
                data_frame, priors)
            data_frame['log_likelihood'] = getattr(
                self, 'log_likelihood_evaluations', np.nan)
            if self.log_prior_evaluations is None:
                ln_prior = list()
                for ii in range(len(data_frame)):
                    ln_prior.append(
                        self.priors.ln_prob(dict(
                            data_frame[self.search_parameter_keys].iloc[ii])))
                data_frame['log_prior'] = np.array(ln_prior)
            else:
                data_frame['log_prior'] = self.log_prior_evaluations
        if conversion_function is not None:
            data_frame = conversion_function(data_frame, likelihood, priors)
        self.posterior = data_frame

    def calculate_prior_values(self, priors):
        """
        Evaluate prior probability for each parameter for each sample.

        Parameters
        ----------
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

    def get_all_injection_credible_levels(self, keys=None):
        """
        Get credible levels for all parameters

        Parameters
        ----------
        keys: list, optional
            A list of keys for which return the credible levels, if None,
            defaults to search_parameter_keys

        Returns
        -------
        credible_levels: dict
            The credible levels at which the injected parameters are found.
        """
        if keys is None:
            keys = self.search_parameter_keys
        if self.injection_parameters is None:
            raise(TypeError, "Result object has no 'injection_parameters'. "
                             "Cannot compute credible levels.")
        credible_levels = {key: self.get_injection_credible_level(key)
                           for key in keys
                           if isinstance(self.injection_parameters[key], float)}
        return credible_levels

    def get_injection_credible_level(self, parameter):
        """
        Get the credible level of the injected parameter

        Calculated as CDF(injection value)

        Parameters
        ----------
        parameter: str
            Parameter to get credible level for
        Returns
        -------
        float: credible level
        """
        if self.injection_parameters is None:
            raise(TypeError, "Result object has no 'injection_parameters'. "
                             "Cannot copmute credible levels.")
        if parameter in self.posterior and\
                parameter in self.injection_parameters:
            credible_level =\
                sum(self.posterior[parameter].values <
                    self.injection_parameters[parameter]) / len(self.posterior)
            return credible_level
        else:
            return np.nan

    def _check_attribute_match_to_other_object(self, name, other_object):
        """ Check attribute name exists in other_object and is the same

        Parameters
        ----------
        name: str
            Name of the attribute in this instance
        other_object: object
            Other object with attributes to compare with

        Returns
        -------
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
        """ Calculate the posterior probabily for a new sample

        This queries a Kernel Density Estimate of the posterior to calculate
        the posterior probability density for the new sample.

        Parameters
        ----------
        sample: dict, or list of dictionaries
            A dictionary containing all the keys from
            self.search_parameter_keys and corresponding values at which to
            calculate the posterior probability

        Returns
        -------
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


def plot_multiple(results, filename=None, labels=None, colours=None,
                  save=True, evidences=False, **kwargs):
    """ Generate a corner plot overlaying two sets of results

    Parameters
    ----------
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
        All other keyword arguments are passed to `result.plot_corner`.
        However, `show_titles` and `truths` are ignored since they would be
        ambiguous on such a plot.
    evidences: bool, optional
        Add the log-evidence calculations to the legend. If available, the
        Bayes factor will be used instead.

    Returns
    -------
    fig:
        A matplotlib figure instance

    """

    kwargs['show_titles'] = False
    kwargs['truths'] = None

    fig = results[0].plot_corner(save=False, **kwargs)
    default_filename = '{}/{}'.format(results[0].outdir, 'combined')
    lines = []
    default_labels = []
    for i, result in enumerate(results):
        if colours:
            c = colours[i]
        else:
            c = 'C{}'.format(i)
        hist_kwargs = kwargs.get('hist_kwargs', dict())
        hist_kwargs['color'] = c
        fig = result.plot_corner(fig=fig, save=False, color=c, **kwargs)
        default_filename += '_{}'.format(result.label)
        lines.append(mpllines.Line2D([0], [0], color=c))
        default_labels.append(result.label)

    # Rescale the axes
    for i, ax in enumerate(fig.axes):
        ax.autoscale()
    plt.draw()

    if labels is None:
        labels = default_labels

    if evidences:
        if np.isnan(results[0].log_bayes_factor):
            template = ' $\mathrm{{ln}}(Z)={lnz:1.3g}$'
        else:
            template = ' $\mathrm{{ln}}(B)={lnbf:1.3g}$'
        labels = [template.format(lnz=result.log_evidence,
                                  lnbf=result.log_bayes_factor)
                  for ii, result in enumerate(results)]

    axes = fig.get_axes()
    ndim = int(np.sqrt(len(axes)))
    axes[ndim - 1].legend(lines, labels)

    if filename is None:
        filename = default_filename

    if save:
        fig.savefig(filename)
    return fig


def make_pp_plot(results, filename=None, save=True, confidence_interval=0.9,
                 lines=None, legend_fontsize=9, keys=None, **kwargs):
    """
    Make a P-P plot for a set of runs with injected signals.

    Parameters
    ----------
    results: list
        A list of Result objects, each of these should have injected_parameters
    filename: str, optional
        The name of the file to save, the default is "outdir/pp.png"
    save: bool, optional
        Whether to save the file, default=True
    confidence_interval: float, optional
        The confidence interval to be plotted, defaulting to 0.9 (90%)
    lines: list
        If given, a list of matplotlib line formats to use, must be greater
        than the number of parameters.
    legend_fontsize: float
        The font size for the legend
    keys: list
        A list of keys to use, if None defaults to search_parameter_keys
    kwargs:
        Additional kwargs to pass to matplotlib.pyplot.plot

    Returns
    -------
    fig:
        matplotlib figure
    """

    credible_levels = pd.DataFrame()
    for result in results:
        credible_levels = credible_levels.append(
            result.get_all_injection_credible_levels(keys), ignore_index=True)

    if lines is None:
        colors = ["C{}".format(i) for i in range(8)]
        linestyles = ["-", "--", ":"]
        lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    if len(lines) < len(credible_levels.keys()):
        raise ValueError("Larger number of parameters than unique linestyles")

    x_values = np.linspace(0, 1, 1001)

    # Putting in the confidence bands
    N = len(credible_levels)
    edge_of_bound = (1. - confidence_interval) / 2.
    lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
    upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
    # The binomial point percent function doesn't always return 0 @ 0,
    # so set those bounds explicitly to be sure
    lower[0] = 0
    upper[0] = 0
    fig, ax = plt.subplots()

    ax.fill_between(x_values, lower, upper, alpha=0.2, color='k')

    for ii, key in enumerate(credible_levels):
        pp = np.array([sum(credible_levels[key].values < xx) /
                       len(credible_levels) for xx in x_values])
        plt.plot(x_values, pp, lines[ii], label=key, **kwargs)

    ax.legend(fontsize=legend_fontsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if save:
        if filename is None:
            filename = 'outdir/pp.png'
        fig.savefig(filename, dpi=500)
    return fig


class ResultError(Exception):
    """ Base exception for all Result related errors """


class FileMovedError(ResultError):
    """ Exceptions that occur when files have been moved """
