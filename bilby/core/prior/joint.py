import re

import numpy as np
import scipy.stats
from scipy.special import erfinv

from .base import Prior, PriorException
from ..utils import logger, infer_args_from_method, get_dict_with_properties
from ..utils import random


class BaseJointPriorDist(object):
    def __init__(self, names, bounds=None):
        """
        A class defining JointPriorDist that will be overwritten with child
        classes defining the joint prior distributions between given parameters,


        Parameters
        ==========
        names: list (required)
            A list of the parameter names in the JointPriorDist. The
            listed parameters must have the same order that they appear in
            the lists of statistical parameters that may be passed in child class
        bounds: list (optional)
            A list of bounds on each parameter. The defaults are for bounds at
            +/- infinity.
        """
        self.distname = "joint_dist"
        if not isinstance(names, list):
            self.names = [names]
        else:
            self.names = names

        self.num_vars = len(self.names)

        # set the bounds for each parameter
        if isinstance(bounds, list):
            if len(bounds) != len(self):
                raise ValueError("Wrong number of parameter bounds")

            # check bounds
            for bound in bounds:
                if isinstance(bounds, (list, tuple, np.ndarray)):
                    if len(bound) != 2:
                        raise ValueError(
                            "Bounds must contain an upper and lower value."
                        )
                    else:
                        if bound[1] <= bound[0]:
                            raise ValueError("Bounds are not properly set")
                else:
                    raise TypeError("Bound must be a list")
        else:
            bounds = [(-np.inf, np.inf) for _ in self.names]
        self.bounds = {name: val for name, val in zip(self.names, bounds)}

        self._current_sample = {}  # initialise empty sample
        self._uncorrelated = None
        self._current_lnprob = None

        # a dictionary of the parameters as requested by the prior
        self.requested_parameters = dict()
        self.reset_request()

        # a dictionary of the rescaled parameters
        self.rescale_parameters = dict()
        self.reset_rescale()

        # a list of sampled parameters
        self.reset_sampled()

    def reset_sampled(self):
        self.sampled_parameters = []
        self.current_sample = {}

    def filled_request(self):
        """
        Check if all requested parameters have been filled.
        """

        return not np.any([val is None for val in self.requested_parameters.values()])

    def reset_request(self):
        """
        Reset the requested parameters to None.
        """

        for name in self.names:
            self.requested_parameters[name] = None

    def filled_rescale(self):
        """
        Check if all the rescaled parameters have been filled.
        """

        return not np.any([val is None for val in self.rescale_parameters.values()])

    def reset_rescale(self):
        """
        Reset the rescaled parameters to None.
        """

        for name in self.names:
            self.rescale_parameters[name] = None

    def get_instantiation_dict(self):
        subclass_args = infer_args_from_method(self.__init__)
        dict_with_properties = get_dict_with_properties(self)
        instantiation_dict = dict()
        for key in subclass_args:
            if isinstance(dict_with_properties[key], list):
                value = np.asarray(dict_with_properties[key]).tolist()
            else:
                value = dict_with_properties[key]
            instantiation_dict[key] = value
        return instantiation_dict

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        """Overrides the special method __repr__.

        Returns a representation of this instance that resembles how it is instantiated.
        Works correctly for all child classes

        Returns
        =======
        str: A string representation of this instance

        """
        dist_name = self.__class__.__name__
        instantiation_dict = self.get_instantiation_dict()
        args = ", ".join(
            [
                "{}={}".format(key, repr(instantiation_dict[key]))
                for key in instantiation_dict
            ]
        )
        return "{}({})".format(dist_name, args)

    @classmethod
    def from_repr(cls, string):
        """Generate the distribution from its __repr__"""
        return cls._from_repr(string)

    @classmethod
    def _from_repr(cls, string):
        subclass_args = infer_args_from_method(cls.__init__)

        string = string.replace(" ", "")
        kwargs = cls._split_repr(string)
        for key in kwargs:
            val = kwargs[key]
            if key not in subclass_args:
                raise AttributeError(
                    "Unknown argument {} for class {}".format(key, cls.__name__)
                )
            else:
                kwargs[key.strip()] = Prior._parse_argument_string(val)

        return cls(**kwargs)

    @classmethod
    def _split_repr(cls, string):
        string = string.replace(",", ", ")
        # see https://stackoverflow.com/a/72146415/1862861
        args = re.findall(r"(\w+)=(\[.*?]|{.*?}|\S+)(?=\s*,\s*\w+=|\Z)", string)
        kwargs = dict()
        for key, arg in args:
            kwargs[key.strip()] = arg
        return kwargs

    def prob(self, samp):
        """
        Get the probability of a sample. For bounded priors the
        probability will not be properly normalised.
        """

        return np.exp(self.ln_prob(samp))

    def _check_samp(self, value):
        """
        Get the log-probability of a sample. For bounded priors the
        probability will not be properly normalised.

        Parameters
        ==========
        value: array_like
            A 1d vector of the sample, or 2d array of sample values with shape
            NxM, where N is the number of samples and M is the number of
            parameters.

        Returns
        =======
        samp: array_like
            returns the input value as a sample array
        outbounds: array_like
            Boolean Array that selects samples in samp that are out of given bounds
        """
        samp = np.array(value)
        if len(samp.shape) == 1:
            samp = samp.reshape(1, self.num_vars)

        if len(samp.shape) != 2:
            raise ValueError("Array is the wrong shape")
        elif samp.shape[1] != self.num_vars:
            raise ValueError("Array is the wrong shape")

        # check sample(s) is within bounds
        outbounds = np.ones(samp.shape[0], dtype=bool)
        for s, bound in zip(samp.T, self.bounds.values()):
            outbounds = (s < bound[0]) | (s > bound[1])
            if np.any(outbounds):
                break
        return samp, outbounds

    def ln_prob(self, value):
        """
        Get the log-probability of a sample. For bounded priors the
        probability will not be properly normalised.

        Parameters
        ==========
        value: array_like
            A 1d vector of the sample, or 2d array of sample values with shape
            NxM, where N is the number of samples and M is the number of
            parameters.
        """

        samp, outbounds = self._check_samp(value)
        lnprob = -np.inf * np.ones(samp.shape[0])
        lnprob = self._ln_prob(samp, lnprob, outbounds)
        if samp.shape[0] == 1:
            return lnprob[0]
        else:
            return lnprob

    def _ln_prob(self, samp, lnprob, outbounds):
        """
        Get the log-probability of a sample. For bounded priors the
        probability will not be properly normalised. **this method needs overwritten by child class**

        Parameters
        ==========
        samp: vector
            sample to evaluate the ln_prob at
        lnprob: vector
            of -inf passed in with the same shape as the number of samples
        outbounds: array_like
            boolean array showing which samples in lnprob vector are out of the given bounds

        Returns
        =======
        lnprob: vector
            array of lnprob values for each sample given
        """
        """
        Here is where the subclass where overwrite ln_prob method
        """
        return lnprob

    def sample(self, size=1, **kwargs):
        """
        Draw, and set, a sample from the Dist, accompanying method _sample needs to overwritten

        Parameters
        ==========
        size: int
            number of samples to generate, defaults to 1
        """

        if size is None:
            size = 1
        samps = self._sample(size=size, **kwargs)
        for i, name in enumerate(self.names):
            if size == 1:
                self.current_sample[name] = samps[:, i].flatten()[0]
            else:
                self.current_sample[name] = samps[:, i].flatten()

    def _sample(self, size, **kwargs):
        """
        Draw, and set, a sample from the joint dist (**needs to be ovewritten by child class**)

        Parameters
        ==========
        size: int
            number of samples to generate, defaults to 1
        """
        samps = np.zeros((size, len(self)))
        """
        Here is where the subclass where overwrite sampling method
        """
        return samps

    def rescale(self, value, **kwargs):
        """
        Rescale from a unit hypercube to JointPriorDist. Note that no
        bounds are applied in the rescale function. (child classes need to
        overwrite accompanying method _rescale().

        Parameters
        ==========
        value: array
            A 1d vector sample (one for each parameter) drawn from a uniform
            distribution between 0 and 1, or a 2d NxM array of samples where
            N is the number of samples and M is the number of parameters.
        kwargs: dict
            All keyword args that need to be passed to _rescale method, these keyword
            args are called in the JointPrior rescale methods for each parameter

        Returns
        =======
        array:
            An vector sample drawn from the multivariate Gaussian
            distribution.
        """
        samp = np.array(value)
        if len(samp.shape) == 1:
            samp = samp.reshape(1, self.num_vars)

        if len(samp.shape) != 2:
            raise ValueError("Array is the wrong shape")
        elif samp.shape[1] != self.num_vars:
            raise ValueError("Array is the wrong shape")

        samp = self._rescale(samp, **kwargs)
        return np.squeeze(samp)

    def _rescale(self, samp, **kwargs):
        """
        rescale a sample from a unit hypercybe to the joint dist (**needs to be ovewritten by child class**)

        Parameters
        ==========
        samp: numpy array
            this is a vector sample drawn from a uniform distribution to be rescaled to the distribution
        """
        """
        Here is where the subclass where overwrite rescale method
        """
        return samp

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        return self.get_instantiation_dict() == other.get_instantiation_dict()


class MultivariateGaussianDist(BaseJointPriorDist):
    def __init__(
        self,
        names,
        nmodes=1,
        mus=None,
        sigmas=None,
        corrcoefs=None,
        covs=None,
        weights=None,
        bounds=None,
    ):
        """
        A class defining a multi-variate Gaussian, allowing multiple modes for
        a Gaussian mixture model.

        Note: if using a multivariate Gaussian prior, with bounds, this can
        lead to biases in the marginal likelihood estimate and posterior
        estimate for nested samplers routines that rely on sampling from a unit
        hypercube and having a prior transform, e.g., nestle, dynesty and
        MultiNest.

        Parameters
        ==========
        names: list
            A list of the parameter names in the multivariate Gaussian. The
            listed parameters must have the same order that they appear in
            the lists of means, standard deviations, and the correlation
            coefficient, or covariance, matrices.
        nmodes: int
            The number of modes for the mixture model. This defaults to 1,
            which will be checked against the shape of the other inputs.
        mus: array_like
            A list of lists of means of each mode in a multivariate Gaussian
            mixture model. A single list can be given for a single mode. If
            this is None then means at zero will be assumed.
        sigmas: array_like
            A list of lists of the standard deviations of each mode of the
            multivariate Gaussian. If supplying a correlation coefficient
            matrix rather than a covariance matrix these values must be given.
            If this is None unit variances will be assumed.
        corrcoefs: array
            A list of square matrices containing the correlation coefficients
            of the parameters for each mode. If this is None it will be assumed
            that the parameters are uncorrelated.
        covs: array
            A list of square matrices containing the covariance matrix of the
            multivariate Gaussian.
        weights: list
            A list of weights (relative probabilities) for each mode of the
            multivariate Gaussian. This will default to equal weights for each
            mode.
        bounds: list
            A list of bounds on each parameter. The defaults are for bounds at
            +/- infinity.
        """
        super(MultivariateGaussianDist, self).__init__(names=names, bounds=bounds)
        for name in self.names:
            bound = self.bounds[name]
            if bound[0] != -np.inf or bound[1] != np.inf:
                logger.warning(
                    "If using bounded ranges on the multivariate "
                    "Gaussian this will lead to biased posteriors "
                    "for nested sampling routines that require "
                    "a prior transform."
                )
        self.distname = "mvg"
        self.mus = []
        self.covs = []
        self.corrcoefs = []
        self.sigmas = []
        self.logprodsigmas = []   # log of product of sigmas, needed for "standard" multivariate normal
        self.weights = []
        self.eigvalues = []
        self.eigvectors = []
        self.sqeigvalues = []  # square root of the eigenvalues
        self.mvn = []  # list of multivariate normal distributions

        # put values in lists if required
        if nmodes == 1:
            if mus is not None:
                if len(np.shape(mus)) == 1:
                    mus = [mus]
                elif len(np.shape(mus)) == 0:
                    raise ValueError("Must supply a list of means")
            if sigmas is not None:
                if len(np.shape(sigmas)) == 1:
                    sigmas = [sigmas]
                elif len(np.shape(sigmas)) == 0:
                    raise ValueError("Must supply a list of standard deviations")
            if covs is not None:
                if isinstance(covs, np.ndarray):
                    covs = [covs]
                elif isinstance(covs, list):
                    if len(np.shape(covs)) == 2:
                        covs = [np.array(covs)]
                    elif len(np.shape(covs)) != 3:
                        raise TypeError("List of covariances the wrong shape")
                else:
                    raise TypeError("Must pass a list of covariances")
            if corrcoefs is not None:
                if isinstance(corrcoefs, np.ndarray):
                    corrcoefs = [corrcoefs]
                elif isinstance(corrcoefs, list):
                    if len(np.shape(corrcoefs)) == 2:
                        corrcoefs = [np.array(corrcoefs)]
                    elif len(np.shape(corrcoefs)) != 3:
                        raise TypeError(
                            "List of correlation coefficients the wrong shape"
                        )
                elif not isinstance(corrcoefs, list):
                    raise TypeError("Must pass a list of correlation coefficients")
            if weights is not None:
                if isinstance(weights, (int, float)):
                    weights = [weights]
                elif isinstance(weights, list):
                    if len(weights) != 1:
                        raise ValueError("Wrong number of weights given")

        for val in [mus, sigmas, covs, corrcoefs, weights]:
            if val is not None and not isinstance(val, list):
                raise TypeError("Value must be a list")
            else:
                if val is not None and len(val) != nmodes:
                    raise ValueError("Wrong number of modes given")

        # add the modes
        self.nmodes = 0
        for i in range(nmodes):
            mu = mus[i] if mus is not None else None
            sigma = sigmas[i] if sigmas is not None else None
            corrcoef = corrcoefs[i] if corrcoefs is not None else None
            cov = covs[i] if covs is not None else None
            weight = weights[i] if weights is not None else 1.0

            self.add_mode(mu, sigma, corrcoef, cov, weight)

    def add_mode(self, mus=None, sigmas=None, corrcoef=None, cov=None, weight=1.0):
        """
        Add a new mode.
        """

        # add means
        if mus is not None:
            try:
                self.mus.append(list(mus))  # means
            except TypeError:
                raise TypeError("'mus' must be a list")
        else:
            self.mus.append(np.zeros(self.num_vars))

        # add the covariances if supplied
        if cov is not None:
            self.covs.append(np.asarray(cov))

            if len(self.covs[-1].shape) != 2:
                raise ValueError("Covariance matrix must be a 2d array")

            if (
                self.covs[-1].shape[0] != self.covs[-1].shape[1]
                or self.covs[-1].shape[0] != self.num_vars
            ):
                raise ValueError("Covariance shape is inconsistent")

            # check matrix is symmetric
            if not np.allclose(self.covs[-1], self.covs[-1].T):
                raise ValueError("Covariance matrix is not symmetric")

            self.sigmas.append(np.sqrt(np.diag(self.covs[-1])))  # standard deviations

            # convert covariance into a correlation coefficient matrix
            D = self.sigmas[-1] * np.identity(self.covs[-1].shape[0])
            Dinv = np.linalg.inv(D)
            self.corrcoefs.append(np.dot(np.dot(Dinv, self.covs[-1]), Dinv))
        elif corrcoef is not None and sigmas is not None:
            self.corrcoefs.append(np.asarray(corrcoef))

            if len(self.corrcoefs[-1].shape) != 2:
                raise ValueError(
                    "Correlation coefficient matrix must be a 2d array."
                )

            if (
                self.corrcoefs[-1].shape[0] != self.corrcoefs[-1].shape[1]
                or self.corrcoefs[-1].shape[0] != self.num_vars
            ):
                raise ValueError(
                    "Correlation coefficient matrix shape is inconsistent"
                )

            # check matrix is symmetric
            if not np.allclose(self.corrcoefs[-1], self.corrcoefs[-1].T):
                raise ValueError("Correlation coefficient matrix is not symmetric")

            # check diagonal is all ones
            if not np.all(np.diag(self.corrcoefs[-1]) == 1.0):
                raise ValueError("Correlation coefficient matrix is not correct")

            try:
                self.sigmas.append(list(sigmas))  # standard deviations
            except TypeError:
                raise TypeError("'sigmas' must be a list")

            if len(self.sigmas[-1]) != self.num_vars:
                raise ValueError(
                    "Number of standard deviations must be the "
                    "same as the number of parameters."
                )

            # convert correlation coefficients to covariance matrix
            D = self.sigmas[-1] * np.identity(self.corrcoefs[-1].shape[0])
            self.covs.append(np.dot(D, np.dot(self.corrcoefs[-1], D)))
        else:
            # set unit variance uncorrelated covariance
            self.corrcoefs.append(np.eye(self.num_vars))
            self.covs.append(np.eye(self.num_vars))
            self.sigmas.append(np.ones(self.num_vars))

        # compute log of product of sigmas, needed for "standard" multivariate normal
        self.logprodsigmas.append(np.log(np.prod(self.sigmas[-1])))

        # get eigen values and vectors
        try:
            evals, evecs = np.linalg.eig(self.corrcoefs[-1])
            self.eigvalues.append(evals)
            self.eigvectors.append(evecs)
        except Exception as e:
            raise RuntimeError(
                "Problem getting eigenvalues and vectors: {}".format(e)
            )

        # check eigenvalues are positive
        if np.any(self.eigvalues[-1] <= 0.0):
            raise ValueError(
                "Correlation coefficient matrix is not positive definite"
            )
        self.sqeigvalues.append(np.sqrt(self.eigvalues[-1]))

        # set the weights
        if weight is None:
            self.weights.append(1.0)
        else:
            self.weights.append(weight)

        # set the cumulative relative weights
        self.cumweights = np.cumsum(self.weights) / np.sum(self.weights)

        # add the mode
        self.nmodes += 1

        # add "standard" multivariate normal distribution
        # - when the typical scales of the parameters are very different,
        #   multivariate_normal() may complain that the covariance matrix is singular
        # - instead pass zero means and correlation matrix instead of covariance matrix
        #   to get the equivalent of a standard normal distribution in higher dimensions
        # - this modifies the multivariate normal PDF as follows:
        #     multivariate_normal(mean=mus, cov=cov).logpdf(x)
        #     = multivariate_normal(mean=0, cov=corrcoefs).logpdf((x - mus)/sigmas) - logprodsigmas
        self.mvn.append(
            scipy.stats.multivariate_normal(mean=np.zeros(self.num_vars), cov=self.corrcoefs[-1])
        )

    def _rescale(self, samp, **kwargs):
        try:
            mode = kwargs["mode"]
        except KeyError:
            mode = None

        if mode is None:
            if self.nmodes == 1:
                mode = 0
            else:
                mode = np.argwhere(self.cumweights - random.rng.uniform(0, 1) > 0)[0][0]

        samp = erfinv(2.0 * samp - 1) * 2.0 ** 0.5

        # rotate and scale to the multivariate normal shape
        samp = self.mus[mode] + self.sigmas[mode] * np.einsum(
            "ij,kj->ik", samp * self.sqeigvalues[mode], self.eigvectors[mode]
        )
        return samp

    def _sample(self, size, **kwargs):
        try:
            mode = kwargs["mode"]
        except KeyError:
            mode = None

        if mode is None:
            if self.nmodes == 1:
                mode = 0
            else:
                if size == 1:
                    mode = np.argwhere(self.cumweights - random.rng.uniform(0, 1) > 0)[0][0]
                else:
                    # pick modes
                    mode = [
                        np.argwhere(self.cumweights - r > 0)[0][0]
                        for r in random.rng.uniform(0, 1, size)
                    ]

        samps = np.zeros((size, len(self)))
        for i in range(size):
            inbound = False
            while not inbound:
                # sample the multivariate Gaussian keys
                vals = random.rng.uniform(0, 1, len(self))

                if isinstance(mode, list):
                    samp = np.atleast_1d(self.rescale(vals, mode=mode[i]))
                else:
                    samp = np.atleast_1d(self.rescale(vals, mode=mode))
                samps[i, :] = samp

                # check sample is in bounds (otherwise perform another draw)
                outbound = False
                for name, val in zip(self.names, samp):
                    if val < self.bounds[name][0] or val > self.bounds[name][1]:
                        outbound = True
                        break

                if not outbound:
                    inbound = True

        return samps

    def _ln_prob(self, samp, lnprob, outbounds):
        for j in range(samp.shape[0]):
            # loop over the modes and sum the probabilities
            for i in range(self.nmodes):
                # self.mvn[i] is a "standard" multivariate normal distribution; see add_mode()
                z = (samp[j] - self.mus[i]) / self.sigmas[i]
                lnprob[j] = np.logaddexp(lnprob[j], self.mvn[i].logpdf(z) - self.logprodsigmas[i])

        # set out-of-bounds values to -inf
        lnprob[outbounds] = -np.inf
        return lnprob

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if sorted(self.__dict__.keys()) != sorted(other.__dict__.keys()):
            return False
        for key in self.__dict__:
            if key == "mvn":
                if len(self.__dict__[key]) != len(other.__dict__[key]):
                    return False
                for thismvn, othermvn in zip(self.__dict__[key], other.__dict__[key]):
                    if not isinstance(
                        thismvn, scipy.stats._multivariate.multivariate_normal_frozen
                    ) or not isinstance(
                        othermvn, scipy.stats._multivariate.multivariate_normal_frozen
                    ):
                        return False
            elif isinstance(self.__dict__[key], (np.ndarray, list)):
                thisarr = np.asarray(self.__dict__[key])
                otherarr = np.asarray(other.__dict__[key])
                if thisarr.dtype == float and otherarr.dtype == float:
                    fin1 = np.isfinite(np.asarray(self.__dict__[key]))
                    fin2 = np.isfinite(np.asarray(other.__dict__[key]))
                    if not np.array_equal(fin1, fin2):
                        return False
                    if not np.allclose(thisarr[fin1], otherarr[fin2], atol=1e-15):
                        return False
                else:
                    if not np.array_equal(thisarr, otherarr):
                        return False
            else:
                if not self.__dict__[key] == other.__dict__[key]:
                    return False
        return True


class MultivariateNormalDist(MultivariateGaussianDist):
    """A synonym for the :class:`~bilby.core.prior.MultivariateGaussianDist` distribution."""


class JointPrior(Prior):
    def __init__(self, dist, name=None, latex_label=None, unit=None):
        """This defines the single parameter Prior object for parameters that belong to a JointPriorDist

        Parameters
        ==========
        dist: ChildClass of BaseJointPriorDist
            The shared JointPriorDistribution that this parameter belongs to
        name: str
            Name of this parameter. Must be contained in dist.names
        latex_label: str
            See superclass
        unit: str
            See superclass
        """
        if not isinstance(dist, BaseJointPriorDist):
            raise TypeError(
                "Must supply a JointPriorDist object instance to be shared by all joint params"
            )

        if name not in dist.names:
            raise ValueError(
                "'{}' is not a parameter in the JointPriorDist".format(name)
            )

        self.dist = dist
        super(JointPrior, self).__init__(
            name=name,
            latex_label=latex_label,
            unit=unit,
            minimum=dist.bounds[name][0],
            maximum=dist.bounds[name][1],
        )

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum
        self.dist.bounds[self.name] = (minimum, self.dist.bounds[self.name][1])

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum
        self.dist.bounds[self.name] = (self.dist.bounds[self.name][0], maximum)

    def rescale(self, val, **kwargs):
        """
        Scale a unit hypercube sample to the prior.

        Parameters
        ==========
        val: array_like
            value drawn from unit hypercube to be rescaled onto the prior
        kwargs: dict
            all kwargs passed to the dist.rescale method
        Returns
        =======
        float:
            A sample from the prior parameter.
        """

        self.dist.rescale_parameters[self.name] = val

        if self.dist.filled_rescale():
            values = np.array(list(self.dist.rescale_parameters.values())).T
            samples = self.dist.rescale(values, **kwargs)
            self.dist.reset_rescale()
            return samples
        else:
            return []  # return empty list

    def sample(self, size=1, **kwargs):
        """
        Draw a sample from the prior.

        Parameters
        ==========
        size: int, float (defaults to 1)
            number of samples to draw
        kwargs: dict
            kwargs passed to the dist.sample method
        Returns
        =======
        float:
            A sample from the prior parameter.
        """

        if self.name in self.dist.sampled_parameters:
            logger.warning(
                "You have already drawn a sample from parameter "
                "'{}'. The same sample will be "
                "returned".format(self.name)
            )

        if len(self.dist.current_sample) == 0:
            # generate a sample
            self.dist.sample(size=size, **kwargs)

        sample = self.dist.current_sample[self.name]

        if self.name not in self.dist.sampled_parameters:
            self.dist.sampled_parameters.append(self.name)

        if len(self.dist.sampled_parameters) == len(self.dist):
            # reset samples
            self.dist.reset_sampled()
        self.least_recently_sampled = sample
        return sample

    def ln_prob(self, val):
        """
        Return the natural logarithm of the prior probability. Note that this
        will not be correctly normalised if there are bounds on the
        distribution.

        Parameters
        ==========
        val: array_like
            value to evaluate the prior log-prob at
        Returns
        =======
        float:
            the logp value for the prior at given sample
        """
        self.dist.requested_parameters[self.name] = val

        if self.dist.filled_request():
            # all required parameters have been set
            values = list(self.dist.requested_parameters.values())

            # check for the same number of values for each parameter
            for i in range(len(self.dist) - 1):
                if isinstance(values[i], (list, np.ndarray)) or isinstance(
                    values[i + 1], (list, np.ndarray)
                ):
                    if isinstance(values[i], (list, np.ndarray)) and isinstance(
                        values[i + 1], (list, np.ndarray)
                    ):
                        if len(values[i]) != len(values[i + 1]):
                            raise ValueError(
                                "Each parameter must have the same "
                                "number of requested values."
                            )
                    else:
                        raise ValueError(
                            "Each parameter must have the same "
                            "number of requested values."
                        )

            lnp = self.dist.ln_prob(np.asarray(values).T)

            # reset the requested parameters
            self.dist.reset_request()
            return lnp
        else:
            # if not all parameters have been requested yet, just return 0
            if isinstance(val, (float, int)):
                return 0.0
            else:
                try:
                    # check value has a length
                    len(val)
                except Exception as e:
                    raise TypeError("Invalid type for ln_prob: {}".format(e))

                if len(val) == 1:
                    return 0.0
                else:
                    return np.zeros_like(val)

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ==========
        val: array_like
            value to evaluate the prior prob at

        Returns
        =======
        float:
            the p value for the prior at given sample
        """

        return np.exp(self.ln_prob(val))


class MultivariateGaussian(JointPrior):
    def __init__(self, dist, name=None, latex_label=None, unit=None):
        if not isinstance(dist, MultivariateGaussianDist):
            raise JointPriorDistError(
                "dist object must be instance of MultivariateGaussianDist"
            )
        super(MultivariateGaussian, self).__init__(
            dist=dist, name=name, latex_label=latex_label, unit=unit
        )


class MultivariateNormal(MultivariateGaussian):
    """A synonym for the :class:`bilby.core.prior.MultivariateGaussian`
    prior distribution."""


class JointPriorDistError(PriorException):
    """Class for Error handling of JointPriorDists for JointPriors"""
