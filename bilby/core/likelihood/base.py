import copy

import numpy as np


class Likelihood(object):

    def __init__(self, parameters=None):
        """Empty likelihood class to be subclassed by other likelihoods

        Parameters
        ==========
        parameters: dict
            A dictionary of the parameter names and associated values
        """
        self.parameters = parameters
        self._meta_data = None
        self._marginalized_parameters = []

    def __repr__(self):
        return self.__class__.__name__ + '(parameters={})'.format(self.parameters)

    def log_likelihood(self):
        """

        Returns
        =======
        float
        """
        return np.nan

    def noise_log_likelihood(self):
        """

        Returns
        =======
        float
        """
        return np.nan

    def log_likelihood_ratio(self):
        """Difference between log likelihood and noise log likelihood

        Returns
        =======
        float
        """
        return self.log_likelihood() - self.noise_log_likelihood()

    @property
    def meta_data(self):
        return getattr(self, '_meta_data', None)

    @meta_data.setter
    def meta_data(self, meta_data):
        if isinstance(meta_data, dict):
            self._meta_data = meta_data
        else:
            raise ValueError("The meta_data must be an instance of dict")

    @property
    def marginalized_parameters(self):
        return self._marginalized_parameters


class ZeroLikelihood(Likelihood):
    """ A special test-only class which already returns zero likelihood

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood
        A likelihood object to mimic

    """

    def __init__(self, likelihood):
        super(ZeroLikelihood, self).__init__(dict.fromkeys(likelihood.parameters))
        self.parameters = likelihood.parameters
        self._parent = likelihood

    def log_likelihood(self):
        return 0

    def noise_log_likelihood(self):
        return 0

    def __getattr__(self, name):
        return getattr(self._parent, name)


class JointLikelihood(Likelihood):
    def __init__(self, *likelihoods):
        """
        A likelihood for combining pre-defined likelihoods.
        The parameters dict is automagically combined through parameters dicts
        of the given likelihoods. If parameters have different values have
        initially different values across different likelihoods, the value
        of the last given likelihood is chosen. This does not matter when
        using the JointLikelihood for sampling, because the parameters will be
        set consistently

        Parameters
        ==========
        *likelihoods: bilby.core.likelihood.Likelihood
            likelihoods to be combined parsed as arguments
        """
        self.likelihoods = likelihoods
        super(JointLikelihood, self).__init__(parameters={})
        self.__sync_parameters()

    def __sync_parameters(self):
        """ Synchronizes parameters between the likelihoods
        so that all likelihoods share a single parameter dict."""
        for likelihood in self.likelihoods:
            self.parameters.update(likelihood.parameters)
        for likelihood in self.likelihoods:
            likelihood.parameters = self.parameters

    @property
    def likelihoods(self):
        """ The list of likelihoods """
        return self._likelihoods

    @likelihoods.setter
    def likelihoods(self, likelihoods):
        likelihoods = copy.deepcopy(likelihoods)
        if isinstance(likelihoods, tuple) or isinstance(likelihoods, list):
            if all(isinstance(likelihood, Likelihood) for likelihood in likelihoods):
                self._likelihoods = list(likelihoods)
            else:
                raise ValueError('Try setting the JointLikelihood like this\n'
                                 'JointLikelihood(first_likelihood, second_likelihood, ...)')
        elif isinstance(likelihoods, Likelihood):
            self._likelihoods = [likelihoods]
        else:
            raise ValueError('Input likelihood is not a list of tuple. You need to set multiple likelihoods.')

    def log_likelihood(self):
        """ This is just the sum of the log likelihoods of all parts of the joint likelihood"""
        return sum([likelihood.log_likelihood() for likelihood in self.likelihoods])

    def noise_log_likelihood(self):
        """ This is just the sum of the noise likelihoods of all parts of the joint likelihood"""
        return sum([likelihood.noise_log_likelihood() for likelihood in self.likelihoods])


class MarginalizedLikelihoodReconstructionError(Exception):
    pass
