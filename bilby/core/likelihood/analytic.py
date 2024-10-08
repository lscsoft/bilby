import numpy as np
from scipy.special import gammaln, xlogy
from scipy.stats import multivariate_normal

from .base import Likelihood
from ..utils import infer_parameters_from_function


class Analytical1DLikelihood(Likelihood):
    """
    A general class for 1D analytical functions. The model
    parameters are inferred from the arguments of function

    Parameters
    ==========
    x, y: array_like
        The data to analyse
    func:
        The python function to fit to the data. Note, this must take the
        dependent variable as its first argument. The other arguments
        will require a prior and will be sampled over (unless a fixed
        value is given).
    """

    def __init__(self, x, y, func, **kwargs):
        parameters = infer_parameters_from_function(func)
        super(Analytical1DLikelihood, self).__init__(dict())
        self.x = x
        self.y = y
        self._func = func
        self._function_keys = [key for key in parameters if key not in kwargs]
        self.kwargs = kwargs

    def __repr__(self):
        return self.__class__.__name__ + '(x={}, y={}, func={})'.format(self.x, self.y, self.func.__name__)

    @property
    def func(self):
        """ Make func read-only """
        return self._func

    @property
    def model_parameters(self):
        """ This sets up the function only parameters (i.e. not sigma for the GaussianLikelihood) """
        return {key: self.parameters[key] for key in self.function_keys}

    @property
    def function_keys(self):
        """ Makes function_keys read_only """
        return self._function_keys

    @property
    def n(self):
        """ The number of data points """
        return len(self.x)

    @property
    def x(self):
        """ The independent variable. Setter assures that single numbers will be converted to arrays internally """
        return self._x

    @x.setter
    def x(self, x):
        if isinstance(x, int) or isinstance(x, float):
            x = np.array([x])
        self._x = x

    @property
    def y(self):
        """ The dependent variable. Setter assures that single numbers will be converted to arrays internally """
        return self._y

    @y.setter
    def y(self, y):
        if isinstance(y, int) or isinstance(y, float):
            y = np.array([y])
        self._y = y

    @property
    def residual(self):
        """ Residual of the function against the data. """
        return self.y - self.func(self.x, **self.model_parameters, **self.kwargs)


class GaussianLikelihood(Analytical1DLikelihood):
    def __init__(self, x, y, func, sigma=None, **kwargs):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of function

        Parameters
        ==========
        x, y: array_like
            The data to analyse
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        sigma: None, float, array_like
            If None, the standard deviation of the noise is unknown and will be
            estimated (note: this requires a prior to be given for sigma). If
            not None, this defines the standard-deviation of the data points.
            This can either be a single float, or an array with length equal
            to that for `x` and `y`.
        """

        super(GaussianLikelihood, self).__init__(x=x, y=y, func=func, **kwargs)
        self.sigma = sigma

        # Check if sigma was provided, if not it is a parameter
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        log_l = np.sum(- (self.residual / self.sigma)**2 / 2 -
                       np.log(2 * np.pi * self.sigma**2) / 2)
        return log_l

    def __repr__(self):
        return self.__class__.__name__ + '(x={}, y={}, func={}, sigma={})' \
            .format(self.x, self.y, self.func.__name__, self.sigma)

    @property
    def sigma(self):
        """
        This checks if sigma has been set in parameters. If so, that value
        will be used. Otherwise, the attribute sigma is used. The logic is
        that if sigma is not in parameters the attribute is used which was
        given at init (i.e. the known sigma as either a float or array).
        """
        return self.parameters.get('sigma', self._sigma)

    @sigma.setter
    def sigma(self, sigma):
        if sigma is None:
            self._sigma = sigma
        elif isinstance(sigma, float) or isinstance(sigma, int):
            self._sigma = sigma
        elif len(sigma) == self.n:
            self._sigma = sigma
        else:
            raise ValueError('Sigma must be either float or array-like x.')


class PoissonLikelihood(Analytical1DLikelihood):
    def __init__(self, x, y, func, **kwargs):
        """
        A general Poisson likelihood for a rate - the model parameters are
        inferred from the arguments of function, which provides a rate.

        Parameters
        ==========

        x: array_like
            A dependent variable at which the Poisson rates will be calculated
        y: array_like
            The data to analyse - this must be a set of non-negative integers,
            each being the number of events within some interval.
        func:
            The python function providing the rate of events per interval to
            fit to the data. The function must be defined with the first
            argument being a dependent parameter (although this does not have
            to be used by the function if not required). The subsequent
            arguments will require priors and will be sampled over (unless a
            fixed value is given).
        """

        super(PoissonLikelihood, self).__init__(x=x, y=y, func=func, **kwargs)

    def log_likelihood(self):
        rate = self.func(self.x, **self.model_parameters, **self.kwargs)
        if not isinstance(rate, np.ndarray):
            raise ValueError(
                "Poisson rate function returns wrong value type! "
                "Is {} when it should be numpy.ndarray".format(type(rate)))
        elif np.any(rate < 0.):
            raise ValueError(("Poisson rate function returns a negative",
                              " value!"))
        elif np.any(rate == 0.):
            return -np.inf
        else:
            return np.sum(-rate + self.y * np.log(rate) - gammaln(self.y + 1))

    def __repr__(self):
        return Analytical1DLikelihood.__repr__(self)

    @property
    def y(self):
        """ Property assures that y-value is a positive integer. """
        return self.__y

    @y.setter
    def y(self, y):
        if not isinstance(y, np.ndarray):
            y = np.array([y])
        # check array is a non-negative integer array
        if y.dtype.kind not in 'ui' or np.any(y < 0):
            raise ValueError("Data must be non-negative integers")
        self.__y = y


class ExponentialLikelihood(Analytical1DLikelihood):
    def __init__(self, x, y, func, **kwargs):
        """
        An exponential likelihood function.

        Parameters
        ==========

        x, y: array_like
            The data to analyse
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given). The model should return the expected mean of
            the exponential distribution for each data point.
        """
        super(ExponentialLikelihood, self).__init__(x=x, y=y, func=func, **kwargs)

    def log_likelihood(self):
        mu = self.func(self.x, **self.model_parameters, **self.kwargs)
        if np.any(mu < 0.):
            return -np.inf
        return -np.sum(np.log(mu) + (self.y / mu))

    def __repr__(self):
        return Analytical1DLikelihood.__repr__(self)

    @property
    def y(self):
        """ Property assures that y-value is positive. """
        return self._y

    @y.setter
    def y(self, y):
        if not isinstance(y, np.ndarray):
            y = np.array([y])
        if np.any(y < 0):
            raise ValueError("Data must be non-negative")
        self._y = y


class StudentTLikelihood(Analytical1DLikelihood):
    def __init__(self, x, y, func, nu=None, sigma=1, **kwargs):
        """
        A general Student's t-likelihood for known or unknown number of degrees
        of freedom, and known or unknown scale (which tends toward the standard
        deviation for large numbers of degrees of freedom) - the model
        parameters are inferred from the arguments of function

        https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution

        Parameters
        ==========
        x, y: array_like
            The data to analyse
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        nu: None, float
            If None, the number of degrees of freedom of the noise is unknown
            and will be estimated (note: this requires a prior to be given for
            nu). If not None, this defines the number of degrees of freedom of
            the data points. As an example a `nu` of `len(x)-2` is equivalent
            to having marginalised a Gaussian distribution over an unknown
            standard deviation parameter using a uniform prior.
        sigma: 1.0, float
            Set the scale of the distribution. If not given then this defaults
            to 1, which specifies a standard (central) Student's t-distribution
        """
        super(StudentTLikelihood, self).__init__(x=x, y=y, func=func, **kwargs)

        self.nu = nu
        self.sigma = sigma

        # Check if nu was provided, if not it is a parameter
        if self.nu is None:
            self.parameters['nu'] = None

    def log_likelihood(self):
        if self.nu <= 0.:
            raise ValueError("Number of degrees of freedom for Student's "
                             "t-likelihood must be positive")

        nu = self.nu
        log_l =\
            np.sum(- (nu + 1) * np.log1p(self.lam * self.residual**2 / nu) / 2 +
                   np.log(self.lam / (nu * np.pi)) / 2 +
                   gammaln((nu + 1) / 2) - gammaln(nu / 2))
        return log_l

    def __repr__(self):
        base_string = '(x={}, y={}, func={}, nu={}, sigma={})'
        return self.__class__.__name__ + base_string.format(
            self.x, self.y, self.func.__name__, self.nu, self.sigma)

    @property
    def lam(self):
        """ Converts 'scale' to 'precision' """
        return 1. / self.sigma ** 2

    @property
    def nu(self):
        """ This checks if nu or sigma have been set in parameters. If so, those
        values will be used. Otherwise, the attribute nu is used. The logic is
        that if nu is not in parameters the attribute is used which was
        given at init (i.e. the known nu as a float)."""
        return self.parameters.get('nu', self._nu)

    @nu.setter
    def nu(self, nu):
        self._nu = nu


class Multinomial(Likelihood):
    """
    Likelihood for system with N discrete possibilities.
    """

    def __init__(self, data, n_dimensions, base="parameter_"):
        """

        Parameters
        ==========
        data: array-like
            The number of objects in each class
        n_dimensions: int
            The number of classes
        base: str
            The base of the parameter labels
        """
        self.data = np.array(data)
        self._total = np.sum(self.data)
        super(Multinomial, self).__init__(dict())
        self.n = n_dimensions
        self.base = base
        self._nll = None

    def log_likelihood(self):
        """
        Since n - 1 parameters are sampled, the last parameter is 1 - the rest
        """
        probs = [self.parameters[self.base + str(ii)]
                 for ii in range(self.n - 1)]
        probs.append(1 - sum(probs))
        return self._multinomial_ln_pdf(probs=probs)

    def noise_log_likelihood(self):
        """
        Our null hypothesis is that all bins have probability 1 / nbins, i.e.,
        no bin is preferred over any other.
        """
        if self._nll is None:
            self._nll = self._multinomial_ln_pdf(probs=1 / self.n)
        return self._nll

    def _multinomial_ln_pdf(self, probs):
        """Lifted from scipy.stats.multinomial._logpdf"""
        ln_prob = gammaln(self._total + 1) + np.sum(
            xlogy(self.data, probs) - gammaln(self.data + 1), axis=-1)
        return ln_prob


class AnalyticalMultidimensionalCovariantGaussian(Likelihood):
    """
        A multivariate Gaussian likelihood
        with known analytic solution.

        Parameters
        ==========
        mean: array_like
            Array with the mean values of distribution
        cov: array_like
            The ndim*ndim covariance matrix
        """

    def __init__(self, mean, cov):
        self.cov = np.atleast_2d(cov)
        self.mean = np.atleast_1d(mean)
        self.sigma = np.sqrt(np.diag(self.cov))
        self.pdf = multivariate_normal(mean=self.mean, cov=self.cov)
        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}
        super(AnalyticalMultidimensionalCovariantGaussian, self).__init__(parameters=parameters)

    @property
    def dim(self):
        return len(self.cov[0])

    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        return self.pdf.logpdf(x)


class AnalyticalMultidimensionalBimodalCovariantGaussian(Likelihood):
    """
        A multivariate Gaussian likelihood
        with known analytic solution.

        Parameters
        ==========
        mean_1: array_like
            Array with the mean value of the first mode
        mean_2: array_like
            Array with the mean value of the second mode
        cov: array_like
        """

    def __init__(self, mean_1, mean_2, cov):
        self.cov = np.atleast_2d(cov)
        self.sigma = np.sqrt(np.diag(self.cov))
        self.mean_1 = np.atleast_1d(mean_1)
        self.mean_2 = np.atleast_1d(mean_2)
        self.pdf_1 = multivariate_normal(mean=self.mean_1, cov=self.cov)
        self.pdf_2 = multivariate_normal(mean=self.mean_2, cov=self.cov)
        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}
        super(AnalyticalMultidimensionalBimodalCovariantGaussian, self).__init__(parameters=parameters)

    @property
    def dim(self):
        return len(self.cov[0])

    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        return -np.log(2) + np.logaddexp(self.pdf_1.logpdf(x), self.pdf_2.logpdf(x))



