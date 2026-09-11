import numpy as np
from scipy.special import (
    betainc,
    betaincinv,
    betaln,
    erf,
    erfinv,
    gammaincinv,
    gammainc,
    gammaln,
    stdtr,
    stdtrit,
    xlogy,
    xlog1py,
)

from .base import Prior
from ..utils import logger
from ...compat.utils import array_module, xp_wrap


class DeltaFunction(Prior):

    _leaves = ["peak", "_minimum", "_maximum", "least_recently_sampled"]

    def __init__(self, peak, name=None, latex_label=None, unit=None):
        """Dirac delta function prior, this always returns peak.

        Parameters
        ==========
        peak: float
            Peak value of the delta function
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass

        """
        super(DeltaFunction, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                            minimum=peak, maximum=peak, check_range_nonzero=False)
        self.peak = peak
        self._is_fixed = True
        self.least_recently_sampled = peak

    def rescale(self, val, *, xp=None):
        """Rescale everything to the peak with the correct shape.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Rescaled probability, equivalent to peak
        """
        return self.peak * val ** 0

    def prob(self, val, *, xp=None):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
         Union[float, array_like]: np.inf if val = peak, 0 otherwise

        """
        at_peak = (val == self.peak)
        # coerce bool to float for some array backends
        return at_peak * 1.0

    def cdf(self, val, *, xp=None):
        return 1.0 * (val > self.peak)


class PowerLaw(Prior):

    _leaves = ["alpha"]

    def __init__(self, alpha, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Power law with bounds and alpha, spectral index

        Parameters
        ==========
        alpha: float
            Power law exponent parameter
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(PowerLaw, self).__init__(name=name, latex_label=latex_label,
                                       minimum=minimum, maximum=maximum, unit=unit,
                                       boundary=boundary)
        self.alpha = alpha

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        =======
        Union[float, array_like]: Rescaled probability
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return xp.where(
                xp.asarray(self.alpha != -1),
                (
                    self.minimum ** (1 + self.alpha)
                    + val * (self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha))
                ) ** (1. / xp.asarray(1 + self.alpha)),
                self.minimum * xp.exp(val * xp.log(xp.asarray(self.maximum) / xp.asarray(self.minimum)))
            )

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = xp.where(
                xp.asarray(self.alpha != -1),
                (1 + self.alpha) / xp.asarray(self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha)),
                1 / xp.log(xp.asarray(self.maximum) / xp.asarray(self.minimum)),
            )
        return xp.nan_to_num(val ** self.alpha * norm) * self.is_in_prior_range(val)

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Return the logarithmic prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float:

        """
        with np.errstate(divide='ignore', invalid='ignore'):
            normalising = xp.where(
                xp.asarray(self.alpha != -1),
                (1 + self.alpha)
                / xp.asarray(self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha)),
                1 / xp.log(xp.asarray(self.maximum) / xp.asarray(self.minimum)),
            )
            ln_in_range = xp.log(1. * self.is_in_prior_range(val))
            ln_p = self.alpha * xp.nan_to_num(xp.log(val)) + xp.log(normalising)

        return ln_p + ln_in_range

    @xp_wrap
    def cdf(self, val, *, xp=None):
        with np.errstate(divide='ignore', invalid='ignore'):
            _cdf = xp.where(
                xp.asarray(self.alpha != -1),
                (val ** (self.alpha + 1) - self.minimum ** (self.alpha + 1))
                / (self.maximum ** (self.alpha + 1) - self.minimum ** (self.alpha + 1)),
                xp.log(val / self.minimum)
                / xp.log(xp.asarray(self.maximum) / xp.asarray(self.minimum)),
            )
        _cdf = xp.clip(_cdf, 0, 1)
        return _cdf


class Uniform(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Uniform prior with bounds

        Parameters
        ==========
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Uniform, self).__init__(name=name, latex_label=latex_label,
                                      minimum=minimum, maximum=maximum, unit=unit,
                                      boundary=boundary)

    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        =======
        Union[float, array_like]: Rescaled probability
        """
        return self.minimum + val * (self.maximum - self.minimum)

    def prob(self, val, *, xp=None):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        return ((val >= self.minimum) & (val <= self.maximum)) / (self.maximum - self.minimum)

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Return the log prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: log probability of val
        """
        with np.errstate(divide="ignore"):
            return xp.log(self.prob(val))

    @xp_wrap
    def cdf(self, val, *, xp=None):
        _cdf = xp.asarray((val - self.minimum) / (self.maximum - self.minimum))
        return xp.clip(_cdf, xp.asarray(0), xp.asarray(1))


class LogUniform(PowerLaw):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Log-Uniform prior with bounds

        Parameters
        ==========
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(LogUniform, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                         minimum=minimum, maximum=maximum, alpha=-1, boundary=boundary)
        if self.minimum <= 0:
            logger.warning('You specified a uniform-in-log prior with minimum={}'.format(self.minimum))


class SymmetricLogUniform(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Symmetric Log-Uniform distributions with bounds

        This is identical to a Log-Uniform distribution, but mirrored about
        the zero-axis and subsequently normalized. As such, the distribution
        has support on the two regions [-maximum, -minimum] and [minimum,
        maximum].

        Parameters
        ==========
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(SymmetricLogUniform, self).__init__(name=name, latex_label=latex_label,
                                                  minimum=minimum, maximum=maximum, unit=unit,
                                                  boundary=boundary)

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        =======
        Union[float, array_like]: Rescaled probability

        Notes
        =====
        The behaviour at the discontinuity changed in :code:`Bilby` version 3.
        Prior to version 3, this returned :code:`self.minimum` for :code:`rescale(0.5)`,
        now this returns :code:`0`.
        """
        return (
            xp.sign(2 * val - 1)
            * self.minimum
            * xp.exp(xp.abs(2 * val - 1) * xp.log(xp.asarray(self.maximum / self.minimum)))
        )

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        val = xp.abs(val)
        return (xp.nan_to_num(0.5 / val / xp.log(xp.asarray(self.maximum / self.minimum))) *
                self.is_in_prior_range(val))

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Return the logarithmic prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float:

        """
        return xp.nan_to_num(- xp.log(2 * xp.abs(val)) - xp.log(xp.log(xp.asarray(self.maximum / self.minimum))))

    @xp_wrap
    def cdf(self, val, *, xp=None):
        asymmetric = (
            xp.log(xp.maximum(xp.abs(val) / self.minimum, xp.asarray(1)))
            / xp.log(xp.asarray(self.maximum / self.minimum))
        )
        return xp.clip(0.5 * (1 + xp.sign(val) * asymmetric), 0, 1)


class Cosine(Prior):

    def __init__(self, minimum=-np.pi / 2, maximum=np.pi / 2, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Cosine prior with bounds

        Parameters
        ==========
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Cosine, self).__init__(minimum=minimum, maximum=maximum, name=name,
                                     latex_label=latex_label, unit=unit, boundary=boundary)

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to a uniform in cosine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        norm = 1 / (xp.sin(xp.asarray(self.maximum)) - xp.sin(xp.asarray(self.minimum)))
        return xp.arcsin(val / norm + xp.sin(xp.asarray(self.minimum)))

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val. Defined over [-pi/2, pi/2].

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        return xp.cos(val) / 2 * self.is_in_prior_range(val)

    @xp_wrap
    def cdf(self, val, *, xp=None):
        _cdf = (
            (xp.sin(val) - xp.sin(xp.asarray(self.minimum))) /
            (xp.sin(xp.asarray(self.maximum)) - xp.sin(xp.asarray(self.minimum)))
        )
        _cdf *= val >= self.minimum
        _cdf *= val <= self.maximum
        _cdf += val > self.maximum
        return _cdf


class Sine(Prior):

    def __init__(self, minimum=0, maximum=np.pi, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Sine prior with bounds

        Parameters
        ==========
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Sine, self).__init__(minimum=minimum, maximum=maximum, name=name,
                                   latex_label=latex_label, unit=unit, boundary=boundary)

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to a uniform in sine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        norm = 1 / (xp.cos(xp.asarray(self.minimum)) - xp.cos(xp.asarray(self.maximum)))
        return xp.arccos(xp.cos(xp.asarray(self.minimum)) - val / norm)

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val. Defined over [0, pi].

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return xp.sin(val) / 2 * self.is_in_prior_range(val)

    @xp_wrap
    def cdf(self, val, *, xp=None):
        _cdf = (
            (xp.cos(val) - xp.cos(xp.asarray(self.minimum)))
            / (xp.cos(xp.asarray(self.maximum)) - xp.cos(xp.asarray(self.minimum)))
        )
        _cdf *= val >= self.minimum
        _cdf *= val <= self.maximum
        _cdf += val > self.maximum
        return _cdf


class Gaussian(Prior):

    _leaves = ["mu", "sigma"]

    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, boundary=None):
        """Gaussian prior with mean mu and width sigma

        Parameters
        ==========
        mu: float
            Mean of the Gaussian prior
        sigma:
            Width/Standard deviation of the Gaussian prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Gaussian, self).__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary)
        self.mu = mu
        self.sigma = sigma

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the appropriate Gaussian prior.

        Parameters
        ==========
        val: Union[float, int, array_like]

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        return self.mu + erfinv(2 * val - 1) * 2 ** 0.5 * self.sigma

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return xp.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / (2 * np.pi) ** 0.5 / self.sigma

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Return the Log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return -0.5 * ((self.mu - val) ** 2 / self.sigma ** 2 + xp.log(xp.asarray(2 * np.pi * self.sigma ** 2)))

    def cdf(self, val, *, xp=None):
        return (1 - erf((self.mu - val) / 2 ** 0.5 / self.sigma)) / 2


class Normal(Gaussian):
    """A synonym for the  Gaussian distribution. """


class TruncatedGaussian(Prior):

    _leaves = ["mu", "sigma"]

    def __init__(self, mu, sigma, minimum, maximum, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Truncated Gaussian prior with mean mu and width sigma

        https://en.wikipedia.org/wiki/Truncated_normal_distribution

        Parameters
        ==========
        mu: float
            Mean of the Gaussian prior
        sigma:
            Width/Standard deviation of the Gaussian prior
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(TruncatedGaussian, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                                minimum=minimum, maximum=maximum, boundary=boundary)
        self.mu = mu
        self.sigma = sigma

    @property
    def normalisation(self):
        """ Calculates the proper normalisation of the truncated Gaussian

        Returns
        =======
        float: Proper normalisation of the truncated Gaussian
        """
        return (erf((self.maximum - self.mu) / 2 ** 0.5 / self.sigma) - erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) / 2

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the appropriate truncated Gaussian prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        return erfinv(2 * val * self.normalisation + erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) * 2 ** 0.5 * self.sigma + self.mu

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        return xp.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / (2 * np.pi) ** 0.5 \
            / self.sigma / self.normalisation * self.is_in_prior_range(val)

    def cdf(self, val, *, xp=None):
        _cdf = (erf((val - self.mu) / 2 ** 0.5 / self.sigma) - erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) / 2 / self.normalisation
        _cdf *= val >= self.minimum
        _cdf *= val <= self.maximum
        _cdf += val > self.maximum
        return _cdf


class TruncatedNormal(TruncatedGaussian):
    """A synonym for the TruncatedGaussian distribution."""


class HalfGaussian(TruncatedGaussian):

    _leaves = ["sigma", "mu"]

    def __init__(self, sigma, name=None, latex_label=None, unit=None, boundary=None):
        """A Gaussian with its mode at zero, and truncated to only be positive.

        Parameters
        ==========
        sigma: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(HalfGaussian, self).__init__(mu=0., sigma=sigma, minimum=0., maximum=np.inf,
                                           name=name, latex_label=latex_label,
                                           unit=unit, boundary=boundary)


class HalfNormal(HalfGaussian):
    """A synonym for the HalfGaussian distribution."""


class LogNormal(Prior):

    _leaves = ["mu", "sigma"]

    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, boundary=None):
        """Log-normal prior with mean mu and width sigma

        https://en.wikipedia.org/wiki/Log-normal_distribution

        Parameters
        ==========
        mu: float
            Mean of the Gaussian prior
        sigma:
            Width/Standard deviation of the Gaussian prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(LogNormal, self).__init__(name=name, minimum=0., latex_label=latex_label,
                                        unit=unit, boundary=boundary)

        if sigma <= 0.:
            raise ValueError("For the LogGaussian prior the standard deviation must be positive")

        self.mu = mu
        self.sigma = sigma

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the appropriate LogNormal prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        return xp.exp(self.mu + (2 * self.sigma ** 2)**0.5 * erfinv(2 * val - 1))

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Returns the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return xp.exp(self.ln_prob(val, xp=xp))

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return xp.nan_to_num((
                -(xp.log(xp.maximum(val, xp.asarray(self.minimum))) - self.mu) ** 2 / self.sigma ** 2 / 2
                - xp.log((2 * np.pi)**0.5 * val * self.sigma)
            ) + xp.log(val > self.minimum), nan=-xp.inf, neginf=-xp.inf, posinf=-xp.inf)

    @xp_wrap
    def cdf(self, val, *, xp=None):
        with np.errstate(divide="ignore"):
            return 0.5 + erf(
                (xp.log(xp.maximum(val, xp.asarray(self.minimum))) - self.mu) / self.sigma / np.sqrt(2)
            ) / 2


class LogGaussian(LogNormal):
    """Synonym of LogNormal prior."""


class Exponential(Prior):

    _leaves = ["mu"]

    def __init__(self, mu, name=None, latex_label=None, unit=None, boundary=None):
        """Exponential prior with mean mu

        Parameters
        ==========
        mu: float
            Mean of the Exponential prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Exponential, self).__init__(name=name, minimum=0., latex_label=latex_label,
                                          unit=unit, boundary=boundary)
        self.mu = mu

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the appropriate Exponential prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        with np.errstate(divide="ignore", over="ignore"):
            return -self.mu * xp.log1p(-val)

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return xp.exp(self.ln_prob(val, xp=xp))

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        with np.errstate(divide="ignore"):
            return -val / self.mu - xp.log(xp.asarray(self.mu)) + xp.log(val >= self.minimum)

    @xp_wrap
    def cdf(self, val, *, xp=None):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return xp.maximum(1. - xp.exp(-val / self.mu), xp.asarray(0.0))


class StudentT(Prior):

    _leaves = ["df", "mu", "scale"]

    def __init__(self, df, mu=0., scale=1., name=None, latex_label=None,
                 unit=None, boundary=None):
        """Student's t-distribution prior with number of degrees of freedom df,
        mean mu and scale

        https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution

        Parameters
        ==========
        df: float
            Number of degrees of freedom for distribution
        mu: float
            Mean of the Student's t-prior
        scale:
            Width of the Student's t-prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(StudentT, self).__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary)

        if df <= 0. or scale <= 0.:
            raise ValueError("For the StudentT prior the number of degrees of freedom and scale must be positive")

        self.df = df
        self.mu = mu
        self.scale = scale

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the appropriate Student's t-prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Notes
        =====
        This explicitly casts to the requested backend, but the computation will be done by scipy.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                xp.nan_to_num(stdtrit(self.df, val) * self.scale + self.mu)
                + xp.log(val > 0)
                - xp.log(val < 1)
            )

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return xp.exp(self.ln_prob(val))

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return (
            gammaln(0.5 * (self.df + 1)) - gammaln(0.5 * self.df)
            - xp.log(xp.asarray((np.pi * self.df)**0.5 * self.scale)) - (self.df + 1) / 2
            * xp.log(1 + ((val - self.mu) / self.scale) ** 2 / self.df)
        )

    def cdf(self, val, *, xp=None):
        return stdtr(self.df, (val - self.mu) / self.scale)


class Beta(Prior):

    _leaves = ["alpha", "beta"]

    def __init__(self, alpha, beta, minimum=0, maximum=1, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Beta distribution

        https://en.wikipedia.org/wiki/Beta_distribution

        This wraps around
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

        Parameters
        ==========
        alpha: float
            first shape parameter
        beta: float
            second shape parameter
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Beta, self).__init__(minimum=minimum, maximum=maximum, name=name,
                                   latex_label=latex_label, unit=unit, boundary=boundary)

        if alpha <= 0. or beta <= 0.:
            raise ValueError("alpha and beta must both be positive values")

        self.alpha = alpha
        self.beta = beta

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the appropriate Beta prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Notes
        =====
        This explicitly casts to the requested backend, but the computation will be done by scipy.
        """
        return (
            xp.asarray(betaincinv(xp.asarray(self.alpha), xp.asarray(self.beta), val))
            * (self.maximum - self.minimum)
            + self.minimum
        )

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return xp.exp(self.ln_prob(val, xp=xp))

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        normalized = (val - self.minimum) / (self.maximum - self.minimum)
        ln_prob = (
            xlog1py(xp.asarray(self.beta - 1.0), -normalized)
            + xlogy(xp.asarray(self.alpha - 1.0), normalized)
            - xp.log(xp.asarray(self.maximum - self.minimum))
            - betaln(xp.asarray(self.alpha), xp.asarray(self.beta))
        )
        return xp.where(
            xp.abs(normalized - 0.5) <= 0.5,
            xp.nan_to_num(ln_prob, nan=-xp.inf, neginf=-xp.inf, posinf=-xp.inf),
            -xp.inf,
        )

    @xp_wrap
    def cdf(self, val, *, xp=None):
        return xp.nan_to_num(
            betainc(
                xp.asarray(self.alpha),
                xp.asarray(self.beta),
                (val - self.minimum) / (self.maximum - self.minimum)
            )
        ) + (val > self.maximum)


class Logistic(Prior):

    _leaves = ["mu", "scale"]

    def __init__(self, mu, scale, name=None, latex_label=None, unit=None, boundary=None):
        """Logistic distribution

        https://en.wikipedia.org/wiki/Logistic_distribution

        Parameters
        ==========
        mu: float
            Mean of the distribution
        scale: float
            Width of the distribution
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Logistic, self).__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary)

        if scale <= 0.:
            raise ValueError("For the Logistic prior the scale must be positive")

        self.mu = mu
        self.scale = scale

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the appropriate Logistic prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        with np.errstate(divide="ignore"):
            val = xp.asarray(val)
            return self.mu + self.scale * xp.log(xp.maximum(val / (1 - val), xp.asarray(0)))

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return xp.exp(self.ln_prob(val))

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        with np.errstate(over="ignore"):
            return -(val - self.mu) / self.scale -\
                2. * xp.log1p(xp.exp(-(val - self.mu) / self.scale)) - xp.log(xp.asarray(self.scale))

    @xp_wrap
    def cdf(self, val, *, xp=None):
        return 1. / (1. + xp.exp(-(val - self.mu) / self.scale))


class Cauchy(Prior):

    _leaves = ["alpha", "beta"]

    def __init__(self, alpha, beta, name=None, latex_label=None, unit=None, boundary=None):
        """Cauchy distribution

        https://en.wikipedia.org/wiki/Cauchy_distribution

        Parameters
        ==========
        alpha: float
            Location parameter
        beta: float
            Scale parameter
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Cauchy, self).__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary)

        if beta <= 0.:
            raise ValueError("For the Cauchy prior the scale must be positive")

        self.alpha = alpha
        self.beta = beta

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the appropriate Cauchy prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        rescaled = self.alpha + self.beta * xp.tan(np.pi * (val - 0.5))
        with np.errstate(divide="ignore", invalid="ignore"):
            return rescaled - xp.log(val < 1) + xp.log(val > 0)

    def prob(self, val, *, xp=None):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return 1. / self.beta / np.pi / (1. + ((val - self.alpha) / self.beta) ** 2)

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Return the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Log prior probability of val
        """
        return - xp.log(xp.asarray(self.beta * np.pi)) - xp.log(1. + ((val - self.alpha) / self.beta) ** 2)

    @xp_wrap
    def cdf(self, val, *, xp=None):
        return 0.5 + xp.arctan((val - self.alpha) / self.beta) / np.pi


class Lorentzian(Cauchy):
    """Synonym for the Cauchy distribution"""


class Gamma(Prior):

    _leaves = ["k", "theta"]

    def __init__(self, k, theta=1., name=None, latex_label=None, unit=None, boundary=None):
        """Gamma distribution

        https://en.wikipedia.org/wiki/Gamma_distribution

        Parameters
        ==========
        k: float
            The shape parameter
        theta: float
            The scale parameter
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Gamma, self).__init__(name=name, minimum=0., latex_label=latex_label,
                                    unit=unit, boundary=boundary)

        if k <= 0 or theta <= 0:
            raise ValueError("For the Gamma prior the shape and scale must be positive")

        self.k = k
        self.theta = theta

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the appropriate Gamma prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Notes
        =====
        Currently, the underlying function :code:`scipy.special.gammaincinv` does not
        support GPU dispatch and so may lead to performance degradation when using
        hardware acceleration.
        """
        return xp.asarray(gammaincinv(self.k, val)) * self.theta

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val.

        Parameters
        ==========
        val:  Union[float, int, array_like]

        Returns
        =======
         Union[float, array_like]: Prior probability of val
        """
        return xp.exp(self.ln_prob(val))

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        with np.errstate(divide="ignore"):
            ln_prob = (
                xlogy(xp.asarray(self.k - 1), val) - val / self.theta
                - xlogy(self.k, self.theta) - gammaln(self.k)
            ) + xp.log(val >= self.minimum)
        return xp.nan_to_num(ln_prob, nan=-xp.inf, neginf=-xp.inf, posinf=xp.inf)

    @xp_wrap
    def cdf(self, val, *, xp=None):
        return gammainc(xp.asarray(self.k), xp.maximum(val, xp.asarray(self.minimum)) / self.theta)


class ChiSquared(Gamma):

    _leaves = ["nu", "theta"]

    def __init__(self, nu, name=None, latex_label=None, unit=None, boundary=None):
        """Chi-squared distribution

        https://en.wikipedia.org/wiki/Chi-squared_distribution

        Parameters
        ==========
        nu: int
            Number of degrees of freedom
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """

        if nu <= 0 or not isinstance(nu, int):
            raise ValueError("For the ChiSquared prior the number of degrees of freedom must be a positive integer")

        super(ChiSquared, self).__init__(name=name, k=nu / 2., theta=2.,
                                         latex_label=latex_label, unit=unit, boundary=boundary)

    @property
    def nu(self):
        return int(self.k * 2)

    @nu.setter
    def nu(self, nu):
        self.k = nu / 2.


class FermiDirac(Prior):

    _leaves = ["sigma", "mu", "r", "expr"]

    def __init__(self, sigma, mu=None, r=None, name=None, latex_label=None,
                 unit=None):
        """A Fermi-Dirac type prior, with a fixed lower boundary at zero
        (see, e.g. Section 2.3.5 of [1]_). The probability distribution
        is defined by Equation 22 of [1]_.

        Parameters
        ==========
        sigma: float (required)
            The range over which the attenuation of the distribution happens
        mu: float
            The point at which the distribution falls to 50% of its maximum
            value
        r: float
            A value giving mu/sigma. This can be used instead of specifying
            mu.
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass

        References
        ==========

        .. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
           <https:arxiv.org/abs/1705.08978v1>`_, 2017.
        """
        super(FermiDirac, self).__init__(name=name, latex_label=latex_label, unit=unit, minimum=0.)

        self.sigma = sigma

        if mu is None and r is None:
            raise ValueError("For the Fermi-Dirac prior either a 'mu' value or 'r' "
                             "value must be given.")

        if r is None and mu is not None:
            self.mu = mu
            self.r = self.mu / self.sigma
        else:
            self.r = r
            self.mu = self.sigma * self.r

        if self.r <= 0. or self.sigma <= 0.:
            raise ValueError("For the Fermi-Dirac prior the values of sigma and r "
                             "must be positive.")

        xp = array_module((mu, sigma, r))
        self.expr = xp.exp(self.r)

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the appropriate Fermi-Dirac prior.

        Parameters
        ==========
        val: Union[float, int, array_like]

        This maps to the inverse CDF. This has been analytically solved for this case,
        see Equation 24 of [1]_.

        References
        ==========

        .. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
           <https:arxiv.org/abs/1705.08978v1>`_, 2017.
        """
        inv = -1 / self.expr + (1 + self.expr)**-val + (1 + self.expr)**-val / self.expr
        return -self.sigma * xp.log(xp.maximum(inv, xp.asarray(0)))

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        return (
            (xp.exp((val - self.mu) / self.sigma) + 1)**-1
            / (self.sigma * xp.log1p(xp.asarray(self.expr)))
            * (val >= self.minimum)
        )

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Return the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Log prior probability of val
        """
        return xp.log(self.prob(val))

    @xp_wrap
    def cdf(self, val, *, xp=None):
        """
        Evaluate the CDF of the Fermi-Dirac distribution using a slightly
        modified form of Equation 23 of [1]_.

        Parameters
        ==========
        val: Union[float, int, array_like]
            The value(s) to evaluate the CDF at

        Returns
        =======
        Union[float, array_like]:
            The CDF value(s)

        References
        ==========

        .. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
           <https:arxiv.org/abs/1705.08978v1>`_, 2017.
        """
        result = (
            (xp.logaddexp(xp.asarray(0.0), -xp.asarray(self.r)) - xp.logaddexp(-val / self.sigma, -xp.asarray(self.r)))
            / xp.logaddexp(xp.asarray(0.0), xp.asarray(self.r))
        )
        return xp.clip(result, 0, 1)


class WeightedDiscreteValues(Prior):
    def __init__(
        self,
        values,
        weights=None,
        name=None,
        latex_label=None,
        unit=None,
        boundary="periodic",
    ):
        """A weighted discrete-valued prior

        Parameters
        ==========
        values: 1d array_like of numeric type
            The discrete values of the prior.
        weights: 1d array_like of numeric type or None
            The weights of each category. If None, then all values are
            equally weighted. Default: None.
        name: str
            The name of the parameter
        latex_label: str
            The latex label of the parameter. Used for plotting.
        unit: str
            The unit of the parameter. Used for plotting.
        """

        xp = array_module(values)
        nvalues = len(values)
        values = xp.asarray(values)
        if values.shape != (nvalues,):
            raise ValueError(
                f"Shape of argument 'values' must be 1d array-like but has shape {values.shape}"
            )
        minimum = xp.min(values)
        # Small delta added to help with MCMC walking
        maximum = xp.max(values) * (1 + 1e-15)
        super(WeightedDiscreteValues, self).__init__(
            name=name, latex_label=latex_label, minimum=minimum,
            maximum=maximum, unit=unit, boundary=boundary)
        self.nvalues = nvalues
        sorter = xp.argsort(values)
        self._values_array = values[sorter]

        # inititialization of priors from repr only supports
        # python buildins
        self.values = self._values_array.tolist()

        weights = (
            xp.asarray(weights) / xp.sum(weights)
            if weights is not None
            else xp.ones(self.nvalues) / self.nvalues
        )
        # check for consistent shape of input
        if weights.shape != (self.nvalues,):
            raise ValueError(
                "Inconsistent shape of weights and number of values:"
                + f"weights has shape {weights.shape} "
                + f"while number of values is {self.values}"
            )
        self._weights_array = weights[sorter]
        self._lnweights_array = xp.log(self._weights_array)

        # save cdf for rescaling
        _cumulative_weights_array = xp.cumsum(self._weights_array)
        # insert 0 for values smaller than minimum
        self._cumulative_weights_array = xp.insert(_cumulative_weights_array, 0, 0)

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the discrete-value prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        =======
        Union[float, array_like]: Rescaled probability
        """
        index = xp.searchsorted(xp.asarray(self._cumulative_weights_array[1:]), val)
        return xp.asarray(self._values_array)[index]

    @xp_wrap
    def cdf(self, val, *, xp=None):
        """Return the cumulative prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: cumulative prior probability of val
        """
        index = xp.searchsorted(xp.asarray(self._values_array), val, side="right")
        return xp.asarray(self._cumulative_weights_array)[index]

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        index = xp.searchsorted(xp.asarray(self._values_array), val)
        index = xp.clip(index, 0, self.nvalues - 1)
        p = xp.where(
            xp.asarray(self._values_array[index]) == val,
            xp.asarray(self._weights_array[index]),
            xp.asarray(0.0),
        )
        # turn 0d numpy array to scalar
        return p[()]

    @xp_wrap
    def ln_prob(self, val, xp=None):
        """Return the logarithmic prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float:

        """
        index = xp.searchsorted(xp.asarray(self._values_array), val)
        index = xp.clip(index, 0, self.nvalues - 1)
        lnp = xp.where(
            xp.asarray(self._values_array[index]) == val,
            xp.asarray(self._lnweights_array[index]),
            -np.inf,
        )
        # turn 0d array to scalar
        return lnp[()]

    @property
    def weights(self):
        return self._weights_array.tolist()

    def pytree_flatten(self):
        children = (
            self._values_array,
            self._weights_array,
            self._lnweights_array,
            self._cumulative_weights_array,
            self.minimum,
            self.maximum,
        )
        aux_data = (
            self.name,
            self.latex_label,
            self.unit,
            self.boundary,
            self.nvalues,
        )
        return children, aux_data

    @classmethod
    def pytree_unflatten(cls, aux_data, children):
        prior = cls.__new__(cls)
        name, latex_label, unit, boundary, nvalues = aux_data
        values, weights, ln_weights, cumulative_weights, minimum, maximum = children
        prior.name = name
        prior.latex_label = latex_label
        prior.unit = unit
        prior.boundary = boundary
        prior.nvalues = nvalues
        prior._values_array = values
        prior._weights_array = weights
        prior._lnweights_array = ln_weights
        prior._cumulative_weights_array = cumulative_weights
        prior._minimum = minimum
        prior._maximum = maximum
        return prior


class DiscreteValues(WeightedDiscreteValues):
    def __init__(self, values, name=None, latex_label=None,
                 unit=None, boundary="periodic"):
        """An equal-weighted discrete-valued prior

        Parameters
        ==========
        values: array
            The discrete values of the prior.
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        """
        weights = np.ones_like(values)
        super(DiscreteValues, self).__init__(
            values=values,
            weights=weights,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
        )


class WeightedCategorical(WeightedDiscreteValues):
    def __init__(
        self,
        ncategories,
        weights=None,
        name=None,
        latex_label=None,
        unit=None,
        boundary="periodic",
    ):
        """A weighted Categorical prior

        Parameters
        ==========
        ncategories: int
            The number of available categories. The prior mass support is then
            integers [0, ncategories - 1].
        weights: 1d array_like or None
            The weights of each category. If None, then all categories are
            equally weighted. Default None.
        name: str
            The name of the parameter
        latex_label: str
            The latex label of the parameter. Used for plotting.
        unit: str
            The unit of the parameter. Used for plotting.
        """
        self.ncategories = ncategories
        values = np.arange(0, ncategories)
        super(WeightedCategorical, self).__init__(
            values=values,
            weights=weights,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
        )

    def pytree_flatten(self):
        children, aux_data = super().pytree_flatten()
        aux_data += (self.ncategories,)
        return children, aux_data

    @classmethod
    def pytree_unflatten(cls, aux_data, children):
        prior = super().pytree_unflatten(aux_data[:-1], children)
        prior.ncategories = aux_data[-1]
        return prior


class Categorical(DiscreteValues):
    def __init__(
        self, ncategories, name=None, latex_label=None, unit=None, boundary="periodic"
    ):
        """An equal-weighted Categorical prior
        Parameters
        ==========
        ncategories: int
            The number of available categories. The prior mass support is then
            integers [0, ncategories - 1].
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        """
        self.ncategories = ncategories
        values = np.arange(0, ncategories)
        super(Categorical, self).__init__(
            values=values,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
        )


class Triangular(Prior):
    """
    Define a new prior class which draws from a triangular distribution.

    For distribution details see: wikipedia.org/wiki/Triangular_distribution

    Here, minimum <= mode <= maximum,
    where the mode has the highest pdf value.

    """

    _leaves = [
        "mode",
        "fractional_mode",
        "scale",
        "rescaled_minimum",
        "rescaled_maximum",
    ]

    def __init__(self, mode, minimum, maximum, name=None, latex_label=None, unit=None):
        super(Triangular, self).__init__(
            name=name,
            latex_label=latex_label,
            unit=unit,
            minimum=minimum,
            maximum=maximum,
        )
        self.mode = mode
        self.fractional_mode = (self.mode - self.minimum) / (
            self.maximum - self.minimum
        )
        self.scale = self.maximum - self.minimum
        self.rescaled_minimum = self.minimum - (self.minimum == self.mode) * self.scale
        self.rescaled_maximum = self.maximum + (self.maximum == self.mode) * self.scale

    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from standard uniform to a triangular distribution.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        =======
        Union[float, array_like]: Rescaled probability

        """
        below_mode = (val * self.scale * (self.mode - self.minimum)) ** 0.5
        above_mode = ((1 - val) * self.scale * (self.maximum - self.mode)) ** 0.5
        return (self.minimum + below_mode) * (val < self.fractional_mode) + (
            self.maximum - above_mode
        ) * (val >= self.fractional_mode)

    def prob(self, val, *, xp=None):
        """
        Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val

        """
        between_minimum_and_mode = (
            (val < self.mode)
            * (self.minimum <= val)
            * (val - self.rescaled_minimum)
            / (self.mode - self.rescaled_minimum)
        )
        between_mode_and_maximum = (
            (self.mode <= val)
            * (val <= self.maximum)
            * (self.rescaled_maximum - val)
            / (self.rescaled_maximum - self.mode)
        )
        return 2.0 * (between_minimum_and_mode + between_mode_and_maximum) / self.scale

    def cdf(self, val, *, xp=None):
        """
        Return the prior cumulative probability at val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: prior cumulative probability at val

        """
        return (
            (val > self.mode)
            + (val > self.minimum)
            * (val <= self.maximum)
            / (self.scale)
            * (
                (val > self.mode)
                * (self.rescaled_maximum - val) ** 2.0
                / (self.mode - self.rescaled_maximum)
                + (val <= self.mode)
                * (val - self.rescaled_minimum) ** 2.0
                / (self.mode - self.rescaled_minimum)
            )
        )
