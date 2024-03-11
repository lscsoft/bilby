import numpy as np
from scipy.interpolate import interp1d

from .base import Prior
from ..utils import logger


class Interped(Prior):

    def __init__(self, xx, yy, minimum=np.nan, maximum=np.nan, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Creates an interpolated prior function from arrays of xx and yy=p(xx)

        Parameters
        ==========
        xx: array_like
            x values for the to be interpolated prior function
        yy: array_like
            p(xx) values for the to be interpolated prior function
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

        Attributes
        ==========
        probability_density: scipy.interpolate.interp1d
            Interpolated prior probability distribution
        cumulative_distribution: scipy.interpolate.interp1d
            Interpolated cumulative prior probability distribution
        inverse_cumulative_distribution: scipy.interpolate.interp1d
            Inverted cumulative prior probability distribution
        YY: array_like
            Cumulative prior probability distribution

        """
        self.xx = xx
        self.min_limit = min(xx)
        self.max_limit = max(xx)
        self._yy = yy
        self.YY = None
        self.probability_density = None
        self.cumulative_distribution = None
        self.inverse_cumulative_distribution = None
        self.__all_interpolated = interp1d(x=xx, y=yy, bounds_error=False, fill_value=0)
        minimum = float(np.nanmax(np.array((min(xx), minimum))))
        maximum = float(np.nanmin(np.array((max(xx), maximum))))
        super(Interped, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                       minimum=minimum, maximum=maximum, boundary=boundary)
        self._update_instance()

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if np.array_equal(self.xx, other.xx) and np.array_equal(self.yy, other.yy):
            return True
        return False

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ==========
        val:  Union[float, int, array_like]

        Returns
        =======
         Union[float, array_like]: Prior probability of val
        """
        return self.probability_density(val)

    def cdf(self, val):
        return self.cumulative_distribution(val)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This maps to the inverse CDF. This is done using interpolation.
        """
        rescaled = self.inverse_cumulative_distribution(val)
        if rescaled.shape == ():
            rescaled = float(rescaled)
        return rescaled

    @property
    def minimum(self):
        """Return minimum of the prior distribution.

        Updates the prior distribution if minimum is set to a different value.

        Yields an error if value is set below instantiated x-array minimum.

        Returns
        =======
        float: Minimum of the prior distribution

        """
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        if minimum < self.min_limit:
            raise ValueError('Minimum cannot be set below {}.'.format(round(self.min_limit, 2)))
        self._minimum = minimum
        if '_maximum' in self.__dict__ and self._maximum < np.inf:
            self._update_instance()

    @property
    def maximum(self):
        """Return maximum of the prior distribution.

        Updates the prior distribution if maximum is set to a different value.

        Yields an error if value is set above instantiated x-array maximum.

        Returns
        =======
        float: Maximum of the prior distribution

        """
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        if maximum > self.max_limit:
            raise ValueError('Maximum cannot be set above {}.'.format(round(self.max_limit, 2)))
        self._maximum = maximum
        if '_minimum' in self.__dict__ and self._minimum < np.inf:
            self._update_instance()

    @property
    def yy(self):
        """Return p(xx) values of the interpolated prior function.

        Updates the prior distribution if it is changed

        Returns
        =======
        array_like: p(xx) values

        """
        return self._yy

    @yy.setter
    def yy(self, yy):
        self._yy = yy
        self.__all_interpolated = interp1d(x=self.xx, y=self._yy, bounds_error=False, fill_value=0)
        self._update_instance()

    def _update_instance(self):
        self.xx = np.linspace(self.minimum, self.maximum, len(self.xx))
        self._yy = self.__all_interpolated(self.xx)
        self._initialize_attributes()

    def _initialize_attributes(self):
        from scipy.integrate import cumulative_trapezoid
        if np.trapz(self._yy, self.xx) != 1:
            logger.debug('Supplied PDF for {} is not normalised, normalising.'.format(self.name))
        self._yy /= np.trapz(self._yy, self.xx)
        self.YY = cumulative_trapezoid(self._yy, self.xx, initial=0)
        # Need last element of cumulative distribution to be exactly one.
        self.YY[-1] = 1
        self.probability_density = interp1d(x=self.xx, y=self._yy, bounds_error=False, fill_value=0)
        self.cumulative_distribution = interp1d(x=self.xx, y=self.YY, bounds_error=False, fill_value=(0, 1))
        self.inverse_cumulative_distribution = interp1d(x=self.YY, y=self.xx, bounds_error=True)


class FromFile(Interped):

    def __init__(self, file_name, minimum=None, maximum=None, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Creates an interpolated prior function from arrays of xx and yy=p(xx) extracted from a file

        Parameters
        ==========
        file_name: str
            Name of the file containing the xx and yy arrays
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
        try:
            self.file_name = file_name
            xx, yy = np.genfromtxt(self.file_name).T
            super(FromFile, self).__init__(xx=xx, yy=yy, minimum=minimum,
                                           maximum=maximum, name=name, latex_label=latex_label,
                                           unit=unit, boundary=boundary)
        except IOError:
            logger.warning("Can't load {}.".format(self.file_name))
            logger.warning("Format should be:")
            logger.warning(r"x\tp(x)")
