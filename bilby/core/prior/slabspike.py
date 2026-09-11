import numpy as np

from .base import Prior
from ..utils import logger
from ...compat.utils import xp_wrap


class SlabSpikePrior(Prior):

    _leaves = ["slab", "_spike_location", "_spike_height", "inverse_cdf_below_spike", "_minimum", "_maximum"]

    def __init__(self, slab, spike_location=None, spike_height=0):
        """'Slab-and-spike' prior, see e.g. https://arxiv.org/abs/1812.07259
        This prior is composed of a `slab`, i.e. any common prior distribution,
        and a Dirac spike at a fixed location. This can effectively be used
        to emulate sampling in the number of dimensions (similar to reversible-
        jump MCMC).

        `SymmetricLogUniform` and `FermiDirac` are currently not supported.

        Parameters
        ==========
        slab: Prior
            Any instance of a bilby prior class. All general prior attributes
            from the slab are copied into the SlabSpikePrior.
            Note that this hasn't been tested for conditional priors.
        spike_location: float, optional
            Location of the Dirac spike. Must be between minimum and maximum
            of the slab. Defaults to the minimum of the slab
        spike_height: float, optional
            Relative weight of the spike compared to the slab. Must be
            between 0 and 1. Defaults to 0, i.e. the prior is just the slab.

        """
        self.slab = slab
        super().__init__(name=self.slab.name, latex_label=self.slab.latex_label, unit=self.slab.unit,
                         minimum=self.slab.minimum, maximum=self.slab.maximum,
                         check_range_nonzero=self.slab.check_range_nonzero, boundary=self.slab.boundary)
        self.spike_location = spike_location
        self.spike_height = spike_height
        try:
            self.inverse_cdf_below_spike = self._find_inverse_cdf_fraction_before_spike()
        except Exception as e:
            logger.warning("Disregard the following warning when running tests:\n {}".format(e))

    @property
    def spike_location(self):
        return self._spike_location

    @spike_location.setter
    def spike_location(self, spike_loc):
        if spike_loc is None:
            spike_loc = self.minimum
        if not self.minimum <= spike_loc <= self.maximum:
            raise ValueError("Spike location {} not within prior domain ".format(spike_loc))
        self._spike_location = spike_loc

    @property
    def spike_height(self):
        return self._spike_height

    @spike_height.setter
    def spike_height(self, spike_height):
        if 0 <= spike_height <= 1:
            self._spike_height = spike_height
        else:
            raise ValueError("Spike height must be between 0 and 1, but is {}".format(spike_height))

    @property
    def slab_fraction(self):
        """ Relative prior weight of the slab. """
        return 1 - self.spike_height

    def _find_inverse_cdf_fraction_before_spike(self):
        return float(self.slab.cdf(self.spike_location)) * self.slab_fraction

    @xp_wrap
    def rescale(self, val, *, xp=None):
        """
        'Rescale' a sample from the unit line element to the prior.

        Parameters
        ==========
        val: Union[float, int, array_like]
            A random number between 0 and 1

        Returns
        =======
        array_like: Associated prior value with input value.
        """
        lower_indices = val < self.inverse_cdf_below_spike
        higher_indices = val >= (self.inverse_cdf_below_spike + self.spike_height)

        slab_scaled = self._contracted_rescale(
            val - self.spike_height * higher_indices, xp=xp
        )

        res = xp.where(
            lower_indices | higher_indices,
            slab_scaled,
            xp.asarray(self.spike_location),
        )
        return res

    @xp_wrap
    def _contracted_rescale(self, val, *, xp=None):
        """
        Contracted version of the rescale function that implements the `rescale` function
        on the pure slab part of the prior.

        Parameters
        ==========
        val: Union[float, int, array_like]
            A random number between 0 and self.slab_fraction

        Returns
        =======
        array_like: Associated prior value with input value.
        """
        return self.slab.rescale(val / self.slab_fraction, xp=xp)

    @xp_wrap
    def prob(self, val, *, xp=None):
        """Return the prior probability of val.
        Returns np.inf for the spike location

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        array_like: Prior probability of val
        """
        res = self.slab.prob(val) * self.slab_fraction
        with np.errstate(invalid="ignore"):
            res += xp.nan_to_num(xp.inf * (val == self.spike_location), posinf=xp.inf)
        return res

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        """Return the Log prior probability of val.
        Returns np.inf for the spike location

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        array_like: Prior probability of val
        """
        res = self.slab.ln_prob(val) + np.log(self.slab_fraction)
        with np.errstate(divide="ignore"):
            res += xp.nan_to_num(xp.inf * (val == self.spike_location), posinf=xp.inf)
        return res

    @xp_wrap
    def cdf(self, val, *, xp=None):
        """ Return the CDF of the prior.
        This calls to the slab CDF and adds a discrete step
        at the spike location.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        array_like: CDF value of val

        """
        res = self.slab.cdf(val, xp=xp) * self.slab_fraction
        res += (val > self.spike_location) * self.spike_height
        return res
