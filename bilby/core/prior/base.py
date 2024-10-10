from importlib import import_module
import json
import os
import re

import numpy as np
import scipy.stats
from scipy.interpolate import interp1d

from ..utils import (
    infer_args_from_method,
    BilbyJsonEncoder,
    decode_bilby_json,
    logger,
    get_dict_with_properties,
)


class Prior(object):
    _default_latex_labels = {}

    def __init__(self, name=None, latex_label=None, unit=None, minimum=-np.inf,
                 maximum=np.inf, check_range_nonzero=True, boundary=None):
        """ Implements a Prior object

        Parameters
        ==========
        name: str, optional
            Name associated with prior.
        latex_label: str, optional
            Latex label associated with prior, used for plotting.
        unit: str, optional
            If given, a Latex string describing the units of the parameter.
        minimum: float, optional
            Minimum of the domain, default=-np.inf
        maximum: float, optional
            Maximum of the domain, default=np.inf
        check_range_nonzero: boolean, optional
            If True, checks that the prior range is non-zero
        boundary: str, optional
            The boundary condition of the prior, can be 'periodic', 'reflective'
            Currently implemented in cpnest, dynesty and pymultinest.
        """
        if check_range_nonzero and maximum <= minimum:
            raise ValueError(
                "maximum {} <= minimum {} for {} prior on {}".format(
                    maximum, minimum, type(self).__name__, name
                )
            )
        self.name = name
        self.latex_label = latex_label
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        self.check_range_nonzero = check_range_nonzero
        self.least_recently_sampled = None
        self.boundary = boundary
        self._is_fixed = False

    def __call__(self):
        """Overrides the __call__ special method. Calls the sample method.

        Returns
        =======
        float: The return value of the sample method.
        """
        return self.sample()

    def __eq__(self, other):
        """
        Test equality of two prior objects.

        Returns true iff:

        - The class of the two priors are the same
        - Both priors have the same keys in the __dict__ attribute
        - The instantiation arguments match

        We don't check that all entries the the __dict__ attribute
        are equal as some attributes are variable for conditional
        priors.

        Parameters
        ==========
        other: Prior
            The prior to compare with

        Returns
        =======
        bool
            Whether the priors are equivalent

        Notes
        =====
        A special case is made for :code `scipy.stats.beta`: instances.
        It may be possible to remove this as we now only check instantiation
        arguments.

        """
        if self.__class__ != other.__class__:
            return False
        if sorted(self.__dict__.keys()) != sorted(other.__dict__.keys()):
            return False
        this_dict = self.get_instantiation_dict()
        other_dict = other.get_instantiation_dict()
        for key in this_dict:
            if key == "least_recently_sampled":
                continue
            if isinstance(this_dict[key], np.ndarray):
                if not np.array_equal(this_dict[key], other_dict[key]):
                    return False
            elif isinstance(this_dict[key], type(scipy.stats.beta(1., 1.))):
                continue
            else:
                if not this_dict[key] == other_dict[key]:
                    return False
        return True

    def sample(self, size=None):
        """Draw a sample from the prior

        Parameters
        ==========
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        =======
        float: A random number between 0 and 1, rescaled to match the distribution of this Prior

        """
        from ..utils.random import rng

        self.least_recently_sampled = self.rescale(rng.uniform(0, 1, size))
        return self.least_recently_sampled

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This should be overwritten by each subclass.

        Parameters
        ==========
        val: Union[float, int, array_like]
            A random number between 0 and 1

        Returns
        =======
        None

        """
        return None

    def prob(self, val):
        """Return the prior probability of val, this should be overwritten

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        np.nan

        """
        return np.nan

    def cdf(self, val):
        """ Generic method to calculate CDF, can be overwritten in subclass """
        from scipy.integrate import cumulative_trapezoid
        if np.any(np.isinf([self.minimum, self.maximum])):
            raise ValueError(
                "Unable to use the generic CDF calculation for priors with"
                "infinite support")
        x = np.linspace(self.minimum, self.maximum, 1000)
        pdf = self.prob(x)
        cdf = cumulative_trapezoid(pdf, x, initial=0)
        interp = interp1d(x, cdf, assume_sorted=True, bounds_error=False,
                          fill_value=(0, 1))
        return interp(val)

    def ln_prob(self, val):
        """Return the prior ln probability of val, this should be overwritten

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        np.nan

        """
        with np.errstate(divide='ignore'):
            return np.log(self.prob(val))

    def is_in_prior_range(self, val):
        """Returns True if val is in the prior boundaries, zero otherwise

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        np.nan

        """
        return (val >= self.minimum) & (val <= self.maximum)

    def __repr__(self):
        """Overrides the special method __repr__.

        Returns a representation of this instance that resembles how it is instantiated.
        Works correctly for all child classes

        Returns
        =======
        str: A string representation of this instance

        """
        prior_name = self.__class__.__name__
        prior_module = self.__class__.__module__
        instantiation_dict = self.get_instantiation_dict()
        args = ', '.join([f'{key}={repr(instantiation_dict[key])}' for key in instantiation_dict])
        if "bilby.core.prior" in prior_module:
            return f"{prior_name}({args})"
        else:
            return f"{prior_module}.{prior_name}({args})"

    @property
    def is_fixed(self):
        """
        Returns True if the prior is fixed and should not be used in the sampler. Does this by checking if this instance
        is an instance of DeltaFunction.


        Returns
        =======
        bool: Whether it's fixed or not!

        """
        return self._is_fixed

    @property
    def latex_label(self):
        """Latex label that can be used for plots.

        Draws from a set of default labels if no label is given

        Returns
        =======
        str: A latex representation for this prior

        """
        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label=None):
        if latex_label is None:
            self.__latex_label = self.__default_latex_label
        else:
            self.__latex_label = latex_label

    @property
    def unit(self):
        return self.__unit

    @unit.setter
    def unit(self, unit):
        self.__unit = unit

    @property
    def latex_label_with_unit(self):
        """ If a unit is specified, returns a string of the latex label and unit """
        if self.unit is not None:
            return "{} [{}]".format(self.latex_label, self.unit)
        else:
            return self.latex_label

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum

    @property
    def width(self):
        return self.maximum - self.minimum

    def get_instantiation_dict(self):
        subclass_args = infer_args_from_method(self.__init__)
        dict_with_properties = get_dict_with_properties(self)
        return {key: dict_with_properties[key] for key in subclass_args}

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if boundary not in ['periodic', 'reflective', None]:
            raise ValueError('{} is not a valid setting for prior boundaries'.format(boundary))
        self._boundary = boundary

    @property
    def __default_latex_label(self):
        if self.name in self._default_latex_labels.keys():
            label = self._default_latex_labels[self.name]
        else:
            label = self.name
        return label

    def to_json(self):
        return json.dumps(self, cls=BilbyJsonEncoder)

    @classmethod
    def from_json(cls, dct):
        return decode_bilby_json(dct)

    @classmethod
    def from_repr(cls, string):
        """Generate the prior from its __repr__"""
        return cls._from_repr(string)

    @classmethod
    def _from_repr(cls, string):
        subclass_args = infer_args_from_method(cls.__init__)

        string = string.replace(' ', '')
        kwargs = cls._split_repr(string)
        for key in kwargs:
            val = kwargs[key]
            if key not in subclass_args and not hasattr(cls, "reference_params"):
                raise AttributeError('Unknown argument {} for class {}'.format(
                    key, cls.__name__))
            else:
                kwargs[key] = cls._parse_argument_string(val)
            if key in ["condition_func", "conversion_function"] and isinstance(kwargs[key], str):
                if "." in kwargs[key]:
                    module = '.'.join(kwargs[key].split('.')[:-1])
                    name = kwargs[key].split('.')[-1]
                else:
                    module = __name__
                    name = kwargs[key]
                kwargs[key] = getattr(import_module(module), name)
        return cls(**kwargs)

    @classmethod
    def _split_repr(cls, string):
        subclass_args = infer_args_from_method(cls.__init__)
        args = string.split(',')
        remove = list()
        for ii, key in enumerate(args):
            for paren_pair in ['()', '{}', '[]']:
                if paren_pair[0] in key:
                    jj = ii
                    while paren_pair[1] not in args[jj]:
                        jj += 1
                        args[ii] = ','.join([args[ii], args[jj]]).strip()
                        remove.append(jj)
        remove.reverse()
        for ii in remove:
            del args[ii]
        kwargs = dict()
        for ii, arg in enumerate(args):
            if '=' not in arg:
                logger.debug(
                    'Reading priors with non-keyword arguments is dangerous!')
                key = subclass_args[ii]
                val = arg
            else:
                split_arg = arg.split('=')
                key = split_arg[0]
                val = '='.join(split_arg[1:])
            kwargs[key] = val
        return kwargs

    @classmethod
    def _parse_argument_string(cls, val):
        """
        Parse a string into the appropriate type for prior reading.

        Four tests are applied in the following order:

        - If the string is 'None':
            `None` is returned.
        - Else If the string is a raw string, e.g., r'foo':
            A stripped version of the string is returned, e.g., foo.
        - Else If the string contains ', e.g., 'foo':
            A stripped version of the string is returned, e.g., foo.
        - Else If the string contains an open parenthesis, (:
            The string is interpreted as a call to instantiate another prior
            class, Bilby will attempt to recursively construct that prior,
            e.g., Uniform(minimum=0, maximum=1), my.custom.PriorClass(**kwargs).
        - Else If the string contains a ".":
            It is treated as a path to a Python function and imported, e.g.,
            "some_module.some_function" returns
            :code:`import some_module; return some_module.some_function`
        - Else:
            Try to evaluate the string using `eval`. Only built-in functions
            and numpy methods can be used, e.g., np.pi / 2, 1.57.


        Parameters
        ==========
        val: str
            The string version of the argument

        Returns
        =======
        val: object
            The parsed version of the argument.

        Raises
        ======
        TypeError:
            If val cannot be parsed as described above.
        """
        if val == 'None':
            val = None
        elif re.sub(r'\'.*\'', '', val) in ['r', 'u']:
            val = val[2:-1]
        elif val.startswith("'") and val.endswith("'"):
            val = val.strip("'")
        elif '(' in val and not val.startswith(("[", "{")):
            other_cls = val.split('(')[0]
            vals = '('.join(val.split('(')[1:])[:-1]
            if "." in other_cls:
                module = '.'.join(other_cls.split('.')[:-1])
                other_cls = other_cls.split('.')[-1]
            else:
                module = __name__.replace('.' + os.path.basename(__file__).replace('.py', ''), '')
            other_cls = getattr(import_module(module), other_cls)
            val = other_cls.from_repr(vals)
        else:
            try:
                val = eval(val, dict(), dict(np=np, inf=np.inf, pi=np.pi))
            except NameError:
                if "." in val:
                    module = '.'.join(val.split('.')[:-1])
                    func = val.split('.')[-1]
                    new_val = getattr(import_module(module), func, val)
                    if val == new_val:
                        raise TypeError(
                            "Cannot evaluate prior, "
                            f"failed to parse argument {val}"
                        )
                    else:
                        val = new_val
        return val


class Constraint(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None):
        super(Constraint, self).__init__(minimum=minimum, maximum=maximum, name=name,
                                         latex_label=latex_label, unit=unit)
        self._is_fixed = True

    def prob(self, val):
        return (val > self.minimum) & (val < self.maximum)


class PriorException(Exception):
    """ General base class for all prior exceptions """
