import numpy as np

from .base import Prior, PriorException
from .interpolated import Interped
from .analytical import DeltaFunction, PowerLaw, Uniform, LogUniform, \
    SymmetricLogUniform, Cosine, Sine, Gaussian, TruncatedGaussian, HalfGaussian, \
    LogNormal, Exponential, StudentT, Beta, Logistic, Cauchy, Gamma, ChiSquared, FermiDirac
from ..utils import infer_args_from_method, infer_parameters_from_function
from ...compat.utils import xp_wrap


def conditional_prior_factory(prior_class):
    class ConditionalPrior(prior_class):

        _leaves = prior_class._leaves

        __name__ = 'Conditional{}'.format(prior_class.__name__)
        __qualname__ = 'Conditional{}'.format(prior_class.__qualname__)

        def __init__(self, condition_func, name=None, latex_label=None, unit=None,
                     boundary=None, **reference_params):
            """

            Parameters
            ==========
            condition_func: func
                Functional form of the condition for this prior. The first function argument
                has to be a dictionary for the `reference_params` (see below). The following
                arguments are the required variables that are required before we can draw this
                prior.
                It needs to return a dictionary with the modified values for the
                `reference_params` that are being used in the next draw.
                For example if we have a Uniform prior for `x` depending on a different variable `y`
                `p(x|y)` with the boundaries linearly depending on y, then this
                could have the following form:

                .. code-block:: python

                    def condition_func(reference_params, y):
                        return dict(
                            minimum=reference_params['minimum'] + y,
                            maximum=reference_params['maximum'] + y
                        )

            name: str, optional
               See superclass
            latex_label: str, optional
                See superclass
            unit: str, optional
                See superclass
            boundary: str, optional
                See superclass
            reference_params:
                Initial values for attributes such as `minimum`, `maximum`.
                This differs on the `prior_class`, for example for the Gaussian
                prior this is `mu` and `sigma`.
            """
            if 'boundary' in infer_args_from_method(super(ConditionalPrior, self).__init__):
                super(ConditionalPrior, self).__init__(name=name, latex_label=latex_label,
                                                       unit=unit, boundary=boundary, **reference_params)
            else:
                super(ConditionalPrior, self).__init__(name=name, latex_label=latex_label,
                                                       unit=unit, **reference_params)

            self._required_variables = None
            self.condition_func = condition_func
            self._reference_params = reference_params

        def sample(self, size=None, *, random_state=None, **required_variables):
            """Draw a sample from the prior

            Parameters
            ==========
            size: int or tuple of ints, optional
                See superclass
            required_variables:
                Any required variables that this prior depends on

            Returns
            =======
            float: See superclass

            """
            from ..utils import random
            rng = random.resolve_random_state(random_state)

            if isinstance(size, int | np.integer):
                size = (size,)

            self.least_recently_sampled = self.rescale(
                rng.uniform(0, 1, size), **required_variables
            )
            return self.least_recently_sampled

        @xp_wrap
        def rescale(self, val, *, xp=None, **required_variables):
            """
            'Rescale' a sample from the unit line element to the prior.

            Parameters
            ==========
            val: Union[float, int, array_like]
                See superclass
            required_variables:
                Any required variables that this prior depends on


            """
            self.update_conditions(**required_variables)
            return super(ConditionalPrior, self).rescale(val, xp=xp)

        @xp_wrap
        def prob(self, val, *, xp=None, **required_variables):
            """Return the prior probability of val.

            Parameters
            ==========
            val: Union[float, int, array_like]
                See superclass
            required_variables:
                Any required variables that this prior depends on


            Returns
            =======
            float: Prior probability of val
            """
            self.update_conditions(**required_variables)
            return super(ConditionalPrior, self).prob(val, xp=xp)

        @xp_wrap
        def ln_prob(self, val, *, xp=None, **required_variables):
            """Return the natural log prior probability of val.

            Parameters
            ==========
            val: Union[float, int, array_like]
                See superclass
            required_variables:
                Any required variables that this prior depends on


            Returns
            =======
            float: Natural log prior probability of val
            """
            self.update_conditions(**required_variables)
            return super(ConditionalPrior, self).ln_prob(val, xp=xp)

        @xp_wrap
        def cdf(self, val, *, xp=None, **required_variables):
            """Return the cdf of val.

            Parameters
            ==========
            val: Union[float, int, array_like]
                See superclass
            required_variables:
                Any required variables that this prior depends on


            Returns
            =======
            float: CDF of val
            """
            self.update_conditions(**required_variables)
            return super(ConditionalPrior, self).cdf(val, xp=xp)

        def update_conditions(self, **required_variables):
            """
            This method updates the conditional parameters (depending on the parent class
            this could be e.g. `minimum`, `maximum`, `mu`, `sigma`, etc.) of this prior
            class depending on the required variables it depends on.

            If no variables are given, the most recently used conditional parameters are kept

            Parameters
            ==========
            required_variables:
                Any required variables that this prior depends on. If none are given,
                self.reference_params will be used.

            """
            required_variables.pop("xp", None)
            if sorted(list(required_variables)) == sorted(self.required_variables):
                parameters = self.condition_func(self.reference_params.copy(), **required_variables)
                for key, value in parameters.items():
                    setattr(self, key, value)
            elif len(required_variables) == 0:
                return
            else:
                raise IllegalRequiredVariablesException("Expected kwargs for {}. Got kwargs for {} instead."
                                                        .format(self.required_variables,
                                                                list(required_variables.keys())))

        @property
        def reference_params(self):
            """
            Initial values for attributes such as `minimum`, `maximum`.
            This depends on the `prior_class`, for example for the Gaussian
            prior this is `mu` and `sigma`. This is read-only.
            """
            return self._reference_params

        @property
        def condition_func(self):
            return self._condition_func

        @condition_func.setter
        def condition_func(self, condition_func):
            if condition_func is None:
                self._condition_func = lambda reference_params: reference_params
            else:
                self._condition_func = condition_func
            self._required_variables = infer_parameters_from_function(self.condition_func)

        @property
        def required_variables(self):
            """ The required variables to pass into the condition function. """
            return self._required_variables

        def get_instantiation_dict(self):
            instantiation_dict = super(ConditionalPrior, self).get_instantiation_dict()
            for key, value in self.reference_params.items():
                instantiation_dict[key] = value
            return instantiation_dict

        def reset_to_reference_parameters(self):
            """
            Reset the object attributes to match the original reference parameters
            """
            for key, value in self.reference_params.items():
                setattr(self, key, value)

        def __repr__(self):
            """Overrides the special method __repr__.

            Returns a representation of this instance that resembles how it is instantiated.
            Works correctly for all child classes

            Returns
            =======
            str: A string representation of this instance

            """
            prior_name = self.__class__.__name__
            instantiation_dict = self.get_instantiation_dict()
            instantiation_dict["condition_func"] = ".".join([
                instantiation_dict["condition_func"].__module__,
                instantiation_dict["condition_func"].__name__
            ])
            args = ', '.join(['{}={}'.format(key, repr(instantiation_dict[key]))
                              for key in instantiation_dict])
            return "{}({})".format(prior_name, args)

    return ConditionalPrior


class ConditionalBasePrior(conditional_prior_factory(Prior)):
    pass


class ConditionalUniform(conditional_prior_factory(Uniform)):
    pass


class ConditionalDeltaFunction(conditional_prior_factory(DeltaFunction)):
    pass


class ConditionalPowerLaw(conditional_prior_factory(PowerLaw)):
    pass


class ConditionalGaussian(conditional_prior_factory(Gaussian)):
    pass


class ConditionalLogUniform(conditional_prior_factory(LogUniform)):
    pass


class ConditionalSymmetricLogUniform(conditional_prior_factory(SymmetricLogUniform)):
    pass


class ConditionalCosine(conditional_prior_factory(Cosine)):
    pass


class ConditionalSine(conditional_prior_factory(Sine)):
    pass


class ConditionalTruncatedGaussian(conditional_prior_factory(TruncatedGaussian)):
    pass


class ConditionalHalfGaussian(conditional_prior_factory(HalfGaussian)):
    pass


class ConditionalLogNormal(conditional_prior_factory(LogNormal)):
    pass


class ConditionalExponential(conditional_prior_factory(Exponential)):
    pass


class ConditionalStudentT(conditional_prior_factory(StudentT)):
    pass


class ConditionalBeta(conditional_prior_factory(Beta)):
    pass


class ConditionalLogistic(conditional_prior_factory(Logistic)):
    pass


class ConditionalCauchy(conditional_prior_factory(Cauchy)):
    pass


class ConditionalGamma(conditional_prior_factory(Gamma)):
    pass


class ConditionalChiSquared(conditional_prior_factory(ChiSquared)):
    pass


class ConditionalFermiDirac(conditional_prior_factory(FermiDirac)):
    pass


class ConditionalInterped(conditional_prior_factory(Interped)):
    pass


class DirichletElement(ConditionalBeta):
    r"""
    Single element in a dirichlet distribution. The probability defined as,

    .. math::
        p(x_n\mid S_n) := \begin{cases}
            1 & N = 1 \\
            \displaystyle\frac{(N - n - 1)(1 - S_n - x_n)^{N - n - 2}}{(1-S_n)^{N-n-1}}
            & n < N - 1 \\
            \displaystyle\delta(1-S_{N-1}-x_{N-1}) & n = N - 1
        \end{cases}, \qquad 0 \leq x_n \leq 1 - S_n

    where, :math:`\delta` is the Dirac delta function, :math:`N` is the total number of
    dimensions and :math:`S_n` is the sum of all previous dimensions,

    .. math::
        S_n := \sum_{i=0}^{n-1}x_i \qquad n < N

    Examples
    ========

        :code:`n_dimensions` = :math:`N = 1`:

            .. math::
                p(x_0) = 1 ; 0 \leq x_0 \leq 1

        :code:`n_dimensions` = :math:`N = 2`:

            .. math::
                    p(x_0)         = 1                ; \qquad 0 \leq x_0 \leq 1

            .. math::
                    p(x_1\mid x_0) = \delta(1-x_0-x_1); \qquad 0 \leq x_1 \leq 1 - x_0

    Parameters
    ==========
    order: int
        Order of this element of the dirichlet distribution. Equivalent to :math:`n`.
    n_dimensions: int
        Total number of elements of the dirichlet distribution. Equivalent to :math:`N`.
    label: str
        Label for the dirichlet distribution.
        This should be the same for all elements.
    """

    _leaves = []

    def __init__(self, order, n_dimensions, label):
        """ """
        super(DirichletElement, self).__init__(
            minimum=0, maximum=1, alpha=1, beta=n_dimensions - order - 1,
            name=label + str(order),
            condition_func=self.dirichlet_condition
        )
        self.label = label
        self.n_dimensions = n_dimensions
        self.order = order
        self._required_variables = [
            label + str(ii) for ii in range(order)
        ]
        self.__class__.__name__ = 'DirichletElement'
        self.__class__.__qualname__ = 'DirichletElement'

    def dirichlet_condition(self, reference_parms, **kwargs):
        remaining = 1 - sum(
            [kwargs[self.label + str(ii)] for ii in range(self.order)]
        )
        return dict(minimum=reference_parms["minimum"], maximum=remaining)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        return Prior.get_instantiation_dict(self)

    def pytree_flatten(self):
        children = ()
        aux_data = (self.order, self.n_dimensions, self.label)
        return children, aux_data

    @classmethod
    def pytree_unflatten(cls, aux_data, children):
        print(aux_data, children)
        return cls(*aux_data)


class ConditionalPriorException(PriorException):
    """ General base class for all conditional prior exceptions """


class IllegalRequiredVariablesException(ConditionalPriorException):
    """ Exception class for exceptions relating to handling the required variables. """
