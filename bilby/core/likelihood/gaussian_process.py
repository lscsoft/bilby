import numpy as np

from .base import Likelihood
from ..utils import infer_args_from_function_except_n_args


def function_to_celerite_mean_model(func):
    from celerite.modeling import Model as CeleriteModel
    return _function_to_gp_model(func, CeleriteModel)


def function_to_george_mean_model(func):
    from celerite.modeling import Model as GeorgeModel
    return _function_to_gp_model(func, GeorgeModel)


def _function_to_gp_model(func, cls):
    class MeanModel(cls):
        parameter_names = tuple(infer_args_from_function_except_n_args(func=func, n=1))

        def get_value(self, t):
            params = {name: getattr(self, name) for name in self.parameter_names}
            return func(t, **params)

        def compute_gradient(self, *args, **kwargs):
            pass

    return MeanModel


class _GPLikelihood(Likelihood):

    def __init__(self, kernel, mean_model, t, y, yerr=1e-6, gp_class=None):
        """
            Basic Gaussian Process likelihood interface for `celerite` and `george`.
            For `celerite` documentation see: https://celerite.readthedocs.io/en/stable/
            For `george` documentation see: https://george.readthedocs.io/en/latest/

            Parameters
            ==========
            kernel: Union[celerite.term.Term, george.kernels.Kernel]
                `celerite` or `george` kernel. See the respective package documentation about the usage.
            mean_model: Union[celerite.modeling.Model, george.modeling.Model]
                Mean model
            t: array_like
                The `times` or `x` values of the data set.
            y: array_like
                The `y` values of the data set.
            yerr: float, int, array_like, optional
                The error values on the y-values. If a single value is given, it is assumed that the value
                applies for all y-values. Default is 1e-6, effectively assuming that no y-errors are present.
            gp_class: type, None, optional
                GPClass to use. This is determined by the child class used to instantiate the GP. Should usually
                not be given by the user and is mostly used for testing
        """
        self.kernel = kernel
        self.mean_model = mean_model
        self.t = np.array(t)
        self.y = np.array(y)
        self.yerr = np.array(yerr)
        self.GPClass = gp_class
        self.gp = self.GPClass(kernel=self.kernel, mean=self.mean_model, fit_mean=True, fit_white_noise=True)
        self.gp.compute(self.t, yerr=self.yerr)
        super().__init__(parameters=self.gp.get_parameter_dict())

    def set_parameters(self, parameters):
        """
        Safely set a set of parameters to the internal instances of the `gp` and `mean_model`, as well as the
        `parameters` dict.

        Parameters
        ==========
        parameters: dict, pandas.DataFrame
            The set of parameters we would like to set.
        """
        for name, value in parameters.items():
            try:
                self.gp.set_parameter(name=name, value=value)
            except ValueError:
                pass
            self.parameters[name] = value


class CeleriteLikelihood(_GPLikelihood):

    def __init__(self, kernel, mean_model, t, y, yerr=1e-6):
        """
            Basic Gaussian Process likelihood interface for `celerite` and `george`.
            For `celerite` documentation see: https://celerite.readthedocs.io/en/stable/
            For `george` documentation see: https://george.readthedocs.io/en/latest/

            Parameters
            ==========
            kernel: celerite.term.Term
                `celerite` or `george` kernel. See the respective package documentation about the usage.
            mean_model: celerite.modeling.Model
                Mean model
            t: array_like
                The `times` or `x` values of the data set.
            y: array_like
                The `y` values of the data set.
            yerr: float, int, array_like, optional
                The error values on the y-values. If a single value is given, it is assumed that the value
                applies for all y-values. Default is 1e-6, effectively assuming that no y-errors are present.
        """
        import celerite
        super().__init__(kernel=kernel, mean_model=mean_model, t=t, y=y, yerr=yerr, gp_class=celerite.GP)

    def log_likelihood(self):
        """
        Calculate the log-likelihood for the Gaussian process given the current parameters.

        Returns
        =======
        float: The log-likelihood value.
        """
        self.gp.set_parameter_vector(vector=np.array(list(self.parameters.values())))
        try:
            return self.gp.log_likelihood(self.y)
        except Exception:
            return -np.inf


class GeorgeLikelihood(_GPLikelihood):

    def __init__(self, kernel, mean_model, t, y, yerr=1e-6):
        """
            Basic Gaussian Process likelihood interface for `celerite` and `george`.
            For `celerite` documentation see: https://celerite.readthedocs.io/en/stable/
            For `george` documentation see: https://george.readthedocs.io/en/latest/

            Parameters
            ==========
            kernel: george.kernels.Kernel
                `celerite` or `george` kernel. See the respective package documentation about the usage.
            mean_model: george.modeling.Model
                Mean model
            t: array_like
                The `times` or `x` values of the data set.
            y: array_like
                The `y` values of the data set.
            yerr: float, int, array_like, optional
                The error values on the y-values. If a single value is given, it is assumed that the value
                applies for all y-values. Default is 1e-6, effectively assuming that no y-errors are present.
        """
        import george
        super().__init__(kernel=kernel, mean_model=mean_model, t=t, y=y, yerr=yerr, gp_class=george.GP)

    def log_likelihood(self):
        """
        Calculate the log-likelihood for the Gaussian process given the current parameters.

        Returns
        =======
        float: The log-likelihood value.
        """
        for name, value in self.parameters.items():
            try:
                self.gp.set_parameter(name=name, value=value)
            except ValueError:
                raise ValueError(f"Parameter {name} not a valid parameter for the GP.")
        try:
            return self.gp.log_likelihood(self.y)
        except Exception:
            return -np.inf


