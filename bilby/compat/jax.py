import jax
import jax.numpy as jnp
from ..core.likelihood import Likelihood
from . import pytrees  # noqa


class JittedLikelihood(Likelihood):
    """
    A wrapper to just-in-time compile a :code:`Bilby` likelihood for use with :code:`jax`.

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood
        The likelihood to wrap.
    cast_to_float: bool
        Whether to return a float instead of a :code:`jax.Array`.
    """

    def __init__(self, likelihood, cast_to_float=True):
        self._likelihood = likelihood
        self._ll = jax.jit(likelihood.log_likelihood)
        self._llr = jax.jit(likelihood.log_likelihood_ratio)
        self.cast_to_float = cast_to_float
        super().__init__()

    def __getattr__(self, name):
        return getattr(self._likelihood, name)

    def log_likelihood(self, parameters):
        parameters = {k: jnp.array(v) for k, v in parameters.items()}
        ln_l = self._ll(parameters)
        if self.cast_to_float:
            ln_l = float(ln_l)
        return ln_l

    def log_likelihood_ratio(self, parameters):
        parameters = {k: jnp.array(v) for k, v in parameters.items()}
        ln_l = self._llr(parameters)
        if self.cast_to_float:
            ln_l = float(ln_l)
        return ln_l
