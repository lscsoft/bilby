from functools import partial

import jax
from jax.tree_util import register_pytree_node

from ...core.likelihood import (
    Likelihood,
    Analytical1DLikelihood,
    AnalyticalMultidimensionalBimodalCovariantGaussian,
    AnalyticalMultidimensionalCovariantGaussian,
    ExponentialLikelihood,
    GaussianLikelihood,
    JointLikelihood,
    Multinomial,
    PoissonLikelihood,
    StudentTLikelihood,
    ZeroLikelihood,
)


def likelihood_flatten(likelihood: Likelihood):
    children = ()
    aux_data = (
        likelihood.__class__,
        likelihood._marginalized_parameters,
    )
    return children, aux_data


def likelihood_unflatten(aux_data, flat) -> Likelihood:
    likelihood_cls, marginalized_parameters = aux_data[:2]
    likelihood = likelihood_cls.__new__(likelihood_cls)
    likelihood._marginalized_parameters = marginalized_parameters
    likelihood._parameters = dict()
    return likelihood


def zero_likelihood_flatten(likelihood: ZeroLikelihood):
    _, aux_data = likelihood_flatten(likelihood)
    children = (likelihood._parent,)
    return children, aux_data


def zero_likelihood_unflatten(aux_data, flat) -> ZeroLikelihood:
    likelihood = likelihood_unflatten(aux_data, flat)
    parent = flat[0]
    likelihood._parent = parent
    return likelihood


def analytical_1d_likelihood_flatten(likelihood: Analytical1DLikelihood):
    _, aux_data = likelihood_flatten(likelihood)
    children = (likelihood._x, likelihood._y)
    aux_data += (likelihood._func, likelihood._function_keys, likelihood.kwargs)
    return children, aux_data


def analytical_1d_likelihood_unflatten(aux_data, flat) -> Analytical1DLikelihood:
    likelihood = likelihood_unflatten(aux_data, flat)
    func, function_keys, kwargs = aux_data[2:5]
    likelihood._func = func
    likelihood._function_keys = function_keys
    likelihood.kwargs = kwargs
    x, y = flat[:2]
    likelihood._x = x
    likelihood._y = y
    return likelihood


def gaussian_likelihood_flatten(likelihood: GaussianLikelihood):
    children, aux_data = analytical_1d_likelihood_flatten(likelihood)
    children += (likelihood._sigma,)
    return children, aux_data


def gaussian_likelihood_unflatten(aux_data, flat) -> GaussianLikelihood:
    likelihood = analytical_1d_likelihood_unflatten(aux_data, flat)
    sigma = flat[2]
    likelihood._sigma = sigma
    return likelihood


def student_t_likelihood_flatten(likelihood: StudentTLikelihood):
    children, aux_data = analytical_1d_likelihood_flatten(likelihood)
    children += (likelihood.sigma,)
    aux_data += (likelihood._nu,)
    return children, aux_data


def student_t_likelihood_unflatten(aux_data, flat) -> StudentTLikelihood:
    likelihood = analytical_1d_likelihood_unflatten(aux_data, flat)
    sigma = flat[2]
    nu = aux_data[5]
    likelihood._nu = nu
    likelihood.sigma = sigma
    return likelihood


def multinomial_flatten(likelihood: Multinomial):
    children, aux_data = likelihood_flatten(likelihood)
    children += (likelihood.data, likelihood._total, likelihood._nll)
    aux_data += (likelihood.n, likelihood.base)
    return children, aux_data


def multinomial_unflatten(aux_data, flat) -> Multinomial:
    likelihood = likelihood_unflatten(aux_data, flat)
    data, total, nll = flat[:3]
    n, base = aux_data[2:4]
    likelihood.data = data
    likelihood._total = total
    likelihood._nll = nll
    likelihood.n = n
    likelihood.base = base
    return likelihood


def joint_likelihood_flatten(likelihood: JointLikelihood):
    children = tuple(likelihood._likelihoods)
    _, aux_data = likelihood_flatten(likelihood)
    return children, aux_data


def joint_likelihood_unflatten(aux_data, flat) -> JointLikelihood:
    likelihood = likelihood_unflatten(aux_data, flat)
    likelihood._likelihoods = list(flat)
    return likelihood


def analytical_multidimensional_covariant_gaussian_flatten(
    likelihood: AnalyticalMultidimensionalCovariantGaussian
):
    _, aux_data = likelihood_flatten(likelihood)
    children = (likelihood.cov, likelihood.mean, likelihood.sigma)
    return children, aux_data


def analytical_multidimensional_covariant_gaussian_unflatten(
    aux_data, flat
) -> AnalyticalMultidimensionalCovariantGaussian:
    likelihood = likelihood_unflatten(aux_data, flat)
    cov, mean, sigma = flat
    likelihood.cov = cov
    likelihood.mean = mean
    likelihood.sigma = sigma
    likelihood.logpdf = partial(jax.scipy.stats.multivariate_normal.logpdf, mean=mean, cov=cov)
    return likelihood


def analytical_multidimensional_bimodal_covariant_gaussian_flatten(
    likelihood: AnalyticalMultidimensionalBimodalCovariantGaussian
):
    _, aux_data = likelihood_flatten(likelihood)
    children = (likelihood.cov, likelihood.mean_1, likelihood.mean_2, likelihood.sigma)
    return children, aux_data


def analytical_multidimensional_bimodal_covariant_gaussian_unflatten(
    aux_data, flat
) -> AnalyticalMultidimensionalBimodalCovariantGaussian:
    likelihood = likelihood_unflatten(aux_data, flat)
    cov, mean_1, mean_2, sigma = flat
    likelihood.cov = cov
    likelihood.mean_1 = mean_1
    likelihood.mean_2 = mean_2
    likelihood.sigma = sigma
    likelihood.logpdf_1 = partial(jax.scipy.stats.multivariate_normal.logpdf, mean=mean_1, cov=cov)
    likelihood.logpdf_2 = partial(jax.scipy.stats.multivariate_normal.logpdf, mean=mean_2, cov=cov)
    return likelihood


for tpl in [
    (Likelihood, likelihood_flatten, likelihood_unflatten),
    (GaussianLikelihood, gaussian_likelihood_flatten, gaussian_likelihood_unflatten),
    (ZeroLikelihood, zero_likelihood_flatten, zero_likelihood_unflatten),
    (Analytical1DLikelihood, analytical_1d_likelihood_flatten, analytical_1d_likelihood_unflatten),
    (PoissonLikelihood, analytical_1d_likelihood_flatten, analytical_1d_likelihood_unflatten),
    (ExponentialLikelihood, analytical_1d_likelihood_flatten, analytical_1d_likelihood_unflatten),
    (StudentTLikelihood, student_t_likelihood_flatten, student_t_likelihood_unflatten),
    (Multinomial, multinomial_flatten, multinomial_unflatten),
    (JointLikelihood, joint_likelihood_flatten, joint_likelihood_unflatten),
    (
        AnalyticalMultidimensionalCovariantGaussian,
        analytical_multidimensional_covariant_gaussian_flatten,
        analytical_multidimensional_covariant_gaussian_unflatten,
    ),
    (
        AnalyticalMultidimensionalBimodalCovariantGaussian,
        analytical_multidimensional_bimodal_covariant_gaussian_flatten,
        analytical_multidimensional_bimodal_covariant_gaussian_unflatten,
    ),
]:
    register_pytree_node(*tpl)
