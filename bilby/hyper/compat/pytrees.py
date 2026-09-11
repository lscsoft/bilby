from jax.tree_util import register_pytree_node

from ...compat.pytrees.likelihood import likelihood_flatten, likelihood_unflatten
from ..likelihood import HyperparameterLikelihood
from ..model import Model


def hyperparameter_likelihood_flatten(likelihood: HyperparameterLikelihood):
    _, aux_data = likelihood_flatten(likelihood)
    children = (
        likelihood.data,
        likelihood.evidence_factor,
        likelihood.posteriors,
        likelihood.samples_factor,
        likelihood.hyper_prior,
    )
    aux_data += (
        likelihood.sampling_prior,
    )
    return children, aux_data


def hyperparameter_likelihood_unflatten(aux_data, flat) -> HyperparameterLikelihood:
    likelihood = likelihood_unflatten(aux_data, flat)
    data, evidence_factor, posteriors, samples_factor, hyper_prior = flat[:5]
    sampling_prior = aux_data[2]
    likelihood.data = data
    likelihood.evidence_factor = evidence_factor
    likelihood.posteriors = posteriors
    likelihood.samples_factor = samples_factor
    likelihood.hyper_prior = hyper_prior
    likelihood.sampling_prior = sampling_prior
    likelihood.n_posteriors = data["prior"].shape[0]
    likelihood.max_samples = data["prior"].shape[1]
    likelihood.samples_per_posterior = data["prior"].shape[1]
    return likelihood


def model_flatten(model: Model):
    aux_data = tuple(model.models)
    return (), aux_data


def model_unflatten(aux_data, flat) -> Model:
    model = Model.__new__(Model)
    model.cache = False
    model.models = list(aux_data)
    return model


register_pytree_node(
    HyperparameterLikelihood, hyperparameter_likelihood_flatten, hyperparameter_likelihood_unflatten
)
register_pytree_node(Model, model_flatten, model_unflatten)
