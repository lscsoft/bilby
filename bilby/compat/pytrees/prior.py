# flake8: noqa
# global imports make pre-commits unhappy

from jax.tree_util import register_pytree_node

from ...core.prior.analytical import *
from ...core.prior.base import Prior
from ...core.prior.conditional import *
from ...core.prior.dict import PriorDict, ConditionalPriorDict, DirichletPriorDict
from ...core.prior.interpolated import Interped, FromFile
from ...core.prior.slabspike import SlabSpikePrior
from ...gw.prior import *


def prior_flatten(prior: Prior):
    props = prior.get_instantiation_dict()
    for key in ["name", "unit", "latex_label", "_boundary", "_minimum", "_maximum"]:
        if key not in props and key.strip("_") not in props:
            props[key] = getattr(prior, key, None)
    child_props = dict()
    for key in prior._leaves:
        if key in props:
            child_props[key] = props.pop(key)
        elif key.strip("_") in props:
            child_props[key] = props.pop(key.strip("_"))
        else:
            child_props[key] = getattr(prior, key)
    aux_props = {key: props[key] for key in set(props.keys()).difference(prior._leaves)}
    children = (prior.least_recently_sampled, child_props)
    aux_data = (prior.__class__, prior.is_fixed, aux_props)
    # print(prior, children, aux_data)
    return children, aux_data


def prior_unflatten(aux_data, children) -> Prior:
    cls, is_fixed, aux_props = aux_data
    least_recently_sampled, child_props = children
    prior = cls.__new__(cls)
    prior._is_fixed = is_fixed
    prior.least_recently_sampled = least_recently_sampled
    for key in ["name", "unit", "latex_label", "_boundary"]:
        if key in aux_props:
            setattr(prior, key, aux_props.pop(key))
    for k, v in child_props.items():
        setattr(prior, k, v)
    for k, v in aux_props.items():
        setattr(prior, k, v)
    # print(prior)
    return prior


def conditional_flatten(prior: ConditionalBasePrior):
    children, aux_data = prior_flatten(prior)

    children += (prior._reference_params,)

    aux_data += (prior._required_variables, prior._condition_func)

    return children, aux_data


def conditional_unflatten(aux_data, children) -> ConditionalBasePrior:
    prior = prior_unflatten(aux_data[:3], children[:2])
    reference_params = children[2]
    required_variables, condition_func = aux_data[3:5]
    prior._reference_params = reference_params
    prior._required_variables = required_variables
    prior._condition_func = condition_func
    return prior


def dict_flatten(prior_dict: PriorDict):
    prior_dict.convert_floats_to_delta_functions()
    children = (
        {k: v for k, v in prior_dict.items()},
    )
    aux_data = (
        prior_dict.__class__,
        prior_dict.conversion_function,
        prior_dict._cached_normalizations,
    )
    return children, aux_data


def dict_unflatten(aux_data, children) -> PriorDict:
    cls, conversion_function, cached_normalizations = aux_data
    prior_dict = cls.__new__(cls)
    prior_dict.conversion_function = conversion_function
    prior_dict._cached_normalizations = cached_normalizations
    for k, v in children[0].items():
        prior_dict[k] = v
    return prior_dict


def conditional_dict_flatten(prior_dict: ConditionalPriorDict):
    children, aux_data = dict_flatten(prior_dict)

    aux_data += (
        prior_dict._conditional_keys,
        prior_dict._unconditional_keys,
        prior_dict._rescale_keys,
        prior_dict._rescale_indexes,
        prior_dict._least_recently_rescaled_keys,
        prior_dict._resolved,
    )

    return children, aux_data


def conditional_dict_unflatten(aux_data, children) -> ConditionalPriorDict:
    prior_dict = dict_unflatten(aux_data[:3], children[:1])
    (
        conditional,
        unconditional,
        rescale,
        indexes,
        least_recently_rescaled,
        resolved,
    ) = aux_data[3:9]
    prior_dict._conditional_keys = conditional
    prior_dict._unconditional_keys = unconditional
    prior_dict._rescale_keys = rescale
    prior_dict._rescale_indexes = indexes
    prior_dict._least_recently_rescaled_keys = least_recently_rescaled
    prior_dict._resolved = resolved
    return prior_dict


def dirichlet_dict_flatten(prior_dict):
    children, aux_data = conditional_dict_flatten(prior_dict)

    aux_data += (prior_dict.n_dim, prior_dict.label)

    return children, aux_data


def dirichlet_dict_unflatten(aux_data, children):
    prior_dict = conditional_dict_unflatten(aux_data[:8], children[:2])
    n_dim, label = aux_data[8:10]
    prior_dict.n_dim = n_dim
    prior_dict.label = label
    return prior_dict


register_pytree_node(PriorDict, dict_flatten, dict_unflatten)
register_pytree_node(ConditionalPriorDict, conditional_dict_flatten, conditional_dict_unflatten)
register_pytree_node(DirichletPriorDict, dirichlet_dict_flatten, dirichlet_dict_unflatten)
register_pytree_node(CBCPriorDict, conditional_dict_flatten, conditional_dict_unflatten)
register_pytree_node(BBHPriorDict, conditional_dict_flatten, conditional_dict_unflatten)
register_pytree_node(BNSPriorDict, conditional_dict_flatten, conditional_dict_unflatten)

for cls in list(globals().values()):
    if not isinstance(cls, type) or not issubclass(cls, Prior):
        continue
    elif hasattr(cls, "pytree_flatten"):
        register_pytree_node(cls, cls.pytree_flatten, cls.pytree_unflatten)
    elif hasattr(cls, "condition_func"):
        register_pytree_node(cls, conditional_flatten, conditional_unflatten)
    else:
        register_pytree_node(cls, prior_flatten, prior_unflatten)
