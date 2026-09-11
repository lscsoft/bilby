from interpax import Interpolator1D
from jax.tree_util import register_pytree_node

from ...core.series import CoupledTimeAndFrequencySeries
from ...core.utils.calculus import WrappedInterp1d


def interp1d_flatten(interp: WrappedInterp1d):
    children = (interp.x, interp.y, interp.fill_value)
    aux_data = (interp.kind,)
    return children, aux_data


def interp1d_unflatten(aux_data, children) -> Interpolator1D:
    x, y, fill_value = children
    kind = aux_data[0]
    return Interpolator1D(x=x, f=y, method=kind, extrap=fill_value)


def tfs_flatten(tfs: CoupledTimeAndFrequencySeries):
    children = (
        tfs.start_time,
        tfs._time_array,
        tfs._frequency_array,
    )
    aux_data = (
        tfs._duration,
        tfs._sampling_frequency,
        tfs._frequency_array_updated,
        tfs._time_array_updated,
    )
    return children, aux_data


def tfs_unflatten(aux_data, children) -> CoupledTimeAndFrequencySeries:
    tfs = CoupledTimeAndFrequencySeries.__new__(CoupledTimeAndFrequencySeries)

    tfs._start_time = children[0]
    tfs._time_array = children[1]
    tfs._frequency_array = children[2]

    tfs._duration = aux_data[0]
    tfs._sampling_frequency = aux_data[1]
    tfs._frequency_array_updated = aux_data[2]
    tfs._time_array_updated = aux_data[3]

    return tfs


register_pytree_node(WrappedInterp1d, interp1d_flatten, interp1d_unflatten)
register_pytree_node(CoupledTimeAndFrequencySeries, tfs_flatten, tfs_unflatten)
