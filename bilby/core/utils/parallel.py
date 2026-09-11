from contextlib import contextmanager
from copy import deepcopy

import attr

from .log import logger


@attr.s
class _SamplingContainer:
    """
    A container class for objects that are stored independently in each thread
    for some samplers.

    A single instance of this will appear in this module that can be access
    by the individual samplers.

    This includes the:

    - likelihood (bilby.core.likelihood.Likelihood)
    - priors (bilby.core.prior.PriorDict)
    - search_parameter_keys (list)
    - use_ratio (bool)
    """

    likelihood = attr.ib(default=None)
    priors = attr.ib(default=None)
    search_parameter_keys = attr.ib(default=None)
    use_ratio = attr.ib(default=False)
    parameters = attr.ib(default=None)


sampling_convenience_dump = _SamplingContainer()


def initialize_global_variables(
    likelihood,
    priors,
    search_parameter_keys,
    use_ratio,
    parameters,
):
    """
    Store a global copy of the likelihood, priors, and search keys for
    multiprocessing.

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood
        The likelihood to copy into each process
    priors: bilby.core.prior.PriorDict
        The Bilby prior dictionary to copy into each process
    search_parameter_keys: list[str], None
        The names for parameters being sampled over
    use_ratio: bool
        Whether to evaluate the log_likelihood_ratio
    parameters: dict
        A set of default parameters to pass through to the new processes,
        e.g., if there are fixed parameters that the sampler used doesn't use.
    """
    sampling_convenience_dump.likelihood = likelihood
    sampling_convenience_dump.priors = priors
    sampling_convenience_dump.search_parameter_keys = search_parameter_keys
    sampling_convenience_dump.use_ratio = use_ratio
    sampling_convenience_dump.parameters = deepcopy(parameters)


def create_pool(
    likelihood=None,
    priors=None,
    use_ratio=None,
    search_parameter_keys=None,
    npool=None,
    pool=None,
    parameters=None,
):
    """
    Create a parallel pool object that is initialized with variables typically
    needed by Bilby for parallel tasks.

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood, None
        The likelihood to copy into each process
    priors: bilby.core.prior.PriorDict, None
        The Bilby prior dictionary to copy into each process
    use_ratio: bool, None
        Whether to evaluate the log_likelihood_ratio
    search_parameter_keys: list[str], None
        The names for parameters being sampled over
    npool: int, None
        The number of processes to use for multiprocessing.
        If a user pool is not provided and this is either :code:`1` or :code:`None`,
        this functions returns :code:`None`.
    pool: pool-like, str, None
        Either a premade pool object, or the pool kind (:code:`mpi`, :code:`multiprocessing`).
        If a pre-made pool is passed, it is returned directly with no checks
        performed.
    parameters: dict, None
        Parameters to pass through to the new processes, e.g., if default
        parameters are to be passed.

    Returns
    =======
    pool: schwimmbad.MPIPool, multiprocessing.Pool, None
        Returns either a pool that can be used for mapping function calls.
        Each process attached to the pool has been initialized with
        the :code:`bilby.core.utils.parallel.sampling_convenience_dump`.

    Examples
    ========

    >>> import numpy as np
    >>> from bilby.core.likelihood import AnalyticalMultidimensionalCovariantGaussian
    >>> from bilby.core.prior import Normal, PriorDict
    >>> from bilby.core.utils.parallel import close_pool, create_pool

    >>> likelihood = AnalyticalMultidimensionalCovariantGaussian(
    ...     mean=np.zeros(4), cov=np.eye(4)
    ... )
    >>> priors = PriorDict({f"x{ii}": Normal(0, 1) for ii in range(4)})
    >>> parameters = [priors.sample() for _ in range(10)]
    >>> pool = create_pool(likelihood, priors, npool=4)
    >>> log_ls = list(pool.map(likelihood.log_likelihood, parameters))
    >>> close_pool(pool)

    .. note::

        The above example passes the ``likelihood.log_likelihood`` method directly to the pool.
        This is possible, but will lead to the likelihood object being pickled and sent to each
        process for every call to the pool. This can add significant overhead if the likelihood
        carries a lot of data, e.g., for the gravitational-wave transient likelihoods.
        In this case, we recommend creating a wrapper function that uses the
        ``sampling_convenience_dump`` to access the likelihood object, e.g.,

        .. code-block:: python

            >>> from bilby.core.utils.parallel import sampling_convenience_dump

            >>> def parallel_likelihood_eval(parameters):
            ...     return sampling_convenience_dump.likelihood.log_likelihood(parameters)
    """
    from ...core.sampler.base_sampler import initialize_global_variables

    if parameters is None:
        parameters = dict()

    _pool = None
    if pool == "mpi":
        try:
            from schwimmbad import MPIPool
        except ImportError:
            raise ImportError("schwimmbad must be installed to use MPI pool")

        initialize_global_variables(
            likelihood=likelihood,
            priors=priors,
            search_parameter_keys=search_parameter_keys,
            use_ratio=use_ratio,
            parameters=parameters,
        )
        _pool = MPIPool(use_dill=True)
        if _pool.is_master():
            logger.info(f"Created MPI pool with size {_pool.size}")
    elif pool is not None:
        _pool = pool
    elif npool not in (None, 1):
        import multiprocessing

        _pool = multiprocessing.Pool(
            processes=npool,
            initializer=initialize_global_variables,
            initargs=(likelihood, priors, search_parameter_keys, use_ratio, parameters),
        )
        logger.info(f"Created multiprocessing pool with size {npool}")
    else:
        _pool = None
    return _pool


def close_pool(pool):
    """
    Safely close a parallel pool.
    If the pool has a :code:`close` method :code:`pool.close` will be called.
    Then, if the pool has a :code:`join` method :code:`pool.join` will be called.
    """
    if hasattr(pool, "close"):
        pool.close()
    if hasattr(pool, "join"):
        pool.join()


@contextmanager
def bilby_pool(
    likelihood=None,
    priors=None,
    use_ratio=None,
    search_parameter_keys=None,
    npool=None,
    pool=None,
    parameters=None,
):
    """
    Yield a parallel pool object that is initialized with variables typically
    needed by Bilby for parallel tasks that is automatically close when closing
    the context.

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood, None
        The likelihood to copy into each process
    priors: bilby.core.prior.PriorDict, None
        The Bilby prior dictionary to copy into each process
    use_ratio: bool, None
        Whether to evaluate the log_likelihood_ratio
    search_parameter_keys: list[str], None
        The names for parameters being sampled over
    npool: int, None
        The number of processes to use for multiprocessing.
        If a user pool is not provided and this is either :code:`1` or :code:`None`,
        this functions returns :code:`None`.
    pool: pool-like, str, None
        Either a premade pool object, or the pool kind (:code:`mpi`, :code:`multiprocessing`).
        If a pre-made pool is passed, it is returned directly with no checks
        performed.
    parameters: dict, None
        Parameters to pass through to the new processes, e.g., if default
        parameters are to be passed.

    Yields
    ======
    pool: schwimmbad.MPIPool, multiprocessing.Pool, None
        Returns either a pool that can be used for mapping function calls.
        Each process attached to the pool has been initialized with
        the :code:`bilby.core.utils.parallel.sampling_convenience_dump`.

    Examples
    ========

    >>> import numpy as np
    >>> from bilby.core.likelihood import AnalyticalMultidimensionalCovariantGaussian
    >>> from bilby.core.prior import Normal, PriorDict
    >>> from bilby.core.utils.parallel import bilby_pool

    >>> likelihood = AnalyticalMultidimensionalCovariantGaussian(
    ...     mean=np.zeros(4), cov=np.eye(4)
    ... )
    >>> priors = PriorDict({f"x{ii}": Normal(0, 1) for ii in range(4)})
    >>> parameters = [priors.sample() for _ in range(10)]
    >>> with bilby_pool(likelihood, priors, npool=4) as pool:
    ...     log_ls = list(pool.map(likelihood.log_likelihood, parameters))

    .. note::

        The above example passes the ``likelihood.log_likelihood`` method directly to the pool.
        This is possible, but will lead to the likelihood object being pickled and sent to each
        process for every call to the pool. This can add significant overhead if the likelihood
        carries a lot of data, e.g., for the gravitational-wave transient likelihoods.
        In this case, we recommend creating a wrapper function that uses the
        ``sampling_convenience_dump`` to access the likelihood object, e.g.,

        .. code-block:: python

            >>> from bilby.core.utils.parallel import sampling_convenience_dump

            >>> def parallel_likelihood_eval(parameters):
            ...     return sampling_convenience_dump.likelihood.log_likelihood(parameters)
    """
    if hasattr(pool, "map"):
        user_pool = True
    else:
        user_pool = False

    try:
        _pool = create_pool(
            likelihood=likelihood,
            priors=priors,
            search_parameter_keys=search_parameter_keys,
            use_ratio=use_ratio,
            npool=npool,
            pool=pool,
            parameters=parameters,
        )
        yield _pool
    finally:
        if not user_pool and "_pool" in locals():
            close_pool(_pool)
