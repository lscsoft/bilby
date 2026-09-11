================
Parallelisation
================

Many of the samplers supported by Bilby can leverage parallelisation over multiple
CPU cores. There are a range of methods that can be used to achieve this, including
the built-in ``multiprocessing`` module to parallelize over multiple cores on a
single machine, or using MPI to parallelize over multiple nodes on a computing cluster.
This page specifically describes how to use parallelisation of likelihood calls across
multiple processes, rather than parallelisation inside likelihood calls, e.g., using
GPUs via the `Python array API <array_api.rst>`__.

Parallelisation in Bilby relies on global storage of the likelihood and priors inside
``bilby.core.utils.parallel.sampling_convenience_dump``, which must be initialized in
each worker process. We support multiple ways of performing this initialization.

.. note::

    Recent versions of Python include freethreaded builds allowing efficient
    thread-based parallelism. We do not currently support thread-based parallelism
    due to the use of global storage of the likelihood and priors.
    However, we plan to remove this limitation in a future release.

Default parallelisation
=======================

The default approach is to set ``npool`` when calling ``run_sampler``. Bilby
creates a :class:`multiprocessing.Pool`, initializes each worker with the
likelihood and priors, and closes the pool when the run completes. An ``npool``
of ``1`` (the default) or ``None`` runs without a pool.

.. code-block:: python

	import bilby

	result = bilby.run_sampler(
		likelihood=likelihood,
		priors=priors,
		sampler="dynesty",
		npool=4,
	)

The number of workers that a sampler actually uses can depend on that
sampler's options. Refer to the sampler-specific documentation when choosing
the pool size.

``run_sampler`` also passes its pool to the result conversion function. This
means that a conversion function which accepts the optional ``npool`` and
``pool`` arguments can use the same workers after sampling. For example, the
gravitational-wave conversion functions use this to calculate quantities for
each posterior sample in parallel.

.. code-block:: python

	result = bilby.run_sampler(
		likelihood=likelihood,
		priors=priors,
		sampler="dynesty",
		npool=4,
		conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
	)

The conversion function should pass ``npool`` and ``pool`` through to any
Bilby helper which supports parallel processing. A minimal function with the
expected signature is:

.. code-block:: python

	def add_parameters(samples, likelihood=None, priors=None, npool=1, pool=None):
		samples = samples.copy()
		# Calculate and add derived parameters to samples here.
		return samples

Managing a pool with ``bilby_pool``
===================================

Use ``bilby_pool`` when performing parallel work outside ``run_sampler``. It
creates and initializes a pool, yields it, and closes only pools it created.
The context manager also accepts an existing pool without closing it.

.. code-block:: python

	from bilby.core.utils.parallel import bilby_pool

	with bilby_pool(likelihood=likelihood, priors=priors, npool=4) as pool:
		values = list(pool.map(evaluate_sample, samples))

For ``npool=1`` or ``None``, the yielded value is ``None``. Code using the
pool should therefore provide a serial fallback:

.. code-block:: python

	with bilby_pool(likelihood=likelihood, priors=priors, npool=npool) as pool:
		map_function = pool.map if pool is not None else map
		values = list(map_function(evaluate_sample, samples))

Using a custom pool
===================

To run with a custom pool, e.g., using MPI, create a pool such as
:class:`schwimmbad.MPIPool` and pass it to ``run_sampler``. A user-supplied pool
remains the caller's responsibility to close. Each MPI rank must initialize
Bilby's worker globals before worker ranks begin waiting for work.
Run the script with an MPI launcher, for example ``mpiexec -n 4 python analysis.py``.

.. code-block:: python

	import sys

	import bilby
	from schwimmbad import MPIPool
	from bilby.core.utils.parallel import initialize_global_variables

	pool = MPIPool(use_dill=True)
	initialize_global_variables(
		likelihood=likelihood,
		priors=priors,
		search_parameter_keys=["mass_1", "mass_2"],
		use_ratio=False,
		parameters=priors.sample(),
	)

	if not pool.is_master():
		pool.wait()
		sys.exit(0)

	try:
		result = bilby.run_sampler(
			likelihood=likelihood,
			priors=priors,
			sampler="dynesty",
			pool=pool,
		)
	finally:
		pool.close()

Set ``search_parameter_keys`` to the names of the parameters sampled by the
run, and set ``use_ratio`` to match the requested likelihood evaluation. For
more control, the same initialization pattern applies to any pool-like object
that supports ``map`` and is supplied through the ``pool`` argument.
