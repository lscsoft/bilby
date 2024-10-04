import warnings

import numpy as np
from dynesty.nestedsamplers import MultiEllipsoidSampler, UnitCubeSampler
from dynesty.utils import apply_reflect, get_random_generator

from ...bilby_mcmc.chain import calculate_tau
from ..utils.log import logger
from .base_sampler import _SamplingContainer


class LivePointSampler(UnitCubeSampler):
    """
    Modified version of dynesty UnitCubeSampler that adapts the MCMC
    length in addition to the proposal scale, this corresponds to
    :code:`bound=live`.

    In order to support live-point based proposals, e.g., differential
    evolution (:code:`diff`), the live points are added to the
    :code:`kwargs` passed to the evolve method.

    Note that this does not perform ellipsoid clustering as with the
    :code:`bound=multi` option, if ellipsoid-based proposals are used, e.g.,
    :code:`volumetric`, consider using the
    :code:`MultiEllipsoidLivePointSampler` (:code:`sample=live-multi`).
    """

    rebuild = False

    def update_user(self, blob, update=True):
        """
        Update the proposal parameters based on the number of accepted steps
        and MCMC chain length.

        There are a number of logical checks performed:
        - if the ACT tracking rwalk method is being used and any parallel
          process has an empty cache, set the :code:`rebuild` flag to force
          the cache to rebuild at the next call. This improves the efficiency
          when using parallelisation.
        - update the :code:`walks` parameter to asymptotically approach the
          desired number of accepted steps for the :code:`FixedRWalk` proposal.
        - update the ellipsoid scale if the ellipsoid proposals are being used.
        """
        # do we need to trigger rebuilding the cache
        if blob.get("remaining", 0) == 1:
            self.rebuild = True
        if update:
            self.kwargs["rebuild"] = self.rebuild
            self.rebuild = False

        # update walks to match target naccept
        accept_prob = max(0.5, blob["accept"]) / self.kwargs["walks"]
        delay = max(self.nlive // 10 - 1, 0)
        n_target = getattr(_SamplingContainer, "naccept", 60)
        self.walks = (self.walks * delay + n_target / accept_prob) / (delay + 1)
        self.kwargs["walks"] = min(int(np.ceil(self.walks)), _SamplingContainer.maxmcmc)

        self.scale = blob["accept"]

    update_rwalk = update_user

    def propose_live(self, *args):
        """
        We need to make sure the live points are passed to the proposal
        function if we are using live point-based proposals.
        """
        self.kwargs["nlive"] = self.nlive
        self.kwargs["live"] = self.live_u
        i = self.rstate.integers(self.nlive)
        u = self.live_u[i, :]
        return u, np.identity(self.ncdim)


class MultiEllipsoidLivePointSampler(MultiEllipsoidSampler):
    """
    Modified version of dynesty MultiEllipsoidSampler that adapts the MCMC
    length in addition to the proposal scale, this corresponds to
    :code:`bound=live-multi`.

    Additionally, in order to support live point-based proposals, e.g.,
    differential evolution (:code:`diff`), the live points are added to the
    :code:`kwargs` passed to the evolve method.

    When just using the :code:`diff` proposal method, consider using the
    :code:`LivePointSampler` (:code:`sample=live`).
    """

    rebuild = False

    def update_user(self, blob, update=True):
        LivePointSampler.update_user(self, blob=blob, update=update)
        super(MultiEllipsoidLivePointSampler, self).update_rwalk(
            blob=blob, update=update
        )

    update_rwalk = update_user

    def propose_live(self, *args):
        """
        We need to make sure the live points are passed to the proposal
        function if we are using ensemble proposals.
        """
        self.kwargs["nlive"] = self.nlive
        self.kwargs["live"] = self.live_u
        return super(MultiEllipsoidLivePointSampler, self).propose_live(*args)


class FixedRWalk:
    """
    Run the MCMC walk for a fixed length. This is nearly equivalent to
    :code:`bilby.sampling.sample_rwalk` except that different proposal
    distributions can be used.
    """

    def __call__(self, args):
        current_u = args.u
        naccept = 0
        ncall = 0

        periodic = args.kwargs["periodic"]
        reflective = args.kwargs["reflective"]
        boundary_kwargs = dict(periodic=periodic, reflective=reflective)

        proposals, common_kwargs, proposal_kwargs = _get_proposal_kwargs(args)
        walks = len(proposals)

        accepted = list()

        for prop in proposals:
            u_prop = proposal_funcs[prop](
                u=current_u, **common_kwargs, **proposal_kwargs[prop]
            )
            u_prop = apply_boundaries_(u_prop=u_prop, **boundary_kwargs)
            if u_prop is None:
                accepted.append(0)
                continue

            v_prop = args.prior_transform(u_prop)
            logl_prop = args.loglikelihood(v_prop)
            ncall += 1

            if logl_prop > args.loglstar:
                current_u = u_prop
                current_v = v_prop
                logl = logl_prop
                naccept += 1
                accepted.append(1)
            else:
                accepted.append(0)

        if naccept == 0:
            logger.debug(
                "Unable to find a new point using walk: returning a random point. "
                "If this warning occurs often, increase naccept."
            )
            # Technically we can find out the likelihood value
            # stored somewhere
            # But I'm currently recomputing it
            current_u = common_kwargs["rstate"].uniform(0, 1, len(current_u))
            current_v = args.prior_transform(current_u)
            logl = args.loglikelihood(current_v)

        blob = {
            "accept": naccept,
            "reject": walks - naccept,
            "scale": args.scale,
        }

        return current_u, current_v, logl, ncall, blob


class ACTTrackingRWalk:
    """
    Run the MCMC sampler for many iterations in order to reliably estimate
    the autocorrelation time.

    This builds a cache of :code:`nact / thin` points consistent with the
    likelihood constraint.
    While this approach is the most robust, it is not well optimized for
    parallelised sampling as the length of the MCMC will be different for each
    parallel process.
    """

    # the _cache is a class level variable to avoid being forgotten at every
    # iteration when using multiprocessing
    _cache = list()

    def __init__(self):
        self.act = 1
        self.thin = getattr(_SamplingContainer, "nact", 2)
        self.maxmcmc = getattr(_SamplingContainer, "maxmcmc", 5000) * 50

    def __call__(self, args):
        self.args = args
        if args.kwargs.get("rebuild", False):
            logger.debug("Force rebuilding cache")
            self.build_cache()
        while self.cache[0][2] < args.loglstar:
            self.cache.pop(0)
        current_u, current_v, logl, ncall, blob = self.cache.pop(0)
        blob["remaining"] = len(self.cache)
        return current_u, current_v, logl, ncall, blob

    @property
    def cache(self):
        if len(self._cache) == 0:
            self.build_cache()
        else:
            logger.debug(f"Not rebuilding cache, remaining size {len(self._cache)}")
        return self._cache

    def build_cache(self):
        args = self.args
        # Bounds
        periodic = args.kwargs.get("periodic", None)
        reflective = args.kwargs.get("reflective", None)
        boundary_kwargs = dict(periodic=periodic, reflective=reflective)

        proposals, common_kwargs, proposal_kwargs = _get_proposal_kwargs(args)

        # Setup
        current_u = args.u
        check_interval = self.integer_act
        target_nact = 50
        next_check = check_interval
        n_checks = 0

        # Initialize internal variables
        current_v = args.prior_transform(np.array(current_u))
        logl = args.loglikelihood(np.array(current_v))
        accept = 0
        reject = 0
        nfail = 0
        ncall = 0
        act = np.inf
        u_list = list()
        v_list = list()
        logl_list = list()
        most_failures = 0
        current_failures = 0

        iteration = 0
        while iteration < min(target_nact * act, self.maxmcmc):
            iteration += 1

            prop = proposals[iteration % len(proposals)]
            u_prop = proposal_funcs[prop](
                u=current_u, **common_kwargs, **proposal_kwargs[prop]
            )
            u_prop = apply_boundaries_(u_prop=u_prop, **boundary_kwargs)
            success = False
            if u_prop is not None:
                v_prop = args.prior_transform(np.array(u_prop))
                logl_prop = args.loglikelihood(np.array(v_prop))
                ncall += 1
                if logl_prop > args.loglstar:
                    success = True
                    current_u = u_prop
                    current_v = v_prop
                    logl = logl_prop

            u_list.append(current_u)
            v_list.append(current_v)
            logl_list.append(logl)

            if success:
                accept += 1
                most_failures = max(most_failures, current_failures)
                current_failures = 0
            else:
                nfail += 1
                current_failures += 1

            # If we've taken the minimum number of steps, calculate the ACT
            if iteration > next_check and accept > target_nact:
                n_checks += 1
                most_failures = max(most_failures, current_failures)
                act = self._calculate_act(
                    accept=accept,
                    iteration=iteration,
                    samples=np.array(u_list),
                    most_failures=most_failures,
                )
                to_next_update = (act * target_nact - iteration) // 2
                to_next_update = max(to_next_update, iteration // 100)
                next_check += to_next_update
                logger.debug(
                    f"it: {iteration}, accept: {accept}, reject: {reject}, "
                    f"fail: {nfail}, act: {act:.2f}, nact: {iteration / act:.2f} "
                )
            elif iteration > next_check:
                next_check += check_interval

        most_failures = max(most_failures, current_failures)
        self.act = self._calculate_act(
            accept=accept,
            iteration=iteration,
            samples=np.array(u_list),
            most_failures=most_failures,
        )
        reject += nfail
        blob = {"accept": accept, "reject": reject, "scale": args.scale}
        iact = self.integer_act
        thin = self.thin * iact

        if accept == 0:
            logger.warning(
                "Unable to find a new point using walk: returning a random point"
            )
            u = common_kwargs["rstate"].uniform(size=len(current_u))
            v = args.prior_transform(u)
            logl = args.loglikelihood(v)
            self._cache.append((u, v, logl, ncall, blob))
        elif not np.isfinite(act):
            logger.warning(
                "Unable to find a new point using walk: try increasing maxmcmc"
            )
            self._cache.append((current_u, current_v, logl, ncall, blob))
        elif (self.thin == -1) or (len(u_list) <= thin):
            self._cache.append((current_u, current_v, logl, ncall, blob))
        else:
            u_list = u_list[thin::thin]
            v_list = v_list[thin::thin]
            logl_list = logl_list[thin::thin]
            n_found = len(u_list)
            accept = max(accept // n_found, 1)
            reject //= n_found
            nfail //= n_found
            ncall_list = [ncall // n_found] * n_found
            blob_list = [
                dict(accept=accept, reject=reject, fail=nfail, scale=args.scale)
            ] * n_found
            self._cache.extend(zip(u_list, v_list, logl_list, ncall_list, blob_list))
            logger.debug(
                f"act: {self.act:.2f}, max failures: {most_failures}, thin: {thin}, "
                f"iteration: {iteration}, n_found: {n_found}"
            )
        logger.debug(
            f"Finished building cache with length {len(self._cache)} after "
            f"{iteration} iterations with {ncall} likelihood calls and ACT={self.act:.2f}"
        )

    @staticmethod
    def _calculate_act(accept, iteration, samples, most_failures):
        """
        Take the maximum of three ACT estimates, leading to a conservative estimate:

        - a full ACT estimate as done in :code:`bilby_mcmc`. This is almost always the
          longest estimator, and is the most computationally expensive. The other
          methods mostly catch cases where the estimated ACT is very small.
        - the naive ACT used for the acceptance tracking walk.
        - the most failed proposals between any pair of accepted steps. This is a strong
          lower bound, because we know that if we thin by less than this, there will be
          duplicate points.
        """
        if accept > 0:
            naive_act = 2 / accept * iteration - 1
        else:
            return np.inf
        return max(calculate_tau(samples), naive_act, most_failures)

    @property
    def integer_act(self):
        if np.isinf(self.act):
            return self.act
        else:
            return int(np.ceil(self.act))


class AcceptanceTrackingRWalk:
    """
    This is a modified version of dynesty.sampling.sample_rwalk that runs the
    MCMC random walk for a user-specified number of a crude approximation to
    the autocorrelation time.

    This is the proposal method used by default for :code:`Bilby<2` and
    corresponds to specifying :code:`sample="rwalk"`
    """

    # to retain state between calls to pool.Map, this needs to be a class
    # level attribute
    old_act = None

    def __init__(self, old_act=None):
        self.maxmcmc = getattr(_SamplingContainer, "maxmcmc", 5000)
        self.nact = getattr(_SamplingContainer, "nact", 40)

    def __call__(self, args):
        rstate = get_random_generator(args.rseed)

        periodic = args.kwargs.get("periodic", None)
        reflective = args.kwargs.get("reflective", None)
        boundary_kwargs = dict(periodic=periodic, reflective=reflective)

        u = args.u
        nlive = args.kwargs.get("nlive", args.kwargs.get("walks", 100))

        proposals, common_kwargs, proposal_kwargs = _get_proposal_kwargs(args)

        accept = 0
        reject = 0
        nfail = 0
        act = np.inf

        iteration = 0
        while iteration < self.nact * act:
            iteration += 1

            prop = proposals[iteration % len(proposals)]
            u_prop = proposal_funcs[prop](u, **common_kwargs, **proposal_kwargs[prop])
            u_prop = apply_boundaries_(u_prop, **boundary_kwargs)

            if u_prop is None:
                nfail += 1
                continue

            # Check proposed point.
            v_prop = args.prior_transform(np.array(u_prop))
            logl_prop = args.loglikelihood(np.array(v_prop))
            if logl_prop > args.loglstar:
                u = u_prop
                v = v_prop
                logl = logl_prop
                accept += 1
            else:
                reject += 1

            # If we've taken the minimum number of steps, calculate the ACT
            if iteration > self.nact:
                act = self.estimate_nmcmc(
                    accept_ratio=accept / (accept + reject + nfail),
                    safety=1,
                    tau=nlive,
                )

            # If we've taken too many likelihood evaluations then break
            if accept + reject > self.maxmcmc:
                warnings.warn(
                    f"Hit maximum number of walks {self.maxmcmc} with accept={accept},"
                    f" reject={reject}, and nfail={nfail} try increasing maxmcmc"
                )
                break

        if not (np.isfinite(act) and accept > 0):
            logger.debug(
                "Unable to find a new point using walk: returning a random point"
            )
            u = rstate.uniform(size=len(u))
            v = args.prior_transform(u)
            logl = args.loglikelihood(v)

        blob = {"accept": accept, "reject": reject + nfail, "scale": args.scale}
        AcceptanceTrackingRWalk.old_act = act

        ncall = accept + reject
        return u, v, logl, ncall, blob

    def estimate_nmcmc(self, accept_ratio, safety=5, tau=None):
        """Estimate autocorrelation length of chain using acceptance fraction

        Using ACL = (2/acc) - 1 multiplied by a safety margin. Code adapted from:

        - https://github.com/farr/Ensemble.jl
        - https://github.com/johnveitch/cpnest/blob/master/cpnest/sampler.py

        Parameters
        ==========
        accept_ratio: float [0, 1]
            Ratio of the number of accepted points to the total number of points
        old_act: int
            The ACT of the last iteration
        maxmcmc: int
            The maximum length of the MCMC chain to use
        safety: int
            A safety factor applied in the calculation
        tau: int (optional)
            The ACT, if given, otherwise estimated.

        Notes
        =====
        This method does not compute a reliable estimate of the autocorrelation
        length for our proposal distributions.
        """
        if tau is None:
            tau = self.maxmcmc / safety

        if accept_ratio == 0.0:
            if self.old_act is None:
                Nmcmc_exact = np.inf
            else:
                Nmcmc_exact = (1 + 1 / tau) * self.old_act
        else:
            estimated_act = 2 / accept_ratio - 1
            Nmcmc_exact = safety * estimated_act
            if self.old_act is not None:
                Nmcmc_exact = (1 - 1 / tau) * self.old_act + Nmcmc_exact / tau
        Nmcmc_exact = float(min(Nmcmc_exact, self.maxmcmc))
        return max(safety, Nmcmc_exact)


def _get_proposal_kwargs(args):
    """
    Resolve the proposal cycle from the provided keyword arguments.

    The steps involved are:

    - extract the requested proposal types from the :code:`_SamplingContainer`.
      If none are specified, only differential evolution will be used.
    - differential evolution requires the live points to be passed. If they are
      not present, raise an error.
    - if a dictionary, e.g., :code:`dict(diff=5, volumetric=1)` is specified,
      the keys will be used weighted by the values, e.g., 5:1 differential
      evolution to volumetric.
    - each proposal needs different keyword arguments, see the specific functions
      for what requires which.


    Parameters
    ==========
    args: dynesty.sampler.SamplerArgument
        Object that carries around various pieces of information about the
        analysis.
    """
    rstate = get_random_generator(args.rseed)
    walks = args.kwargs.get("walks", 100)
    current_u = args.u
    n_cluster = args.axes.shape[0]

    proposals = getattr(_SamplingContainer, "proposals", None)
    if proposals is None:
        proposals = ["diff"]
    if "diff" in proposals:
        live = args.kwargs.get("live", None)
        if live is None:
            raise ValueError(
                "Live points not passed for differential evolution, specify "
                "bound='live' to use differential evolution proposals."
            )
        live = np.unique(live, axis=0)
        matches = np.where(np.equal(current_u, live).all(axis=1))[0]
        np.delete(live, matches, 0)

    if isinstance(proposals, (list, set, tuple)):
        proposals = rstate.choice(proposals, int(walks))
    elif isinstance(proposals, dict):
        props, weights = zip(*proposals.items())
        weights = np.array(weights) / sum(weights)
        proposals = rstate.choice(list(props), int(walks), p=weights)

    common_kwargs = dict(
        n=len(current_u),
        n_cluster=n_cluster,
        rstate=rstate,
    )
    proposal_kwargs = dict()
    if "diff" in proposals:
        proposal_kwargs["diff"] = dict(
            live=live[:, :n_cluster],
            mix=0.5,
            scale=2.38 / (2 * n_cluster) ** 0.5,
        )
    if "volumetric" in proposals:
        proposal_kwargs["volumetric"] = dict(
            axes=args.axes,
            scale=args.scale,
        )
    return proposals, common_kwargs, proposal_kwargs


def propose_differetial_evolution(
    u,
    live,
    n,
    n_cluster,
    rstate,
    mix=0.5,
    scale=1,
):
    r"""
    Propose a new point using ensemble differential evolution
    (`ter Braak + (2006) <https://doi.org/10.1007/s11222-006-8769-1>`_).

    .. math::

        u_{\rm prop} = u + \gamma (v_{a} - v_{b})

    We consider two choices for :math:`\gamma`: weighted by :code:`mix`.

    - :math:`\gamma = 1`: this is a mode-hopping mode for efficiently
      exploring multi-modal spaces
    - :math:`\gamma \sim \Gamma\left(\gamma; k=4, \theta=\frac{\kappa}{4}\right)`

    Here :math:`\kappa = 2.38 / \sqrt{2 n}` unless specified by the user and
    we scale by a random draw from a Gamma distribution. The specific
    distribution was chosen somewhat arbitrarily to have mean and mode close to
    :math:`\kappa` and give good acceptance and autocorrelation times on a subset
    of problems.

    Parameters
    ----------
    u: np.ndarray
        The current point.
    live: np.ndarray
        The ensemble of live points to select :math:`v` from.
    n: int
        The number of dimensions being explored
    n_cluster: int
        The number of dimensions to run the differential evolution over, the
        first :code:`n_cluster` dimensions are used. The rest are randomly
        sampled from the prior.
    rstate: np.random.RandomState
        The random state to use to generate random numbers.
    mix: float
        The fraction of proposed points that should follow the specified scale
        rather than mode hopping. :code:`default=0.5`
    scale: float
        The amount to scale the difference vector by.
        :code:`default = 2.38 / (2 * n_cluster)**0.5)`

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    """
    delta = np.diff(rstate.choice(live, 2, replace=False), axis=0)[0]
    if rstate.uniform(0, 1) < mix:
        if scale is None:
            scale = 2.38 / (2 * n_cluster) ** 0.5
        scale *= rstate.gamma(4, 0.25)
    else:
        scale = 1
    u_prop = u.copy()
    u_prop[:n_cluster] += delta * scale
    u_prop[n_cluster:] = rstate.uniform(0, 1, n - n_cluster)
    return u_prop


def propose_volumetric(
    u,
    axes,
    scale,
    n,
    n_cluster,
    rstate,
):
    """
    Propose a new point using the default :code:`dynesty` proposal.

    The new difference vector is scaled by a vector isotropically drawn
    from an ellipsoid centered on zero with covariance given by the
    provided axis. Note that the magnitude of this proposal is heavily
    skewed to the size of the ellipsoid.

    Parameters
    ----------
    u: np.ndarray
        The current point.
    n: int
        The number of dimensions being explored.
    scale: float
        The amount to scale the proposed point by.
    n_cluster: int
        The number of dimensions to run the differential evolution over, the
        first :code:`n_cluster` dimensions are used. The rest are randomly
        sampled from the prior.
    rstate: np.random.RandomState
        The random state to use to generate random numbers.

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    """
    # Propose a direction on the unit n-sphere.
    drhat = rstate.normal(0, 1, n_cluster)
    drhat /= np.linalg.norm(drhat)

    # Scale based on dimensionality.
    dr = drhat * rstate.uniform(0, 1) ** (1.0 / n_cluster)

    # Transform to proposal distribution.
    delta = scale * np.dot(axes, dr)
    u_prop = u.copy()
    u_prop[:n_cluster] += delta
    u_prop[n_cluster:] = rstate.uniform(0, 1, n - n_cluster)
    return u_prop


def apply_boundaries_(u_prop, periodic, reflective):
    """
    Apply the periodic and reflective boundaries and test if we are inside the
    unit cube.

    Parameters
    ----------
    u_prop: np.ndarray
        The proposed point in the unit hypercube space.
    periodic: np.ndarray
        Indices of the parameters with periodic boundaries.
    reflective: np.ndarray
        Indices of the parameters with reflective boundaries.

    Returns
    =======
    [np.ndarray, None]:
        Either the remapped proposed point, or None if the proposed point
        lies outside the unit cube.
    """
    # Wrap periodic parameters
    if periodic is not None:
        u_prop[periodic] = np.mod(u_prop[periodic], 1)

    # Reflect
    if reflective is not None:
        u_prop[reflective] = apply_reflect(u_prop[reflective])

    if u_prop.min() < 0 or u_prop.max() > 1:
        return None
    else:
        return u_prop


proposal_funcs = dict(diff=propose_differetial_evolution, volumetric=propose_volumetric)
