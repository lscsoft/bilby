import warnings
from collections import namedtuple

import numpy as np
from dynesty.internal_samplers import InternalSampler, SamplerReturn
from dynesty.utils import SamplerHistoryItem, apply_reflect, get_random_generator

from ...bilby_mcmc.chain import calculate_tau
from ..utils.log import logger

EnsembleSamplerArgument = namedtuple(
    "EnsembleSamplerArgument",
    [
        "u",
        "loglstar",
        "live_points",
        "prior_transform",
        "loglikelihood",
        "rseed",
        "kwargs",
    ],
)
EnsembleAxisSamplerArgument = namedtuple(
    "EnsembleAxisSamplerArgument",
    [
        "u",
        "loglstar",
        "axes",
        "live_points",
        "prior_transform",
        "loglikelihood",
        "rseed",
        "kwargs",
    ],
)


class BaseEnsembleSampler(InternalSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ncdim = kwargs.get("ncdim")
        self.sampler_kwargs["ncdim"] = self.ncdim
        self.sampler_kwargs["proposals"] = kwargs.get("proposals", ["diff"])

    def prepare_sampler(
        self,
        loglstar=None,
        points=None,
        axes=None,
        seeds=None,
        prior_transform=None,
        loglikelihood=None,
        nested_sampler=None,
    ):
        """
        Prepare the list of arguments for sampling.

        Parameters
        ----------
        loglstar : float
            Ln(likelihood) bound.
        points : `~numpy.ndarray` with shape (n, ndim)
            Initial sample points.
        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new points.
        seeds : `~numpy.ndarray` with shape (n,)
            Random number generator seeds.
        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.
        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.
        nested_sampler : `~dynesty.samplers.Sampler`
            The nested sampler object used to sample.

        Returns
        -------
        arglist:
            List of `SamplerArgument` objects containing the parameters
            needed for sampling.
        """
        arg_list = []
        kwargs = self.sampler_kwargs.copy()
        self.nlive = nested_sampler.nlive
        for curp, curaxes, curseed in zip(points, axes, seeds):
            vals = dict(
                u=curp,
                loglstar=loglstar,
                live_points=nested_sampler.live_u,
                prior_transform=prior_transform,
                loglikelihood=loglikelihood,
                rseed=curseed,
                kwargs=kwargs,
            )
            if "volumetric" in kwargs["proposals"]:
                vals["axes"] = curaxes
                curarg = EnsembleAxisSamplerArgument(**vals)
            else:
                curarg = EnsembleSamplerArgument(**vals)
            arg_list.append(curarg)
        return arg_list


class EnsembleWalkSampler(BaseEnsembleSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.walks = max(2, kwargs.get("walks", 25))
        self.sampler_kwargs["walks"] = self.walks
        self.naccept = kwargs.get("naccept", 10)
        self.maxmcmc = kwargs.get("maxmcmc", 5000)

    def tune(self, tuning_info, update=True):
        """
        Update the proposal parameters based on the number of accepted steps
        and MCMC chain length.

        The :code:`walks` parameter to asymptotically approaches the desired number
        of accepted steps. Update :code:`self.scale` for inclusion in the state plot.
        """
        # update walks to match target naccept
        accept_prob = max(0.5, tuning_info["accept"]) / self.sampler_kwargs["walks"]
        delay = max(self.nlive // 10 - 1, 0)
        self.scale = tuning_info["accept"]
        self.walks = (self.walks * delay + self.naccept / accept_prob) / (delay + 1)
        self.sampler_kwargs["walks"] = min(int(np.ceil(self.walks)), self.maxmcmc)

    @staticmethod
    def sample(args):
        """
        Return a new live point proposed by random walking away from an
        existing live point.

        Parameters
        ----------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the initial sample. **This is a copy of an existing
            live point.**

        loglstar : float
            Ln(likelihood) bound.

        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new points. For random walks new positions are
            proposed using the :class:`~dynesty.bounding.Ellipsoid` whose
            shape is defined by axes.

        scale : float
            Value used to scale the provided axes.

        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.

        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.

        kwargs : dict
            A dictionary of additional method-specific parameters.

        Returns
        -------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point within the unit cube.

        v : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point in the target parameter space.

        logl : float
            Ln(likelihood) of the final proposed point.

        nc : int
            Number of function calls used to generate the sample.

        sampling_info : dict
            Collection of ancillary quantities used to tune :data:`scale`.

        """
        current_u = args.u
        naccept = 0
        ncall = 0

        periodic = args.kwargs["periodic"]
        reflective = args.kwargs["reflective"]
        boundary_kwargs = dict(periodic=periodic, reflective=reflective)

        proposals, common_kwargs, proposal_kwargs = _get_proposal_kwargs(args)
        walks = len(proposals)
        evaluation_history = list()

        for prop in proposals:
            u_prop = proposal_funcs[prop](
                u=current_u, **common_kwargs, **proposal_kwargs[prop]
            )
            u_prop = apply_boundaries_(u_prop=u_prop, **boundary_kwargs)
            if u_prop is None:
                continue

            v_prop = args.prior_transform(u_prop)
            logl_prop = args.loglikelihood(v_prop)
            evaluation_history.append(
                SamplerHistoryItem(u=v_prop, v=u_prop, logl=logl_prop)
            )
            ncall += 1

            if logl_prop > args.loglstar:
                current_u = u_prop
                current_v = v_prop
                logl = logl_prop
                naccept += 1

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

        sampling_info = {
            "accept": naccept,
            "reject": walks - naccept,
        }

        return SamplerReturn(
            u=current_u,
            v=current_v,
            logl=logl,
            tuning_info=sampling_info,
            ncalls=ncall,
            proposal_stats=sampling_info,
            evaluation_history=evaluation_history,
        )

        # return current_u, current_v, logl, ncall, sampling_info


class ACTTrackingEnsembleWalk(BaseEnsembleSampler):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.act = 1
        self.thin = kwargs.get("nact", 2)
        self.maxmcmc = kwargs.get("maxmcmc", 5000) * 50
        self.sampler_kwargs["rebuild"] = True
        self.sampler_kwargs["thin"] = self.thin
        self.sampler_kwargs["act"] = self.act
        self.sampler_kwargs["maxmcmc"] = self.maxmcmc
        # reset the cache at instantiation to avoid contamination from
        # previous analyses
        self.__class__._cache = list()

    def prepare_sampler(
        self,
        loglstar=None,
        points=None,
        axes=None,
        seeds=None,
        prior_transform=None,
        loglikelihood=None,
        nested_sampler=None,
    ):
        """
        Prepare the list of arguments for sampling.

        Parameters
        ----------
        loglstar : float
            Ln(likelihood) bound.
        points : `~numpy.ndarray` with shape (n, ndim)
            Initial sample points.
        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new points.
        seeds : `~numpy.ndarray` with shape (n,)
            Random number generator seeds.
        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.
        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.
        nested_sampler : `~dynesty.samplers.Sampler`
            The nested sampler object used to sample.

        Returns
        -------
        arglist:
            List of `SamplerArgument` objects containing the parameters
            needed for sampling.
        """
        arg_list = super().prepare_sampler(
            loglstar=loglstar,
            points=points,
            axes=axes,
            seeds=seeds,
            prior_transform=prior_transform,
            loglikelihood=loglikelihood,
            nested_sampler=nested_sampler,
        )
        self.sampler_kwargs["rebuild"] = False
        return arg_list

    def tune(self, tuning_info, update=True):
        """
        Update the proposal parameters based on the number of accepted steps
        and MCMC chain length.

        The :code:`walks` parameter to asymptotically approach the
        desired number of accepted steps.
        """
        if tuning_info.get("remaining", 0) == 0:
            self.sampler_kwargs["rebuild"] = True
        self.scale = tuning_info["accept"]
        self.sampler_kwargs["act"] = tuning_info["act"]

    @staticmethod
    def sample(args):
        cache = ACTTrackingEnsembleWalk._cache
        if args.kwargs.get("rebuild", False):
            logger.debug(f"Force rebuilding cache with {len(cache)}.")
            cache.clear()
        if len(cache) == 0:
            ACTTrackingEnsembleWalk.build_cache(args)

        while len(cache) > 0 and cache[0][2] < args.loglstar:
            state = cache.pop(0)

        if len(cache) == 0:
            current_u, current_v, logl, ncall, blob, evaluation_history = state
        else:
            current_u, current_v, logl, ncall, blob, evaluation_history = cache.pop(0)

        blob["remaining"] = len(cache)
        logger.debug(f"Returning point from cache with {len(cache)} remaining")
        return SamplerReturn(
            u=current_u,
            v=current_v,
            logl=logl,
            tuning_info=blob,
            ncalls=ncall,
            proposal_stats=blob,
            evaluation_history=evaluation_history,
        )

    @staticmethod
    def build_cache(args):
        # Bounds
        periodic = args.kwargs.get("periodic", None)
        reflective = args.kwargs.get("reflective", None)
        boundary_kwargs = dict(periodic=periodic, reflective=reflective)

        proposals, common_kwargs, proposal_kwargs = _get_proposal_kwargs(args)

        # Setup
        current_u = args.u
        old_act = args.kwargs.get("act", 100)
        check_interval = ACTTrackingEnsembleWalk.integer_act(old_act)
        target_nact = 50
        next_check = check_interval
        n_checks = 0
        maxmcmc = args.kwargs.get("maxmcmc", 5000)
        evaluation_history = list()

        # Initialize internal variables
        current_v = args.prior_transform(np.array(current_u))
        logl = args.loglikelihood(np.array(current_v))
        evaluation_history.append(
            SamplerHistoryItem(u=current_u, v=current_v, logl=logl)
        )
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
        while iteration < min(target_nact * act, maxmcmc):
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
                evaluation_history.append(
                    SamplerHistoryItem(u=v_prop, v=u_prop, logl=logl_prop)
                )
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
                act = ACTTrackingEnsembleWalk._calculate_act(
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
        act = ACTTrackingEnsembleWalk._calculate_act(
            accept=accept,
            iteration=iteration,
            samples=np.array(u_list),
            most_failures=most_failures,
        )
        reject += nfail
        blob = {"accept": accept, "reject": reject, "act": act}
        iact = ACTTrackingEnsembleWalk.integer_act(act)
        thin = args.kwargs.get("thin", 2) * iact

        cache = ACTTrackingEnsembleWalk._cache

        if accept == 0:
            logger.warning(
                "Unable to find a new point using walk: returning a random point"
            )
            u = common_kwargs["rstate"].uniform(size=len(current_u))
            v = args.prior_transform(u)
            logl = args.loglikelihood(v)
            evaluation_history = [
                SamplerHistoryItem(u=current_u, v=current_v, logl=logl)
            ]
            cache.append((u, v, logl, ncall, blob, evaluation_history))
        elif not np.isfinite(act):
            logger.warning(
                "Unable to find a new point using walk: try increasing maxmcmc"
            )
            cache.append((current_u, current_v, logl, ncall, blob, evaluation_history))
        elif (thin == -1) or (len(u_list) <= thin):
            cache.append((current_u, current_v, logl, ncall, blob, evaluation_history))
        else:
            u_list = u_list[thin::thin]
            v_list = v_list[thin::thin]
            logl_list = logl_list[thin::thin]
            evaluation_history_list = (
                evaluation_history[thin * ii : thin * (ii + 1)]
                for ii in range(len(u_list))
            )
            n_found = len(u_list)
            accept = max(accept // n_found, 1)
            reject //= n_found
            nfail //= n_found
            ncall_list = [ncall // n_found] * n_found
            blob_list = [
                dict(accept=accept, reject=reject, fail=nfail, act=act)
            ] * n_found
            cache.extend(
                zip(
                    u_list,
                    v_list,
                    logl_list,
                    ncall_list,
                    blob_list,
                    evaluation_history_list,
                )
            )
            logger.debug(
                f"act: {act:.2f}, max failures: {most_failures}, thin: {thin}, "
                f"iteration: {iteration}, n_found: {n_found}"
            )
        logger.debug(
            f"Finished building cache with length {len(cache)} after "
            f"{iteration} iterations with {ncall} likelihood calls and ACT={act:.2f}"
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

    @staticmethod
    def integer_act(act):
        if np.isinf(act):
            return act
        else:
            return int(np.ceil(act))


class AcceptanceTrackingRWalk(EnsembleWalkSampler):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nact = kwargs.get("nact", 40)
        self.sampler_kwargs["nact"] = self.nact
        self.sampler_kwargs["maxmcmc"] = self.maxmcmc

    def tune(self, tuning_info, update=True):
        """
        The tuning all happens inside the MCMC for this proposal
        so all we do is extract the number of accepted steps for plotting.
        """
        self.scale = tuning_info["accept"]

    @staticmethod
    def sample(args):
        rstate = get_random_generator(args.rseed)

        periodic = args.kwargs.get("periodic", None)
        reflective = args.kwargs.get("reflective", None)
        boundary_kwargs = dict(periodic=periodic, reflective=reflective)

        current_u = args.u
        nlive = args.kwargs.get("nlive", args.kwargs.get("walks", 100))

        proposals, common_kwargs, proposal_kwargs = _get_proposal_kwargs(args)

        accept = 0
        reject = 0
        nfail = 0
        act = np.inf
        nact = args.kwargs.get("nact", 40)
        maxmcmc = args.kwargs.get("maxmcmc", 5000)
        evaluation_history = list()

        iteration = 0
        while iteration < nact * act:
            iteration += 1

            prop = proposals[iteration % len(proposals)]
            u_prop = proposal_funcs[prop](
                current_u, **common_kwargs, **proposal_kwargs[prop]
            )
            u_prop = apply_boundaries_(u_prop, **boundary_kwargs)

            if u_prop is None:
                nfail += 1
                continue

            # Check proposed point.
            v_prop = args.prior_transform(np.array(u_prop))
            logl_prop = args.loglikelihood(np.array(v_prop))
            evaluation_history.append(
                SamplerHistoryItem(u=v_prop, v=u_prop, logl=logl_prop)
            )
            if logl_prop > args.loglstar:
                current_u = u_prop
                current_v = v_prop
                logl = logl_prop
                accept += 1
            else:
                reject += 1

            # If we've taken the minimum number of steps, calculate the ACT
            if iteration > nact:
                act = AcceptanceTrackingRWalk.estimate_nmcmc(
                    accept_ratio=accept / (accept + reject + nfail),
                    safety=1,
                    tau=nlive,
                    maxmcmc=maxmcmc,
                    old_act=AcceptanceTrackingRWalk.old_act,
                )

            # If we've taken too many likelihood evaluations then break
            if accept + reject > maxmcmc:
                warnings.warn(
                    f"Hit maximum number of walks {maxmcmc} with accept={accept},"
                    f" reject={reject}, and nfail={nfail} try increasing maxmcmc"
                )
                break

        if not (np.isfinite(act) and accept > 0):
            logger.debug(
                "Unable to find a new point using walk: returning a random point"
            )
            current_u = rstate.uniform(size=len(current_u))
            current_v = args.prior_transform(current_u)
            logl = args.loglikelihood(current_v)

        sampling_info = {"accept": accept, "reject": reject + nfail}
        AcceptanceTrackingRWalk.old_act = act

        ncall = accept + reject
        return SamplerReturn(
            u=current_u,
            v=current_v,
            logl=logl,
            tuning_info=sampling_info,
            ncalls=ncall,
            proposal_stats=sampling_info,
            evaluation_history=evaluation_history,
        )

    @staticmethod
    def estimate_nmcmc(accept_ratio, safety=5, tau=None, maxmcmc=5000, old_act=None):
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
            tau = maxmcmc / safety

        if accept_ratio == 0.0:
            if old_act is None:
                Nmcmc_exact = np.inf
            else:
                Nmcmc_exact = (1 + 1 / tau) * old_act
        else:
            estimated_act = 2 / accept_ratio - 1
            Nmcmc_exact = safety * estimated_act
            if old_act is not None:
                Nmcmc_exact = (1 - 1 / tau) * old_act + Nmcmc_exact / tau
        Nmcmc_exact = float(min(Nmcmc_exact, maxmcmc))
        return max(safety, Nmcmc_exact)


def _get_proposal_kwargs(args):
    """
    Resolve the proposal cycle from the provided keyword arguments.

    The steps involved are:

    - extract the requested proposal types from the kwargs passed through from dynesty.
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

    proposals = args.kwargs.get("proposals", None)
    if proposals is None:
        proposals = ["diff"]

    n_cluster = args.kwargs.get("ncdim", None)
    if n_cluster is None:
        if hasattr(args, "live_points"):
            n_cluster = args.live_points.shape[1]
        elif hasattr(args, "axes"):
            n_cluster = args.axes.shape[0]

    if "diff" in proposals:
        live = args.live_points
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


def propose_differential_evolution(
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
    rstate: numpy.random.Generator
        The numpy generator instance used for random number generation.
        Consider using built in `bilby.core.utils.random.rng`.
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
    rstate: numpy.random.Generator
        The numpy generator instance used for random number generation.
        Consider using built in `bilby.core.utils.random.rng`.

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


proposal_funcs = dict(
    diff=propose_differential_evolution, volumetric=propose_volumetric
)
