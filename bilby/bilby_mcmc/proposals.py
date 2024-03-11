import importlib
import time
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

from ..core.fisher import FisherMatrixPosteriorEstimator
from ..core.prior import PriorDict
from ..core.sampler.base_sampler import SamplerError
from ..core.utils import logger, random, reflect
from ..gw.source import PARAMETER_SETS


class ProposalCycle(object):
    def __init__(self, proposal_list):
        self.proposal_list = proposal_list
        self.weights = [prop.weight for prop in self.proposal_list]
        self.normalized_weights = [w / sum(self.weights) for w in self.weights]
        self.weighted_proposal_list = [
            random.rng.choice(self.proposal_list, p=self.normalized_weights)
            for _ in range(10 * int(1 / min(self.normalized_weights)))
        ]
        self.nproposals = len(self.weighted_proposal_list)
        self._position = 0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = np.mod(position, self.nproposals)

    def get_proposal(self):
        prop = self.weighted_proposal_list[self._position]
        self.position += 1
        return prop

    def __str__(self):
        string = "ProposalCycle:\n"
        for prop in self.proposal_list:
            string += f"  {prop}\n"
        return string


class BaseProposal(object):
    _accepted = 0
    _rejected = 0
    __metaclass__ = ABCMeta

    def __init__(self, priors, weight=1, subset=None):
        self._str_attrs = ["acceptance_ratio", "n"]

        self.parameters = priors.non_fixed_keys
        self.weight = weight
        self.subset = subset

        # Restrict to a subset
        if self.subset is not None:
            self.parameters = [p for p in self.parameters if p in subset]
            self._str_attrs.append("parameters")

        if len(self.parameters) == 0:
            raise ValueError("Proposal requested with zero parameters")

        self.ndim = len(self.parameters)

        self.prior_boundary_dict = {key: priors[key].boundary for key in priors}
        self.prior_minimum_dict = {key: np.max(priors[key].minimum) for key in priors}
        self.prior_maximum_dict = {key: np.min(priors[key].maximum) for key in priors}
        self.prior_width_dict = {key: np.max(priors[key].width) for key in priors}

    @property
    def accepted(self):
        return self._accepted

    @accepted.setter
    def accepted(self, accepted):
        self._accepted = accepted

    @property
    def rejected(self):
        return self._rejected

    @rejected.setter
    def rejected(self, rejected):
        self._rejected = rejected

    @property
    def acceptance_ratio(self):
        if self.n == 0:
            return np.nan
        else:
            return self.accepted / self.n

    @property
    def n(self):
        return self.accepted + self.rejected

    def __str__(self):
        msg = [f"{type(self).__name__}("]
        for attr in self._str_attrs:
            val = getattr(self, attr, "N/A")
            if isinstance(val, (float, int)):
                val = f"{val:1.2g}"
            msg.append(f"{attr}:{val},")
        return "".join(msg) + ")"

    def apply_boundaries(self, point):
        for key in self.parameters:
            boundary = self.prior_boundary_dict[key]
            if boundary is None:
                continue
            elif boundary == "periodic":
                point[key] = self.apply_periodic_boundary(key, point[key])
            elif boundary == "reflective":
                point[key] = self.apply_reflective_boundary(key, point[key])
            else:
                raise SamplerError(f"Boundary {boundary} not implemented")
        return point

    def apply_periodic_boundary(self, key, val):
        minimum = self.prior_minimum_dict[key]
        width = self.prior_width_dict[key]
        return minimum + np.mod(val - minimum, width)

    def apply_reflective_boundary(self, key, val):
        minimum = self.prior_minimum_dict[key]
        width = self.prior_width_dict[key]
        val_normalised = (val - minimum) / width
        val_normalised_reflected = reflect(np.array(val_normalised))
        return minimum + width * val_normalised_reflected

    def __call__(self, chain, likelihood=None, priors=None):

        if getattr(self, "needs_likelihood_and_priors", False):
            sample, log_factor = self.propose(chain, likelihood, priors)
        else:
            sample, log_factor = self.propose(chain)

        if log_factor == 0:
            sample = self.apply_boundaries(sample)

        return sample, log_factor

    @abstractmethod
    def propose(self, chain):
        """Propose a new point

        This method must be overwritten by implemented proposals. The propose
        method is called by __call__, then boundaries applied, before returning
        the proposed point.

        Parameters
        ----------
        chain: bilby.core.sampler.bilby_mcmc.chain.Chain
            The chain to use for the proposal

        Returns
        -------
        proposal: bilby.core.sampler.bilby_mcmc.Sample
            The proposed point
        log_factor: float
            The natural-log of the additional factor entering the acceptance
            probability to ensure detailed balance. For symmetric proposals,
            a value of 0 should be returned.
        """
        pass

    @staticmethod
    def check_dependencies(warn=True):
        """Check the dependencies required to use the proposal

        Parameters
        ----------
        warn: bool
            If true, print a warning

        Returns
        -------
        check: bool
            If true, dependencies exist
        """
        return True


class FixedGaussianProposal(BaseProposal):
    """A proposal using a fixed non-correlated Gaussian distribution

    Parameters
    ----------
    priors: bilby.core.prior.PriorDict
        The set of priors
    weight: float
        Weighting factor
    subset: list
        A list of keys for which to restrict the proposal to (other parameters
        will be kept fixed)
    sigma: float
        The scaling factor for proposals
    """

    def __init__(self, priors, weight=1, subset=None, sigma=0.01):
        super(FixedGaussianProposal, self).__init__(priors, weight, subset)
        self.sigmas = {}
        for key in self.parameters:
            if np.isinf(self.prior_width_dict[key]):
                self.prior_width_dict[key] = 1
            if isinstance(sigma, float):
                self.sigmas[key] = sigma
            elif isinstance(sigma, dict):
                self.sigmas[key] = sigma[key]
            else:
                raise SamplerError("FixedGaussianProposal sigma not understood")

    def propose(self, chain):
        sample = chain.current_sample
        for key in self.parameters:
            sigma = self.prior_width_dict[key] * self.sigmas[key]
            sample[key] += sigma * random.rng.normal(0, 1)
        log_factor = 0
        return sample, log_factor


class AdaptiveGaussianProposal(BaseProposal):
    def __init__(
        self,
        priors,
        weight=1,
        subset=None,
        sigma=1,
        scale_init=1e0,
        stop=1e5,
        target_facc=0.234,
    ):
        super(AdaptiveGaussianProposal, self).__init__(priors, weight, subset)
        self.sigmas = {}
        for key in self.parameters:
            if np.isinf(self.prior_width_dict[key]):
                self.prior_width_dict[key] = 1
            if isinstance(sigma, (float, int)):
                self.sigmas[key] = sigma
            elif isinstance(sigma, dict):
                self.sigmas[key] = sigma[key]
            else:
                raise SamplerError("AdaptiveGaussianProposal sigma not understood")

        self.target_facc = target_facc
        self.scale = scale_init
        self.stop = stop
        self._str_attrs.append("scale")
        self._last_accepted = 0

    def propose(self, chain):
        sample = chain.current_sample
        self.update_scale(chain)
        if random.rng.uniform(0, 1) < 1e-3:
            factor = 1e1
        elif random.rng.uniform(0, 1) < 1e-4:
            factor = 1e2
        else:
            factor = 1
        for key in self.parameters:
            sigma = factor * self.scale * self.prior_width_dict[key] * self.sigmas[key]
            sample[key] += sigma * random.rng.normal(0, 1)
        log_factor = 0
        return sample, log_factor

    def update_scale(self, chain):
        """
        The adaptation of the scale follows (35)/(36) of https://arxiv.org/abs/1409.7215
        """
        if 0 < self.n < self.stop:
            s_gamma = (self.stop / self.n) ** 0.2 - 1
            if self.accepted > self._last_accepted:
                self.scale += s_gamma * (1 - self.target_facc) / 100
            else:
                self.scale -= s_gamma * self.target_facc / 100
            self._last_accepted = self.accepted
            self.scale = max(self.scale, 1 / self.stop)


class DifferentialEvolutionProposal(BaseProposal):
    """A proposal using Differential Evolution

    Parameters
    ----------
    priors: bilby.core.prior.PriorDict
        The set of priors
    weight: float
        Weighting factor
    subset: list
        A list of keys for which to restrict the proposal to (other parameters
        will be kept fixed)
    mode_hopping_frac: float
        The fraction of proposals which use 'mode hopping'
    """

    def __init__(self, priors, weight=1, subset=None, mode_hopping_frac=0.5):
        super(DifferentialEvolutionProposal, self).__init__(priors, weight, subset)
        self.mode_hopping_frac = mode_hopping_frac

    def propose(self, chain):
        theta = chain.current_sample
        theta1 = chain.random_sample
        theta2 = chain.random_sample
        if random.rng.uniform(0, 1) > self.mode_hopping_frac:
            gamma = 1
        else:
            # Base jump size
            gamma = random.rng.normal(0, 2.38 / np.sqrt(2 * self.ndim))
            # Scale uniformly in log between 0.1 and 10 times
            gamma *= np.exp(np.log(0.1) + np.log(100.0) * random.rng.uniform(0, 1))

        for key in self.parameters:
            theta[key] += gamma * (theta2[key] - theta1[key])

        log_factor = 0
        return theta, log_factor


class UniformProposal(BaseProposal):
    """A proposal using uniform draws from the prior support

    Note: for priors with infinite support, this proposal will not propose a
    point, leading to inefficient sampling. You may wish to omit this proposal
    if you have priors with infinite support.

    Parameters
    ----------
    priors: bilby.core.prior.PriorDict
        The set of priors
    weight: float
        Weighting factor
    subset: list
        A list of keys for which to restrict the proposal to (other parameters
        will be kept fixed)
    """

    def __init__(self, priors, weight=1, subset=None):
        super(UniformProposal, self).__init__(priors, weight, subset)

    def propose(self, chain):
        sample = chain.current_sample
        for key in self.parameters:
            width = self.prior_width_dict[key]
            if np.isinf(width) is False:
                sample[key] = random.rng.uniform(
                    self.prior_minimum_dict[key], self.prior_maximum_dict[key]
                )
            else:
                # Unable to generate a uniform sample on infinite support
                pass
        log_factor = 0
        return sample, log_factor


class PriorProposal(BaseProposal):
    """A proposal using draws from the prior distribution

    Note: for priors which use interpolation, this proposal can be problematic
    as the proposal gets pickled in multiprocessing. Either, use serial
    processing (npool=1) or fall back to a UniformProposal.

    Parameters
    ----------
    priors: bilby.core.prior.PriorDict
        The set of priors
    weight: float
        Weighting factor
    subset: list
        A list of keys for which to restrict the proposal to (other parameters
        will be kept fixed)
    """

    def __init__(self, priors, weight=1, subset=None):
        super(PriorProposal, self).__init__(priors, weight, subset)
        self.priors = PriorDict({key: priors[key] for key in self.parameters})

    def propose(self, chain):
        sample = chain.current_sample
        lnp_theta = self.priors.ln_prob(sample.as_dict(self.parameters))
        prior_sample = self.priors.sample()
        for key in self.parameters:
            sample[key] = prior_sample[key]
        lnp_thetaprime = self.priors.ln_prob(sample.as_dict(self.parameters))
        log_factor = lnp_theta - lnp_thetaprime
        return sample, log_factor


_density_estimate_doc = """ A proposal using draws from a {estimator} fit to the chain

Parameters
----------
priors: bilby.core.prior.PriorDict
    The set of priors
weight: float
    Weighting factor
subset: list
    A list of keys for which to restrict the proposal to (other parameters
    will be kept fixed)
first_fit: int
    The number of steps to take before first fitting the KDE
fit_multiplier: int
    The multiplier for the next fit
nsamples_for_density: int
    The number of samples to use when fitting the KDE
fallback: bilby.core.sampler.bilby_mcmc.proposal.BaseProposal
    A proposal to use before first training
scale_fits: int
    A scaling factor for both the initial and subsequent updates
"""


class DensityEstimateProposal(BaseProposal):
    def __init__(
        self,
        priors,
        weight=1,
        subset=None,
        first_fit=1000,
        fit_multiplier=10,
        nsamples_for_density=1000,
        fallback=AdaptiveGaussianProposal,
        scale_fits=1,
    ):
        super(DensityEstimateProposal, self).__init__(priors, weight, subset)
        self.nsamples_for_density = nsamples_for_density
        self.fallback = fallback(priors, weight, subset)
        self.fit_multiplier = fit_multiplier * scale_fits

        # Counters
        self.steps_since_refit = 0
        self.next_refit_time = first_fit * scale_fits
        self.density = None
        self.trained = False
        self._str_attrs.append("trained")

    density_name = None
    __doc__ = _density_estimate_doc.format(estimator=density_name)

    def _fit(self, dataset):
        raise NotImplementedError

    def _evaluate(self, point):
        raise NotImplementedError

    def _sample(self, nsamples=None):
        raise NotImplementedError

    def refit(self, chain):
        current_density = self.density
        start = time.time()

        # Draw two (possibly overlapping) data sets for training and verification
        dataset = []
        verification_dataset = []
        nsamples_for_density = min(chain.position, self.nsamples_for_density)
        for _ in range(nsamples_for_density):
            s = chain.random_sample
            dataset.append([s[key] for key in self.parameters])
            s = chain.random_sample
            verification_dataset.append([s[key] for key in self.parameters])

        # Fit the density
        self.density = self._fit(np.array(dataset).T)

        # Print a log message
        took = time.time() - start
        logger.debug(
            f"{self.density_name} construction at {self.steps_since_refit} finished"
            f" for length {chain.position} chain, took {took:0.2f}s."
            f" Current accept-ratio={self.acceptance_ratio:0.2f}"
        )

        # Reset counters for next training
        self.steps_since_refit = 0
        self.next_refit_time *= self.fit_multiplier

        # Verify training hasn't overconstrained
        new_draws = np.atleast_2d(self._sample(1000))
        verification_dataset = np.array(verification_dataset)
        fail_parameters = []
        for ii, key in enumerate(self.parameters):
            std_draws = np.std(new_draws[:, ii])
            std_verification = np.std(verification_dataset[:, ii])
            if std_draws < 0.1 * std_verification:
                fail_parameters.append(key)

        if len(fail_parameters) > 0:
            logger.debug(
                f"{self.density_name} construction failed verification and is discarded"
            )
            self.density = current_density
        else:
            self.trained = True

    def propose(self, chain):
        self.steps_since_refit += 1

        # Check if we refit
        testA = self.steps_since_refit >= self.next_refit_time
        if testA:
            try:
                self.refit(chain)
            except Exception as e:
                logger.warning(f"Failed to refit chain due to error {e}")

        # If KDE is yet to be fitted, use the fallback
        if self.trained is False:
            return self.fallback.propose(chain)

        # Grab the current sample and it's probability under the KDE
        theta = chain.current_sample
        ln_p_theta = self._evaluate(list(theta.as_dict(self.parameters).values()))

        # Sample and update theta
        new_sample = self._sample(1)
        for key, val in zip(self.parameters, new_sample):
            theta[key] = val

        # Calculate the probability of the new sample and the KDE
        ln_p_thetaprime = self._evaluate(list(theta.as_dict(self.parameters).values()))

        # Calculate Q(theta|theta') / Q(theta'|theta)
        log_factor = ln_p_theta - ln_p_thetaprime

        return theta, log_factor


class KDEProposal(DensityEstimateProposal):
    density_name = "Gaussian KDE"
    __doc__ = _density_estimate_doc.format(estimator=density_name)

    def _fit(self, dataset):
        return gaussian_kde(dataset)

    def _evaluate(self, point):
        return self.density.logpdf(point)[0]

    def _sample(self, nsamples=None):
        return np.atleast_1d(np.squeeze(self.density.resample(nsamples)))


class GMMProposal(DensityEstimateProposal):
    density_name = "Gaussian Mixture Model"
    __doc__ = _density_estimate_doc.format(estimator=density_name)

    def _fit(self, dataset):
        from sklearn.mixture import GaussianMixture

        density = GaussianMixture(n_components=10)
        density.fit(dataset.T)
        return density

    def _evaluate(self, point):
        return np.squeeze(self.density.score_samples(np.atleast_2d(point)))

    def _sample(self, nsamples=None):
        return np.squeeze(self.density.sample(n_samples=nsamples)[0])

    @staticmethod
    def check_dependencies(warn=True):
        if importlib.util.find_spec("sklearn") is None:
            if warn:
                logger.warning(
                    "Unable to utilise GMMProposal as sklearn is not installed"
                )
            return False
        else:
            return True


class NormalizingFlowProposal(DensityEstimateProposal):
    density_name = "Normalizing Flow"
    __doc__ = _density_estimate_doc.format(estimator=density_name) + (
        """
        js_factor: float
            The factor to use in determining the max-JS factor to terminate
            training.
        max_training_epochs: int
            The maximum bumber of traning steps to take
        """
    )

    def __init__(
        self,
        priors,
        weight=1,
        subset=None,
        first_fit=1000,
        fit_multiplier=10,
        max_training_epochs=1000,
        scale_fits=1,
        nsamples_for_density=1000,
        js_factor=10,
        fallback=AdaptiveGaussianProposal,
    ):
        super(NormalizingFlowProposal, self).__init__(
            priors=priors,
            weight=weight,
            subset=subset,
            first_fit=first_fit,
            fit_multiplier=fit_multiplier,
            nsamples_for_density=nsamples_for_density,
            fallback=fallback,
            scale_fits=scale_fits,
        )
        self.initialised = False
        self.max_training_epochs = max_training_epochs
        self.js_factor = js_factor

    def initialise(self):
        self.setup_flow()
        self.setup_optimizer()
        self.initialised = True

    def setup_flow(self):
        if self.ndim < 3:
            self.setup_basic_flow()
        else:
            self.setup_NVP_flow()

    def setup_NVP_flow(self):
        from .flows import NVPFlow

        self.flow = NVPFlow(
            features=self.ndim,
            hidden_features=self.ndim * 2,
            num_layers=2,
            num_blocks_per_layer=2,
            batch_norm_between_layers=True,
            batch_norm_within_layers=True,
        )

    def setup_basic_flow(self):
        from .flows import BasicFlow

        self.flow = BasicFlow(features=self.ndim)

    def setup_optimizer(self):
        from torch import optim

        self.optimizer = optim.Adam(self.flow.parameters())

    def get_training_data(self, chain):
        training_data = []
        nsamples_for_density = min(chain.position, self.nsamples_for_density)
        for _ in range(nsamples_for_density):
            s = chain.random_sample
            training_data.append([s[key] for key in self.parameters])
        return training_data

    def _calculate_js(self, validation_samples, training_samples_draw):
        # Calculate the maximum JS between the validation and draw
        max_js = 0
        for i in range(self.ndim):
            A = validation_samples[:, i]
            B = training_samples_draw[:, i]
            xmin = np.min([np.min(A), np.min(B)])
            xmax = np.min([np.max(A), np.max(B)])
            xval = np.linspace(xmin, xmax, 100)
            Apdf = gaussian_kde(A)(xval)
            Bpdf = gaussian_kde(B)(xval)
            js = jensenshannon(Apdf, Bpdf)
            max_js = max(max_js, js)
        return np.power(max_js, 2)

    def train(self, chain):
        logger.debug("Starting NF training")

        import torch

        start = time.time()

        training_samples = np.array(self.get_training_data(chain))
        validation_samples = np.array(self.get_training_data(chain))

        training_tensor = torch.tensor(training_samples, dtype=torch.float32)

        max_js_threshold = self.js_factor / self.nsamples_for_density

        for epoch in range(1, self.max_training_epochs + 1):
            self.optimizer.zero_grad()
            loss = -self.flow.log_prob(inputs=training_tensor).mean()
            loss.backward()
            self.optimizer.step()

            # Draw from the current flow
            self.flow.eval()
            training_samples_draw = (
                self.flow.sample(self.nsamples_for_density).detach().numpy()
            )
            self.flow.train()

            if np.mod(epoch, 10) == 0:
                max_js_bits = self._calculate_js(
                    validation_samples, training_samples_draw
                )
                if max_js_bits < max_js_threshold:
                    logger.debug(
                        f"Training complete after {epoch} steps, "
                        f"max_js_bits={max_js_bits:0.5f}<{max_js_threshold}"
                    )
                    break

        took = time.time() - start
        logger.debug(
            f"Flow training step ({self.steps_since_refit}) finished"
            f" for length {chain.position} chain, took {took:0.2f}s."
            f" Current accept-ratio={self.acceptance_ratio:0.2f}"
        )
        self.steps_since_refit = 0
        self.next_refit_time *= self.fit_multiplier
        self.trained = True

    def propose(self, chain):
        if self.initialised is False:
            self.initialise()

        import torch

        self.steps_since_refit += 1
        theta = chain.current_sample

        # Check if we retrain the NF
        testA = self.steps_since_refit >= self.next_refit_time
        if testA:
            try:
                self.train(chain)
            except Exception as e:
                logger.warning(f"Failed to retrain chain due to error {e}")

        if self.trained is False:
            return self.fallback.propose(chain)

        self.flow.eval()
        theta_prime_T = self.flow.sample(1)

        logp_theta_prime = self.flow.log_prob(theta_prime_T).detach().numpy()[0]
        theta_T = torch.tensor(
            np.atleast_2d([theta[key] for key in self.parameters]), dtype=torch.float32
        )
        logp_theta = self.flow.log_prob(theta_T).detach().numpy()[0]
        log_factor = logp_theta - logp_theta_prime

        flow_sample_values = np.atleast_1d(np.squeeze(theta_prime_T.detach().numpy()))
        for key, val in zip(self.parameters, flow_sample_values):
            theta[key] = val

        return theta, float(log_factor)

    @staticmethod
    def check_dependencies(warn=True):
        if importlib.util.find_spec("glasflow") is None:
            if warn:
                logger.warning(
                    "Unable to utilise NormalizingFlowProposal as glasflow is not installed"
                )
            return False
        else:
            return True


class FixedJumpProposal(BaseProposal):
    def __init__(self, priors, jumps=1, subset=None, weight=1, scale=1e-4):
        super(FixedJumpProposal, self).__init__(priors, weight, subset)
        self.scale = scale
        if isinstance(jumps, (int, float)):
            self.jumps = {key: jumps for key in self.parameters}
        elif isinstance(jumps, dict):
            self.jumps = jumps
        else:
            raise SamplerError("jumps not understood")

    def propose(self, chain):
        sample = chain.current_sample
        for key, jump in self.jumps.items():
            sign = random.rng.integers(2) * 2 - 1
            sample[key] += sign * jump + self.epsilon * self.prior_width_dict[key]
        log_factor = 0
        return sample, log_factor

    @property
    def epsilon(self):
        return self.scale * random.rng.normal()


class FisherMatrixProposal(AdaptiveGaussianProposal):
    needs_likelihood_and_priors = True
    """Fisher Matrix Proposals

    Uses a finite differencing approach motivated by BayesWave (see, e.g.
    https://arxiv.org/abs/1410.3835). The inverse Fisher Information Matrix
    is calculated from the current sample, then proposals are drawn from a
    multivariate Gaussian and scaled by an adaptive parameter.
    """

    def __init__(
        self,
        priors,
        subset=None,
        weight=1,
        update_interval=100,
        scale_init=1e0,
        fd_eps=1e-4,
        adapt=False,
    ):
        super(FisherMatrixProposal, self).__init__(
            priors, weight, subset, scale_init=scale_init
        )
        self.update_interval = update_interval
        self.steps_since_update = update_interval
        self.adapt = adapt
        self.mean = np.zeros(len(self.parameters))
        self.fd_eps = fd_eps

    def propose(self, chain, likelihood, priors):
        sample = chain.current_sample
        if self.adapt:
            self.update_scale(chain)
        if self.steps_since_update >= self.update_interval:
            fmp = FisherMatrixPosteriorEstimator(
                likelihood, priors, parameters=self.parameters, fd_eps=self.fd_eps
            )
            try:
                self.iFIM = fmp.calculate_iFIM(sample.dict)
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
                logger.warning(f"FisherMatrixProposal failed with {e}")
                if hasattr(self, "iFIM") is False:
                    # No past iFIM exists, return sample
                    return sample, 0
            self.steps_since_update = 0

        jump = self.scale * random.rng.multivariate_normal(
            self.mean, self.iFIM, check_valid="ignore"
        )

        for key, val in zip(self.parameters, jump):
            sample[key] += val

        log_factor = 0
        self.steps_since_update += 1
        return sample, log_factor


class BaseGravitationalWaveTransientProposal(BaseProposal):
    def __init__(self, priors, weight=1):
        super(BaseGravitationalWaveTransientProposal, self).__init__(
            priors, weight=weight
        )
        if "phase" in priors:
            self.phase_key = "phase"
        elif "delta_phase" in priors:
            self.phase_key = "delta_phase"
        else:
            self.phase_key = None

    def get_cos_theta_jn(self, sample):
        if "cos_theta_jn" in sample.parameter_keys:
            cos_theta_jn = sample["cos_theta_jn"]
        elif "theta_jn" in sample.parameter_keys:
            cos_theta_jn = np.cos(sample["theta_jn"])
        else:
            raise SamplerError()
        return cos_theta_jn

    def get_phase(self, sample):
        if "phase" in sample.parameter_keys:
            return sample["phase"]
        elif "delta_phase" in sample.parameter_keys:
            cos_theta_jn = self.get_cos_theta_jn(sample)
            delta_phase = sample["delta_phase"]
            psi = sample["psi"]
            phase = np.mod(delta_phase - np.sign(cos_theta_jn) * psi, 2 * np.pi)
        else:
            raise SamplerError()
        return phase

    def get_delta_phase(self, phase, sample):
        cos_theta_jn = self.get_cos_theta_jn(sample)
        psi = sample["psi"]
        delta_phase = phase + np.sign(cos_theta_jn) * psi
        return delta_phase


class CorrelatedPolarisationPhaseJump(BaseGravitationalWaveTransientProposal):
    def __init__(self, priors, weight=1):
        super(CorrelatedPolarisationPhaseJump, self).__init__(priors, weight=weight)

    def propose(self, chain):
        sample = chain.current_sample
        phase = self.get_phase(sample)

        alpha = sample["psi"] + phase
        beta = sample["psi"] - phase

        draw = random.rng.random()
        if draw < 0.5:
            alpha = 3.0 * np.pi * random.rng.random()
        else:
            beta = 3.0 * np.pi * random.rng.random() - 2 * np.pi

        # Update
        sample["psi"] = (alpha + beta) * 0.5
        phase = (alpha - beta) * 0.5

        if self.phase_key == "delta_phase":
            sample["delta_phase"] = self.get_delta_phase(phase, sample)
        else:
            sample["phase"] = phase

        log_factor = 0
        return sample, log_factor


class PhaseReversalProposal(BaseGravitationalWaveTransientProposal):
    def __init__(self, priors, weight=1, fuzz=True, fuzz_sigma=1e-1):
        super(PhaseReversalProposal, self).__init__(priors, weight)
        self.fuzz = fuzz
        self.fuzz_sigma = fuzz_sigma
        if self.phase_key is None:
            raise SamplerError(
                f"{type(self).__name__} initialised without a phase prior"
            )

    def propose(self, chain):
        sample = chain.current_sample
        phase = sample[self.phase_key]
        sample[self.phase_key] = np.mod(phase + np.pi + self.epsilon, 2 * np.pi)
        log_factor = 0
        return sample, log_factor

    @property
    def epsilon(self):
        if self.fuzz:
            return random.rng.normal(0, self.fuzz_sigma)
        else:
            return 0


class PolarisationReversalProposal(PhaseReversalProposal):
    def __init__(self, priors, weight=1, fuzz=True, fuzz_sigma=1e-3):
        super(PolarisationReversalProposal, self).__init__(
            priors, weight, fuzz, fuzz_sigma
        )
        self.fuzz = fuzz

    def propose(self, chain):
        sample = chain.current_sample
        psi = sample["psi"]
        sample["psi"] = np.mod(psi + np.pi / 2 + self.epsilon, np.pi)
        log_factor = 0
        return sample, log_factor


class PhasePolarisationReversalProposal(PhaseReversalProposal):
    def __init__(self, priors, weight=1, fuzz=True, fuzz_sigma=1e-1):
        super(PhasePolarisationReversalProposal, self).__init__(
            priors, weight, fuzz, fuzz_sigma
        )
        self.fuzz = fuzz

    def propose(self, chain):
        sample = chain.current_sample
        sample[self.phase_key] = np.mod(
            sample[self.phase_key] + np.pi + self.epsilon, 2 * np.pi
        )
        sample["psi"] = np.mod(sample["psi"] + np.pi / 2 + self.epsilon, np.pi)
        log_factor = 0
        return sample, log_factor


class StretchProposal(BaseProposal):
    """The Goodman & Weare (2010) Stretch proposal for an MCMC chain

    Implementation of the Stretch proposal using a sample drawn from the chain.
    We assume the form of g(z) from Equation (9) of [1].

    References
    ----------
    [1] Goodman & Weare (2010)
        https://ui.adsabs.harvard.edu/abs/2010CAMCS...5...65G/abstract

    """

    def __init__(self, priors, weight=1, subset=None, scale=2):
        super(StretchProposal, self).__init__(priors, weight, subset)
        self.scale = scale

    def propose(self, chain):
        sample = chain.current_sample

        # Draw a random sample
        rand = chain.random_sample

        return _stretch_move(sample, rand, self.scale, self.ndim, self.parameters)


def _stretch_move(sample, complement, scale, ndim, parameters):
    # Draw z
    u = random.rng.uniform(0, 1)
    z = (u * (scale - 1) + 1) ** 2 / scale

    log_factor = (ndim - 1) * np.log(z)

    for key in parameters:
        sample[key] = complement[key] + (sample[key] - complement[key]) * z

    return sample, log_factor


class EnsembleProposal(BaseProposal):
    """Base EnsembleProposal class for ensemble-based swap proposals"""

    def __init__(self, priors, weight=1):
        super(EnsembleProposal, self).__init__(priors, weight)

    def __call__(self, chain, chain_complement):
        sample, log_factor = self.propose(chain, chain_complement)
        if log_factor == 0:
            sample = self.apply_boundaries(sample)
        return sample, log_factor


class EnsembleStretch(EnsembleProposal):
    """The Goodman & Weare (2010) Stretch proposal for an Ensemble

    Implementation of the Stretch proposal using a sample drawn from complement.
    We assume the form of g(z) from Equation (9) of [1].

    References
    ----------
    [1] Goodman & Weare (2010)
        https://ui.adsabs.harvard.edu/abs/2010CAMCS...5...65G/abstract

    """

    def __init__(self, priors, weight=1, scale=2):
        super(EnsembleStretch, self).__init__(priors, weight)
        self.scale = scale

    def propose(self, chain, chain_complement):
        sample = chain.current_sample
        completement = chain_complement[
            random.rng.integers(len(chain_complement))
        ].current_sample
        return _stretch_move(
            sample, completement, self.scale, self.ndim, self.parameters
        )


def get_default_ensemble_proposal_cycle(priors):
    return ProposalCycle([EnsembleStretch(priors)])


def get_proposal_cycle(string, priors, L1steps=1, warn=True):
    big_weight = 10
    small_weight = 5
    tiny_weight = 0.5

    if "gwA" in string:
        # Parameters for learning proposals
        learning_kwargs = dict(
            first_fit=1000, nsamples_for_density=10000, fit_multiplier=2
        )

        all_but_cal = [key for key in priors if "recalib" not in key]
        plist = [
            AdaptiveGaussianProposal(priors, weight=small_weight, subset=all_but_cal),
            DifferentialEvolutionProposal(
                priors, weight=small_weight, subset=all_but_cal
            ),
        ]

        if GMMProposal.check_dependencies(warn=warn) is False:
            raise SamplerError(
                "the gwA proposal_cycle required the GMMProposal dependencies"
            )

        if priors.intrinsic:
            intrinsic = PARAMETER_SETS["intrinsic"]
            plist += [
                AdaptiveGaussianProposal(priors, weight=small_weight, subset=intrinsic),
                DifferentialEvolutionProposal(
                    priors, weight=small_weight, subset=intrinsic
                ),
                KDEProposal(
                    priors, weight=small_weight, subset=intrinsic, **learning_kwargs
                ),
                GMMProposal(
                    priors, weight=small_weight, subset=intrinsic, **learning_kwargs
                ),
            ]

        if priors.extrinsic:
            extrinsic = PARAMETER_SETS["extrinsic"]
            plist += [
                AdaptiveGaussianProposal(priors, weight=small_weight, subset=extrinsic),
                DifferentialEvolutionProposal(
                    priors, weight=small_weight, subset=extrinsic
                ),
                KDEProposal(
                    priors, weight=small_weight, subset=extrinsic, **learning_kwargs
                ),
                GMMProposal(
                    priors, weight=small_weight, subset=extrinsic, **learning_kwargs
                ),
            ]

        if priors.mass:
            mass = PARAMETER_SETS["mass"]
            plist += [
                DifferentialEvolutionProposal(priors, weight=small_weight, subset=mass),
                GMMProposal(
                    priors, weight=small_weight, subset=mass, **learning_kwargs
                ),
                FisherMatrixProposal(
                    priors,
                    weight=small_weight,
                    subset=mass,
                ),
            ]

        if priors.spin:
            spin = PARAMETER_SETS["spin"]
            plist += [
                DifferentialEvolutionProposal(priors, weight=small_weight, subset=spin),
                GMMProposal(
                    priors, weight=small_weight, subset=spin, **learning_kwargs
                ),
                FisherMatrixProposal(
                    priors,
                    weight=big_weight,
                    subset=spin,
                ),
            ]
        if priors.measured_spin:
            measured_spin = PARAMETER_SETS["measured_spin"]
            plist += [
                AdaptiveGaussianProposal(
                    priors, weight=small_weight, subset=measured_spin
                ),
                FisherMatrixProposal(
                    priors,
                    weight=small_weight,
                    subset=measured_spin,
                ),
            ]

        if priors.mass and priors.spin:
            primary_spin_and_q = PARAMETER_SETS["primary_spin_and_q"]
            plist += [
                DifferentialEvolutionProposal(
                    priors, weight=small_weight, subset=primary_spin_and_q
                ),
            ]

        if getattr(priors, "tidal", False):
            tidal = PARAMETER_SETS["tidal"]
            plist += [
                DifferentialEvolutionProposal(
                    priors, weight=small_weight, subset=tidal
                ),
                PriorProposal(priors, weight=small_weight, subset=tidal),
            ]
        if priors.phase:
            plist += [
                PhaseReversalProposal(priors, weight=tiny_weight),
            ]
        if priors.phase and "psi" in priors.non_fixed_keys:
            plist += [
                CorrelatedPolarisationPhaseJump(priors, weight=tiny_weight),
                PhasePolarisationReversalProposal(priors, weight=tiny_weight),
            ]
        if priors.sky:
            sky = PARAMETER_SETS["sky"]
            plist += [
                FisherMatrixProposal(
                    priors,
                    weight=small_weight,
                    subset=sky,
                ),
                GMMProposal(
                    priors,
                    weight=small_weight,
                    subset=sky,
                    **learning_kwargs,
                ),
            ]
        for key in ["time_jitter", "psi", "phi_12", "tilt_2", "lambda_1", "lambda_2"]:
            if key in priors.non_fixed_keys:
                plist.append(PriorProposal(priors, subset=[key], weight=tiny_weight))
        if "chi_1_in_plane" in priors and "chi_2_in_plane" in priors:
            in_plane = ["chi_1_in_plane", "chi_2_in_plane", "phi_12"]
            plist.append(UniformProposal(priors, subset=in_plane, weight=tiny_weight))
        if any("recalib_" in key for key in priors):
            calibration = [key for key in priors if "recalib_" in key]
            plist.append(PriorProposal(priors, subset=calibration, weight=small_weight))
    else:
        plist = [
            AdaptiveGaussianProposal(priors, weight=big_weight),
            DifferentialEvolutionProposal(priors, weight=big_weight),
            UniformProposal(priors, weight=tiny_weight),
            KDEProposal(priors, weight=big_weight, scale_fits=L1steps),
            FisherMatrixProposal(priors, weight=big_weight),
        ]
        if GMMProposal.check_dependencies(warn=warn):
            plist.append(GMMProposal(priors, weight=big_weight, scale_fits=L1steps))

    plist = remove_proposals_using_string(plist, string)
    return ProposalCycle(plist)


def remove_proposals_using_string(plist, string):
    mapping = dict(
        DE=DifferentialEvolutionProposal,
        AG=AdaptiveGaussianProposal,
        ST=StretchProposal,
        FG=FixedGaussianProposal,
        NF=NormalizingFlowProposal,
        KD=KDEProposal,
        GM=GMMProposal,
        PR=PriorProposal,
        UN=UniformProposal,
        FM=FisherMatrixProposal,
    )

    for element in string.split("no")[1:]:
        if element in mapping:
            plist = [p for p in plist if isinstance(p, mapping[element]) is False]
    return plist
