import inspect
import pickle
import os

import numpy as np
import sbi
import sbi.utils
import sbi.inference
import torch

from .base import Likelihood
from ..utils import logger, check_directory_exists_and_if_not_mkdir


class GenerateData(object):
    """
    A generic base class for data generator objects

    SimulateData instances generate data from an underlying model in a form
    suitable to pass into the SBI package

    Parameters
    ==========
    model: function or callable
        A function that takes as input a set of arguments and return a
        realisation of the data given those parameters.
    fixed_arguments: dict
        A dictionary containing keys (pertaining to the model arguments) and
        values that are to be fixed.
    """

    def __init__(self, parameters, call_parameter_key_list):
        self.parameters = parameters
        self.call_parameter_key_list = call_parameter_key_list

    def get_data(parameters: dict):
        NotImplementedError("Method get_data() should be implemented by subclass")

    def fix_parameter(self, key, val):
        self.parameters[key] = val
        self.call_parameter_key_list.pop(self.call_parameter_key_list.index(key))

    def __call__(self, new_parameter_list):
        if len(new_parameter_list) != len(self.call_parameter_key_list):
            raise ValueError()

        for key, val in zip(self.call_parameter_key_list, new_parameter_list):
            self.parameters[key] = val

        return torch.as_tensor(self.get_data(self.parameters))


class GenerateWhiteGaussianNoise(GenerateData):
    """
    TBD

    Parameters
    ==========
    num_data:
    """

    def __init__(self, data_shape):
        super(GenerateWhiteGaussianNoise, self).__init__(
            parameters=dict(sigma=None),
            call_parameter_key_list=["sigma"],
        )
        self.data_shape = data_shape

    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        return np.random.normal(0, self.parameters["sigma"], self.data_shape)


class GenerateDeterministicModel(GenerateData):
    """
    TBD

    Parameters
    ==========
    model: function or callable
        A function that takes as input a set of arguments and return a
        realisation of the data given those parameters.
    fixed_arguments: dict
        A dictionary containing keys (pertaining to the model arguments) and
        values that are to be fixed.
    """

    def __init__(self, model, fixed_arguments_dict=None):
        model_arguments = inspect.getfullargspec(model).args
        parameters = {key: None for key in model_arguments}
        call_parameter_key_list = [
            key for key in parameters if key not in fixed_arguments_dict
        ]

        super(GenerateDeterministicModel, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
        )
        self.model = model
        self.fixed_arguments_dict = fixed_arguments_dict
        self.parameters.update(fixed_arguments_dict)

    def get_data(self, parameters: dict):
        kwargs = self.fixed_arguments_dict | {key: parameters[key] for key in self.call_parameter_key_list}
        return self.model(**kwargs)


class AdditiveWhiteGaussianNoise(GenerateData):
    def __init__(self, model, fixed_arguments_dict, bilby_prior):
        # Create the signal instance
        self.signal = GenerateDeterministicModel(
            model=model, fixed_arguments_dict=fixed_arguments_dict
        )

        # Update the signal with the fixed parameters from the prior
        for key, val in bilby_prior.items():
            if val.is_fixed and key in self.signal.call_parameter_key_list:
                self.signal.fixed_arguments_dict[key] = val.peak
                self.signal.fix_parameter(key, val.peak)

        # Create the noise instance
        signal_shape = self.signal.get_data(bilby_prior.sample()).shape
        self.noise = GenerateWhiteGaussianNoise(data_shape=signal_shape)

        for key, val in bilby_prior.items():
            if val.is_fixed and key in self.noise.call_parameter_key_list:
                self.noise.fix_parameter(key, val.peak)

        # Extract the parameters and keys
        call_parameter_key_list = (
            self.signal.call_parameter_key_list + self.noise.call_parameter_key_list
        )
        joint_parameters = self.noise.parameters | self.signal.parameters
        parameters = {
            key: val
            for key, val in joint_parameters.items()
            if key in call_parameter_key_list
        }

        super(AdditiveWhiteGaussianNoise, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
        )

    def get_data(self, parameters: dict):
        sparameters = {
            key: val
            for key, val in parameters.items()
            if key in self.signal.call_parameter_key_list
        }
        nparameters = {
            key: val
            for key, val in parameters.items()
            if key in self.noise.call_parameter_key_list
        }
        sdata = self.signal.get_data(sparameters)
        ndata = self.noise.get_data(nparameters)
        return sdata + ndata


class NLELikelihood(Likelihood):
    def __init__(
        self,
        yobs,
        generator,
        bilby_prior,
        label,
        num_simulations=1000,
        num_workers=1,
        cache=True,
        cache_directory="likelihood_cache",
    ):
        """
        Neural likelihood estimated with SNLE

        Parameters
        ----------
        yobs: array_like
            TBD
        generate: instance of SimulateData
            An instance of the SimulateData class
        bilby_prior: bilby.core.prior.PriorDict
            The Bilby prior
        label: str
            A unique string used to cache the likelihood
        num_simulations: int
            The number of simulations used to train the neural network
        num_workers: int
            The number of workers used to..FIX ME
        cache: bool
            If true, write a copy of the likelihood to avoid retraining
        cache_directory: str
            The directory to store the likelihood cache
        """
        super().__init__(generator.parameters)

        self.yobs = yobs
        self.generator = generator
        self.bilby_prior = bilby_prior
        self.label = label
        self.num_simulations = num_simulations
        self.num_workers = num_workers
        self.cache = cache
        self.cache_directory = cache_directory
        self.fixed_parameters = [
            key for key in self.generator.call_parameter_key_list if bilby_prior[key].is_fixed
        ]

        # Initialise SBI elements
        self.init_prior()
        self.init_simulator()
        self.init_training()
        self.init_potential_fn()

    def update_parameters_from_prior(self):
        for key, val in self.bilby_prior.items():
            if val.is_fixed:
                self.parameters[key] = val.peak

    def init_prior(self):
        logger.info("Initialise the SBI prior")
        prior_min, prior_max = [], []
        for key in self.generator.call_parameter_key_list:
            if self.bilby_prior[key].is_fixed is False:
                prior_min.append(self.bilby_prior[key].minimum)
                prior_max.append(self.bilby_prior[key].maximum)

        torch_prior = sbi.utils.torchutils.BoxUniform(
            low=torch.as_tensor(prior_min),
            high=torch.as_tensor(prior_max))
        (
            self.sbi_prior,
            self.sbi_num_parameters,
            self.sbi_prior_returns_numpy,
        ) = sbi.utils.user_input_checks.process_prior(torch_prior)

    def init_simulator(self):
        logger.info("Initialise the SBI simulator")
        self.sbi_generator = sbi.utils.user_input_checks.process_simulator(
            self.generator, self.sbi_prior, self.sbi_prior_returns_numpy
        )
        sbi.utils.user_input_checks.check_sbi_inputs(
            self.sbi_generator, self.sbi_prior
        )

    def init_training(self):
        if os.path.exists(self.cache_filename):
            self.load_trained_likelihood()
        else:
            self.train_likelihood()

    def load_trained_likelihood(self):
        logger.info(f"Loading in cached NLE {self.cache_filename}")
        with open(self.cache_filename, "rb") as file:
            self.sbi_likelihood_estimator = pickle.load(file)

    def train_likelihood(self):
        logger.info("Train the NLE")
        inference = sbi.inference.SNLE(prior=self.sbi_prior)
        simulated_params, simulated_yobs = sbi.inference.simulate_for_sbi(
            self.sbi_generator,
            proposal=self.sbi_prior,
            num_simulations=self.num_simulations,
            num_workers=self.num_workers,
        )

        logger.info("Append the simulations")
        inf_and_sims = inference.append_simulations(simulated_params, simulated_yobs)

        logger.info("Train")
        self.sbi_likelihood_estimator = inf_and_sims.train()

        if self.cache:
            logger.info(f"Writing the cached NLE to {self.cache_filename}")
            check_directory_exists_and_if_not_mkdir(self.cache_directory)
            with open(self.cache_filename, "wb") as file:
                pickle.dump(self.sbi_likelihood_estimator, file)

    @property
    def cache_filename(self):
        return f"{self.cache_directory}/{self.label}_{self.num_simulations}.pkl"

    def init_potential_fn(self):
        self.sbi_potential_fn, _ = sbi.inference.likelihood_estimator_based_potential(
            self.sbi_likelihood_estimator, self.sbi_prior, self.yobs
        )

    def log_likelihood(self):
        parameters = [
            np.float32(self.parameters[key])
            for key in self.generator.call_parameter_key_list
            if key not in self.fixed_parameters
        ]
        parameter_tensor = torch.as_tensor(parameters)
        logl = self.sbi_potential_fn(parameter_tensor)
        return float(logl)
