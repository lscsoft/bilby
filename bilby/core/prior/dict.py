import json
import os
import re
from importlib import import_module
from io import open as ioopen

import numpy as np

from .analytical import DeltaFunction
from .base import Prior, Constraint
from .joint import JointPrior
from ..utils import (
    logger,
    check_directory_exists_and_if_not_mkdir,
    BilbyJsonEncoder,
    decode_bilby_json,
)


class PriorDict(dict):
    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        """A dictionary of priors

        Parameters
        ==========
        dictionary: Union[dict, str, None]
            If given, a dictionary to generate the prior set.
        filename: Union[str, None]
            If given, a file containing the prior to generate the prior set.
        conversion_function: func
            Function to convert between sampled parameters and constraints.
            Default is no conversion.
        """
        super(PriorDict, self).__init__()
        if isinstance(dictionary, dict):
            self.from_dictionary(dictionary)
        elif type(dictionary) is str:
            logger.debug(
                'Argument "dictionary" is a string.'
                + " Assuming it is intended as a file name."
            )
            self.from_file(dictionary)
        elif type(filename) is str:
            self.from_file(filename)
        elif dictionary is not None:
            raise ValueError("PriorDict input dictionary not understood")
        self._cached_normalizations = {}

        self.convert_floats_to_delta_functions()

        if conversion_function is not None:
            self.conversion_function = conversion_function
        else:
            self.conversion_function = self.default_conversion_function

    def evaluate_constraints(self, sample):
        out_sample = self.conversion_function(sample)
        prob = 1
        for key in self:
            if isinstance(self[key], Constraint) and key in out_sample:
                prob *= self[key].prob(out_sample[key])
        return prob

    def default_conversion_function(self, sample):
        """
        Placeholder parameter conversion function.

        Parameters
        ==========
        sample: dict
            Dictionary to convert

        Returns
        =======
        sample: dict
            Same as input
        """
        return sample

    def to_file(self, outdir, label):
        """Write the prior distribution to file.

        Parameters
        ==========
        outdir: str
            output directory name
        label: str
            Output file naming scheme
        """

        check_directory_exists_and_if_not_mkdir(outdir)
        prior_file = os.path.join(outdir, "{}.prior".format(label))
        logger.debug("Writing priors to {}".format(prior_file))
        joint_dists = []
        with open(prior_file, "w") as outfile:
            for key in self.keys():
                if JointPrior in self[key].__class__.__mro__:
                    distname = "_".join(self[key].dist.names) + "_{}".format(
                        self[key].dist.distname
                    )
                    if distname not in joint_dists:
                        joint_dists.append(distname)
                        outfile.write("{} = {}\n".format(distname, self[key].dist))
                    diststr = repr(self[key].dist)
                    priorstr = repr(self[key])
                    outfile.write(
                        "{} = {}\n".format(key, priorstr.replace(diststr, distname))
                    )
                else:
                    outfile.write("{} = {}\n".format(key, self[key]))

    def _get_json_dict(self):
        self.convert_floats_to_delta_functions()
        total_dict = {key: json.loads(self[key].to_json()) for key in self}
        total_dict["__prior_dict__"] = True
        total_dict["__module__"] = self.__module__
        total_dict["__name__"] = self.__class__.__name__
        return total_dict

    def to_json(self, outdir, label):
        check_directory_exists_and_if_not_mkdir(outdir)
        prior_file = os.path.join(outdir, "{}_prior.json".format(label))
        logger.debug("Writing priors to {}".format(prior_file))
        with open(prior_file, "w") as outfile:
            json.dump(self._get_json_dict(), outfile, cls=BilbyJsonEncoder, indent=2)

    def from_file(self, filename):
        """Reads in a prior from a file specification

        Parameters
        ==========
        filename: str
            Name of the file to be read in

        Notes
        =====
        Lines beginning with '#' or empty lines will be ignored.
        Priors can be loaded from:

        - bilby.core.prior as, e.g.,    :code:`foo = Uniform(minimum=0, maximum=1)`
        - floats, e.g.,                 :code:`foo = 1`
        - bilby.gw.prior as, e.g.,      :code:`foo = bilby.gw.prior.AlignedSpin()`
        - other external modules, e.g., :code:`foo = my.module.CustomPrior(...)`

        """

        comments = ["#", "\n"]
        prior = dict()
        with ioopen(filename, "r", encoding="unicode_escape") as f:
            for line in f:
                if line[0] in comments:
                    continue
                line.replace(" ", "")
                elements = line.split("=")
                key = elements[0].replace(" ", "")
                val = "=".join(elements[1:]).strip()
                prior[key] = val
        self.from_dictionary(prior)

    @classmethod
    def _get_from_json_dict(cls, prior_dict):
        try:
            class_ = getattr(
                import_module(prior_dict["__module__"]), prior_dict["__name__"]
            )
        except ImportError:
            logger.debug(
                "Cannot import prior module {}.{}".format(
                    prior_dict["__module__"], prior_dict["__name__"]
                )
            )
            class_ = cls
        except KeyError:
            logger.debug("Cannot find module name to load")
            class_ = cls
        for key in ["__module__", "__name__", "__prior_dict__"]:
            if key in prior_dict:
                del prior_dict[key]
        obj = class_(prior_dict)
        return obj

    @classmethod
    def from_json(cls, filename):
        """Reads in a prior from a json file

        Parameters
        ==========
        filename: str
            Name of the file to be read in
        """
        with open(filename, "r") as ff:
            obj = json.load(ff, object_hook=decode_bilby_json)

        # make sure priors containing JointDists are properly handled and point
        # to the same object when required
        jointdists = {}
        for key in obj:
            if isinstance(obj[key], JointPrior):
                for name in obj[key].dist.names:
                    jointdists[name] = obj[key].dist
        # set dist for joint values so that they point to the same object
        for key in obj:
            if isinstance(obj[key], JointPrior):
                obj[key].dist = jointdists[key]

        return obj

    def from_dictionary(self, dictionary):
        mvgkwargs = {}
        for key in list(dictionary.keys()):
            val = dictionary[key]
            if isinstance(val, Prior):
                continue
            elif isinstance(val, (int, float)):
                dictionary[key] = DeltaFunction(peak=val)
            elif isinstance(val, str):
                cls = val.split("(")[0]
                args = "(".join(val.split("(")[1:])[:-1]
                try:
                    dictionary[key] = DeltaFunction(peak=float(cls))
                    logger.debug("{} converted to DeltaFunction prior".format(key))
                    continue
                except ValueError:
                    pass
                if "." in cls:
                    module = ".".join(cls.split(".")[:-1])
                    cls = cls.split(".")[-1]
                else:
                    module = __name__.replace(
                        "." + os.path.basename(__file__).replace(".py", ""), ""
                    )
                try:
                    cls = getattr(import_module(module), cls, cls)
                except ModuleNotFoundError:
                    logger.error(
                        "Cannot import prior class {} for entry: {}={}".format(
                            cls, key, val
                        )
                    )
                    raise
                if key.lower() in ["conversion_function", "condition_func"]:
                    setattr(self, key, cls)
                elif isinstance(cls, str):
                    if "(" in val:
                        raise TypeError("Unable to parse prior class {}".format(cls))
                    else:
                        continue
                elif cls.__name__ in [
                    "MultivariateGaussianDist",
                    "MultivariateNormalDist",
                ]:
                    dictionary.pop(key)
                    if key not in mvgkwargs:
                        mvgkwargs[key] = cls.from_repr(args)
                elif cls.__name__ in ["MultivariateGaussian", "MultivariateNormal"]:
                    mgkwargs = {
                        item[0].strip(): cls._parse_argument_string(item[1])
                        for item in cls._split_repr(
                            ", ".join(
                                [arg for arg in args.split(",") if "dist=" not in arg]
                            )
                        ).items()
                    }
                    keymatch = re.match(r"dist=(?P<distkey>\S+),", args)
                    if keymatch is None:
                        raise ValueError(
                            "'dist' argument for MultivariateGaussian is not specified"
                        )

                    if keymatch["distkey"] not in mvgkwargs:
                        raise ValueError(
                            f"MultivariateGaussianDist {keymatch['distkey']} must be defined before {cls.__name__}"
                        )

                    mgkwargs["dist"] = mvgkwargs[keymatch["distkey"]]
                    dictionary[key] = cls(**mgkwargs)
                else:
                    try:
                        dictionary[key] = cls.from_repr(args)
                    except TypeError as e:
                        raise TypeError(
                            "Unable to parse prior, bad entry: {} "
                            "= {}. Error message {}".format(key, val, e)
                        )
            elif isinstance(val, dict):
                try:
                    _class = getattr(
                        import_module(val.get("__module__", "none")),
                        val.get("__name__", "none"),
                    )
                    dictionary[key] = _class(**val.get("kwargs", dict()))
                except ImportError:
                    logger.debug(
                        "Cannot import prior module {}.{}".format(
                            val.get("__module__", "none"), val.get("__name__", "none")
                        )
                    )
                    logger.warning(
                        "Cannot convert {} into a prior object. "
                        "Leaving as dictionary.".format(key)
                    )
                    continue
            else:
                raise TypeError(
                    "Unable to parse prior, bad entry: {} "
                    "= {} of type {}".format(key, val, type(val))
                )
        self.update(dictionary)

    def convert_floats_to_delta_functions(self):
        """Convert all float parameters to delta functions"""
        for key in self:
            if isinstance(self[key], Prior):
                continue
            elif isinstance(self[key], float) or isinstance(self[key], int):
                self[key] = DeltaFunction(self[key])
                logger.debug("{} converted to delta function prior.".format(key))
            else:
                logger.debug(
                    "{} cannot be converted to delta function prior.".format(key)
                )

    def fill_priors(self, likelihood, default_priors_file=None):
        """
        Fill dictionary of priors based on required parameters of likelihood

        Any floats in prior will be converted to delta function prior. Any
        required, non-specified parameters will use the default.

        Note: if `likelihood` has `non_standard_sampling_parameter_keys`, then
        this will set-up default priors for those as well.

        Parameters
        ==========
        likelihood: bilby.likelihood.GravitationalWaveTransient instance
            Used to infer the set of parameters to fill the prior with
        default_priors_file: str, optional
            If given, a file containing the default priors.


        Returns
        =======
        prior: dict
            The filled prior dictionary

        """

        self.convert_floats_to_delta_functions()

        missing_keys = set(likelihood.parameters) - set(self.keys())

        for missing_key in missing_keys:
            if not self.test_redundancy(missing_key):
                default_prior = create_default_prior(missing_key, default_priors_file)
                if default_prior is None:
                    set_val = likelihood.parameters[missing_key]
                    logger.warning(
                        "Parameter {} has no default prior and is set to {}, this"
                        " will not be sampled and may cause an error.".format(
                            missing_key, set_val
                        )
                    )
                else:
                    self[missing_key] = default_prior

        for key in self:
            self.test_redundancy(key)

    def sample(self, size=None):
        """Draw samples from the prior set

        Parameters
        ==========
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        =======
        dict: Dictionary of the samples
        """
        return self.sample_subset_constrained(keys=list(self.keys()), size=size)

    def sample_subset_constrained_as_array(self, keys=iter([]), size=None):
        """Return an array of samples

        Parameters
        ==========
        keys: list
            A list of keys to sample in
        size: int
            The number of samples to draw

        Returns
        =======
        array: array_like
            An array of shape (len(key), size) of the samples (ordered by keys)
        """
        samples_dict = self.sample_subset_constrained(keys=keys, size=size)
        samples_dict = {key: np.atleast_1d(val) for key, val in samples_dict.items()}
        samples_list = [samples_dict[key] for key in keys]
        return np.array(samples_list)

    def sample_subset(self, keys=iter([]), size=None):
        """Draw samples from the prior set for parameters which are not a DeltaFunction

        Parameters
        ==========
        keys: list
            List of prior keys to draw samples from
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        =======
        dict: Dictionary of the drawn samples
        """
        self.convert_floats_to_delta_functions()
        samples = dict()
        for key in keys:
            if isinstance(self[key], Constraint):
                continue
            elif isinstance(self[key], Prior):
                samples[key] = self[key].sample(size=size)
            else:
                logger.debug("{} not a known prior.".format(key))
        return samples

    @property
    def non_fixed_keys(self):
        keys = self.keys()
        keys = [k for k in keys if isinstance(self[k], Prior)]
        keys = [k for k in keys if self[k].is_fixed is False]
        keys = [k for k in keys if k not in self.constraint_keys]
        return keys

    @property
    def fixed_keys(self):
        return [
            k for k, p in self.items() if (p.is_fixed and k not in self.constraint_keys)
        ]

    @property
    def constraint_keys(self):
        return [k for k, p in self.items() if isinstance(p, Constraint)]

    def sample_subset_constrained(self, keys=iter([]), size=None):
        if size is None or size == 1:
            while True:
                sample = self.sample_subset(keys=keys, size=size)
                if self.evaluate_constraints(sample):
                    return sample
        else:
            needed = np.prod(size)
            for key in keys.copy():
                if isinstance(self[key], Constraint):
                    del keys[keys.index(key)]
            all_samples = {key: np.array([]) for key in keys}
            _first_key = list(all_samples.keys())[0]
            while len(all_samples[_first_key]) < needed:
                samples = self.sample_subset(keys=keys, size=needed)
                keep = np.array(self.evaluate_constraints(samples), dtype=bool)
                for key in keys:
                    all_samples[key] = np.hstack(
                        [all_samples[key], samples[key][keep].flatten()]
                    )
            all_samples = {
                key: np.reshape(all_samples[key][:needed], size) for key in keys
            }
            return all_samples

    def normalize_constraint_factor(
        self, keys, min_accept=10000, sampling_chunk=50000, nrepeats=10
    ):
        if keys in self._cached_normalizations.keys():
            return self._cached_normalizations[keys]
        else:
            factor_estimates = [
                self._estimate_normalization(keys, min_accept, sampling_chunk)
                for _ in range(nrepeats)
            ]
            factor = np.mean(factor_estimates)
            if np.std(factor_estimates) > 0:
                decimals = int(-np.floor(np.log10(3 * np.std(factor_estimates))))
                factor_rounded = np.round(factor, decimals)
            else:
                factor_rounded = factor
            self._cached_normalizations[keys] = factor_rounded
            return factor_rounded

    def _estimate_normalization(self, keys, min_accept, sampling_chunk):
        samples = self.sample_subset(keys=keys, size=sampling_chunk)
        keep = np.atleast_1d(self.evaluate_constraints(samples))
        if len(keep) == 1:
            self._cached_normalizations[keys] = 1
            return 1
        all_samples = {key: np.array([]) for key in keys}
        while np.count_nonzero(keep) < min_accept:
            samples = self.sample_subset(keys=keys, size=sampling_chunk)
            for key in samples:
                all_samples[key] = np.hstack([all_samples[key], samples[key].flatten()])
            keep = np.array(self.evaluate_constraints(all_samples), dtype=bool)
        factor = len(keep) / np.count_nonzero(keep)
        return factor

    def prob(self, sample, **kwargs):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which we want to have the probability of
        kwargs:
            The keyword arguments are passed directly to `np.prod`

        Returns
        =======
        float: Joint probability of all individual sample probabilities

        """
        prob = np.prod([self[key].prob(sample[key]) for key in sample], **kwargs)

        return self.check_prob(sample, prob)

    def check_prob(self, sample, prob):
        ratio = self.normalize_constraint_factor(tuple(sample.keys()))
        if np.all(prob == 0.0):
            return prob * ratio
        else:
            if isinstance(prob, float):
                if self.evaluate_constraints(sample):
                    return prob * ratio
                else:
                    return 0.0
            else:
                constrained_prob = np.zeros_like(prob)
                keep = np.array(self.evaluate_constraints(sample), dtype=bool)
                constrained_prob[keep] = prob[keep] * ratio
                return constrained_prob

    def ln_prob(self, sample, axis=None):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which to calculate the log probability
        axis: None or int
            Axis along which the summation is performed

        Returns
        =======
        float or ndarray:
            Joint log probability of all the individual sample probabilities

        """
        ln_prob = np.sum([self[key].ln_prob(sample[key]) for key in sample], axis=axis)
        return self.check_ln_prob(sample, ln_prob)

    def check_ln_prob(self, sample, ln_prob):
        ratio = self.normalize_constraint_factor(tuple(sample.keys()))
        if np.all(np.isinf(ln_prob)):
            return ln_prob
        else:
            if isinstance(ln_prob, float):
                if self.evaluate_constraints(sample):
                    return ln_prob + np.log(ratio)
                else:
                    return -np.inf
            else:
                constrained_ln_prob = -np.inf * np.ones_like(ln_prob)
                keep = np.array(self.evaluate_constraints(sample), dtype=bool)
                constrained_ln_prob[keep] = ln_prob[keep] + np.log(ratio)
                return constrained_ln_prob

    def cdf(self, sample):
        """Evaluate the cumulative distribution function at the provided points

        Parameters
        ----------
        sample: dict, pandas.DataFrame
            Dictionary of the samples of which to calculate the CDF

        Returns
        -------
        dict, pandas.DataFrame: Dictionary containing the CDF values

        """
        return sample.__class__(
            {key: self[key].cdf(sample) for key, sample in sample.items()}
        )

    def rescale(self, keys, theta):
        """Rescale samples from unit cube to prior

        Parameters
        ==========
        keys: list
            List of prior keys to be rescaled
        theta: list
            List of randomly drawn values on a unit cube associated with the prior keys

        Returns
        =======
        list: List of floats containing the rescaled sample
        """
        from matplotlib.cbook import flatten

        return list(
            flatten([self[key].rescale(sample) for key, sample in zip(keys, theta)])
        )

    def test_redundancy(self, key, disable_logging=False):
        """Empty redundancy test, should be overwritten in subclasses"""
        return False

    def test_has_redundant_keys(self):
        """
        Test whether there are redundant keys in self.

        Returns
        =======
        bool: Whether there are redundancies or not
        """
        redundant = False
        for key in self:
            if isinstance(self[key], Constraint):
                continue
            temp = self.copy()
            del temp[key]
            if temp.test_redundancy(key, disable_logging=True):
                logger.warning(
                    "{} is a redundant key in this {}.".format(
                        key, self.__class__.__name__
                    )
                )
                redundant = True
        return redundant

    def copy(self):
        """
        We have to overwrite the copy method as it fails due to the presence of
        defaults.
        """
        return self.__class__(dictionary=dict(self))


class PriorDictException(Exception):
    """General base class for all prior dict exceptions"""


class ConditionalPriorDict(PriorDict):
    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        """

        Parameters
        ==========
        dictionary: dict
            See parent class
        filename: str
            See parent class
        """
        self._conditional_keys = []
        self._unconditional_keys = []
        self._rescale_keys = []
        self._rescale_indexes = []
        self._least_recently_rescaled_keys = []
        super(ConditionalPriorDict, self).__init__(
            dictionary=dictionary,
            filename=filename,
            conversion_function=conversion_function,
        )
        self._resolved = False
        self._resolve_conditions()

    def _resolve_conditions(self):
        """
        Resolves how priors depend on each other and automatically
        sorts them into the right order.
        1. All unconditional priors are put in front in arbitrary order
        2. We loop through all the unsorted conditional priors to find
        which one can go next
        3. We repeat step 2 len(self) number of times to make sure that
        all conditional priors will be sorted in order
        4. We set the `self._resolved` flag to True if all conditional
        priors were added in the right order
        """
        self._unconditional_keys = [
            key for key in self.keys() if not hasattr(self[key], "condition_func")
        ]
        conditional_keys_unsorted = [
            key for key in self.keys() if hasattr(self[key], "condition_func")
        ]
        self._conditional_keys = []
        for _ in range(len(self)):
            for key in conditional_keys_unsorted[:]:
                if self._check_conditions_resolved(key, self.sorted_keys):
                    self._conditional_keys.append(key)
                    conditional_keys_unsorted.remove(key)

        self._resolved = True
        if len(conditional_keys_unsorted) != 0:
            self._resolved = False

    def _check_conditions_resolved(self, key, sampled_keys):
        """Checks if all required variables have already been sampled so we can sample this key"""
        conditions_resolved = True
        for k in self[key].required_variables:
            if k not in sampled_keys:
                conditions_resolved = False
        return conditions_resolved

    def sample_subset(self, keys=iter([]), size=None):
        self.convert_floats_to_delta_functions()
        add_delta_keys = [
            key
            for key in self.keys()
            if key not in keys and isinstance(self[key], DeltaFunction)
        ]
        use_keys = add_delta_keys + list(keys)
        subset_dict = ConditionalPriorDict({key: self[key] for key in use_keys})
        if not subset_dict._resolved:
            raise IllegalConditionsException(
                "The current set of priors contains unresolvable conditions."
            )
        samples = dict()
        for key in subset_dict.sorted_keys:
            if key not in keys or isinstance(self[key], Constraint):
                continue
            if isinstance(self[key], Prior):
                try:
                    samples[key] = subset_dict[key].sample(
                        size=size, **subset_dict.get_required_variables(key)
                    )
                except ValueError:
                    # Some prior classes can not handle an array of conditional parameters (e.g. alpha for PowerLaw)
                    # If that is the case, we sample each sample individually.
                    required_variables = subset_dict.get_required_variables(key)
                    samples[key] = np.zeros(size)
                    for i in range(size):
                        rvars = {
                            key: value[i] for key, value in required_variables.items()
                        }
                        samples[key][i] = subset_dict[key].sample(**rvars)
            else:
                logger.debug("{} not a known prior.".format(key))
        return samples

    def get_required_variables(self, key):
        """Returns the required variables to sample a given conditional key.

        Parameters
        ==========
        key : str
            Name of the key that we want to know the required variables for

        Returns
        =======
        dict: key/value pairs of the required variables
        """
        return {
            k: self[k].least_recently_sampled
            for k in getattr(self[key], "required_variables", [])
        }

    def prob(self, sample, **kwargs):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which we want to have the probability of
        kwargs:
            The keyword arguments are passed directly to `np.prod`

        Returns
        =======
        float: Joint probability of all individual sample probabilities

        """
        self._prepare_evaluation(*zip(*sample.items()))
        res = [
            self[key].prob(sample[key], **self.get_required_variables(key))
            for key in sample
        ]
        prob = np.prod(res, **kwargs)
        return self.check_prob(sample, prob)

    def ln_prob(self, sample, axis=None):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which we want to have the log probability of
        axis: Union[None, int]
            Axis along which the summation is performed

        Returns
        =======
        float: Joint log probability of all the individual sample probabilities

        """
        self._prepare_evaluation(*zip(*sample.items()))
        res = [
            self[key].ln_prob(sample[key], **self.get_required_variables(key))
            for key in sample
        ]
        ln_prob = np.sum(res, axis=axis)
        return self.check_ln_prob(sample, ln_prob)

    def cdf(self, sample):
        self._prepare_evaluation(*zip(*sample.items()))
        res = {
            key: self[key].cdf(sample[key], **self.get_required_variables(key))
            for key in sample
        }
        return sample.__class__(res)

    def rescale(self, keys, theta):
        """Rescale samples from unit cube to prior

        Parameters
        ==========
        keys: list
            List of prior keys to be rescaled
        theta: list
            List of randomly drawn values on a unit cube associated with the prior keys

        Returns
        =======
        list: List of floats containing the rescaled sample
        """
        from matplotlib.cbook import flatten

        keys = list(keys)
        theta = list(theta)
        self._check_resolved()
        self._update_rescale_keys(keys)
        result = dict()
        for key, index in zip(
            self.sorted_keys_without_fixed_parameters, self._rescale_indexes
        ):
            result[key] = self[key].rescale(
                theta[index], **self.get_required_variables(key)
            )
            self[key].least_recently_sampled = result[key]
        return list(flatten([result[key] for key in keys]))

    def _update_rescale_keys(self, keys):
        if not keys == self._least_recently_rescaled_keys:
            self._rescale_indexes = [
                keys.index(element)
                for element in self.sorted_keys_without_fixed_parameters
            ]
            self._least_recently_rescaled_keys = keys

    def _prepare_evaluation(self, keys, theta):
        self._check_resolved()
        for key, value in zip(keys, theta):
            self[key].least_recently_sampled = value

    def _check_resolved(self):
        if not self._resolved:
            raise IllegalConditionsException(
                "The current set of priors contains unresolveable conditions."
            )

    @property
    def conditional_keys(self):
        return self._conditional_keys

    @property
    def unconditional_keys(self):
        return self._unconditional_keys

    @property
    def sorted_keys(self):
        return self.unconditional_keys + self.conditional_keys

    @property
    def sorted_keys_without_fixed_parameters(self):
        return [
            key
            for key in self.sorted_keys
            if not isinstance(self[key], (DeltaFunction, Constraint))
        ]

    def __setitem__(self, key, value):
        super(ConditionalPriorDict, self).__setitem__(key, value)
        self._resolve_conditions()

    def __delitem__(self, key):
        super(ConditionalPriorDict, self).__delitem__(key)
        self._resolve_conditions()


class DirichletPriorDict(ConditionalPriorDict):
    def __init__(self, n_dim=None, label="dirichlet_"):
        from .conditional import DirichletElement

        self.n_dim = n_dim
        self.label = label
        super(DirichletPriorDict, self).__init__(dictionary=dict())
        for ii in range(n_dim - 1):
            self[label + "{}".format(ii)] = DirichletElement(
                order=ii, n_dimensions=n_dim, label=label
            )

    def copy(self, **kwargs):
        return self.__class__(n_dim=self.n_dim, label=self.label)

    def _get_json_dict(self):
        total_dict = dict()
        total_dict["__prior_dict__"] = True
        total_dict["__module__"] = self.__module__
        total_dict["__name__"] = self.__class__.__name__
        total_dict["n_dim"] = self.n_dim
        total_dict["label"] = self.label
        return total_dict

    @classmethod
    def _get_from_json_dict(cls, prior_dict):
        try:
            cls == getattr(
                import_module(prior_dict["__module__"]), prior_dict["__name__"]
            )
        except ImportError:
            logger.debug(
                "Cannot import prior module {}.{}".format(
                    prior_dict["__module__"], prior_dict["__name__"]
                )
            )
        except KeyError:
            logger.debug("Cannot find module name to load")
        for key in ["__module__", "__name__", "__prior_dict__"]:
            if key in prior_dict:
                del prior_dict[key]
        obj = cls(**prior_dict)
        return obj


class ConditionalPriorDictException(PriorDictException):
    """General base class for all conditional prior dict exceptions"""


def create_default_prior(name, default_priors_file=None):
    """Make a default prior for a parameter with a known name.

    Parameters
    ==========
    name: str
        Parameter name
    default_priors_file: str, optional
        If given, a file containing the default priors.

    Returns
    =======
    prior: Prior
        Default prior distribution for that parameter, if unknown None is
        returned.
    """

    if default_priors_file is None:
        logger.debug("No prior file given.")
        prior = None
    else:
        default_priors = PriorDict(filename=default_priors_file)
        if name in default_priors.keys():
            prior = default_priors[name]
        else:
            logger.debug("No default prior found for variable {}.".format(name))
            prior = None
    return prior


class IllegalConditionsException(ConditionalPriorDictException):
    """Exception class to handle prior dicts that contain unresolvable conditions."""
