import os
import warnings

import numpy as np

from .base_sampler import NestedSampler, signal_wrapper


class PyPolyChord(NestedSampler):

    """
    Bilby wrapper of PyPolyChord
    https://arxiv.org/abs/1506.00171

    .. warning::
        The PyPolyChord sampler interface in bilby is deprecated and will be
        removed in future release. Please use the :code`pypolychord-bilby`
        sampler plugin instead: https://github.com/bilby-dev/pypolychord-bilby

    PolyChordLite is available at:
    https://github.com/PolyChord/PolyChordLite

    Follow the installation instructions at their github page.

    Keyword arguments will be passed into `pypolychord.run_polychord` into the `settings`
    argument. See the PolyChord documentation for what all of those mean.

    To see what the keyword arguments are for, see the docstring of PyPolyChordSettings
    """

    msg = (
        "The PyPolyChord sampler interface in bilby is deprecated and will"
        " be removed in future release. Please use the `pypolychord-bilby`"
        "sampler plugin instead: https://github.com/bilby-dev/pypolychord-bilby."
    )
    warnings.warn(msg, FutureWarning)

    sampler_name = "pypolychord"
    default_kwargs = dict(
        use_polychord_defaults=False,
        nlive=None,
        num_repeats=None,
        nprior=-1,
        do_clustering=True,
        feedback=1,
        precision_criterion=0.001,
        logzero=-1e30,
        max_ndead=-1,
        boost_posterior=0.0,
        posteriors=True,
        equals=True,
        cluster_posteriors=True,
        write_resume=True,
        write_paramnames=False,
        read_resume=True,
        write_stats=True,
        write_live=True,
        write_dead=True,
        write_prior=True,
        compression_factor=np.exp(-1),
        base_dir="outdir",
        file_root="polychord",
        seed=-1,
        grade_dims=None,
        grade_frac=None,
        nlives={},
    )
    hard_exit = True
    sampling_seed_key = "seed"

    @signal_wrapper
    def run_sampler(self):
        import pypolychord
        from pypolychord.settings import PolyChordSettings

        if self.kwargs["use_polychord_defaults"]:
            settings = PolyChordSettings(
                nDims=self.ndim,
                nDerived=self.ndim,
                base_dir=self._sample_file_directory,
                file_root=self.label,
            )
        else:
            self._setup_dynamic_defaults()
            pc_kwargs = self.kwargs.copy()
            pc_kwargs["base_dir"] = self._sample_file_directory
            pc_kwargs["file_root"] = self.label
            pc_kwargs.pop("use_polychord_defaults")
            settings = PolyChordSettings(
                nDims=self.ndim, nDerived=self.ndim, **pc_kwargs
            )
        self._verify_kwargs_against_default_kwargs()
        out = pypolychord.run_polychord(
            loglikelihood=self.log_likelihood,
            nDims=self.ndim,
            nDerived=self.ndim,
            settings=settings,
            prior=self.prior_transform,
        )
        self.result.log_evidence = out.logZ
        self.result.log_evidence_err = out.logZerr
        log_likelihoods, physical_parameters = self._read_sample_file()
        self.result.log_likelihood_evaluations = log_likelihoods
        self.result.samples = physical_parameters
        self.calc_likelihood_count()
        return self.result

    def _setup_dynamic_defaults(self):
        """Sets up some interdependent default argument if none are given by the user"""
        if not self.kwargs["grade_dims"]:
            self.kwargs["grade_dims"] = [self.ndim]
        if not self.kwargs["grade_frac"]:
            self.kwargs["grade_frac"] = [1.0] * len(self.kwargs["grade_dims"])
        if not self.kwargs["nlive"]:
            self.kwargs["nlive"] = self.ndim * 25
        if not self.kwargs["num_repeats"]:
            self.kwargs["num_repeats"] = self.ndim * 5

    def _translate_kwargs(self, kwargs):
        kwargs = super()._translate_kwargs(kwargs)
        if "nlive" not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["nlive"] = kwargs.pop(equiv)

    def log_likelihood(self, theta):
        """Overrides the log_likelihood so that PolyChord understands it"""
        return super(PyPolyChord, self).log_likelihood(theta), theta

    def _read_sample_file(self):
        """
        This method reads out the _equal_weights.txt file that polychord produces.
        The first column is omitted since it is just composed of 1s, i.e. the equal weights/
        The second column are the log likelihoods, the remaining columns are the physical parameters

        Returns
        =======
        array_like, array_like: The log_likelihoods and the associated parameters

        """
        sample_file = (
            self._sample_file_directory + "/" + self.label + "_equal_weights.txt"
        )
        samples = np.loadtxt(sample_file)
        log_likelihoods = -0.5 * samples[:, 1]
        physical_parameters = samples[:, -self.ndim :]
        return log_likelihoods, physical_parameters

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        """Get lists of the expected outputs directories and files.

        These are used by :code:`bilby_pipe` when transferring files via HTCondor.

        Parameters
        ----------
        outdir : str
            The output directory.
        label : str
            Ignored for pypolychord.

        Returns
        -------
        list
            List of file names. This will always be empty for pypolychord.
        list
            List of directory names.
        """
        return [], [os.path.join(outdir, "chains", "")]

    @property
    def _sample_file_directory(self):
        return self.outdir + "/chains"
