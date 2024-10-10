"""
Bilby
=====

Bilby: a user-friendly Bayesian inference library.

The aim of bilby is to provide a user-friendly interface to perform parameter
estimation. It is primarily designed and built for inference of compact
binary coalescence events in interferometric data, but it can also be used for
more general problems.

The code, and many examples are hosted at https://github.com/bilby-dev/bilby.
For installation instructions see
https://bilby-dev.github.io/bilby/installation.html.

"""


import sys

from . import core, gw, hyper

from .core import utils, likelihood, prior, result, sampler
from .core.sampler import run_sampler
from .core.likelihood import Likelihood
from .core.result import read_in_result, read_in_result_list

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = 'unknown'


if sys.version_info < (3,):
    raise ImportError(
"""You are running bilby >= 0.6.4 on Python 2

Bilby 0.6.4 and above are no longer compatible with Python 2, and you still
ended up with this version installed. That's unfortunate; sorry about that.
It should not have happened. Make sure you have pip >= 9.0 to avoid this kind
of issue, as well as setuptools >= 24.2:

 $ pip install pip setuptools --upgrade

Your choices:

- Upgrade to Python 3.

- Install an older version of bilby:

 $ pip install 'bilby<0.6.4'

""")
