import matplotlib

matplotlib.use("Agg")  # noqa

import glob
import importlib.util
import unittest
import os
import parameterized
import pytest
import shutil
import logging

import bilby.core.utils

bilby.core.utils.command_line_args.clean = True
core_examples = glob.glob("examples/core_examples/*.py")
core_examples += glob.glob("examples/core_examples/*/*.py")
core_args = [(fname.split("/")[-1][:-3], fname) for fname in core_examples]

gw_examples = [
    "examples/gw_examples/injection_examples/fast_tutorial.py",
    "examples/gw_examples/data_examples/GW150914.py",
]
gw_args = [(fname.split("/")[-1][:-3], fname) for fname in gw_examples]


def _execute_file(name, fname):
    dname, fname = os.path.split(fname)
    old_directory = os.getcwd()
    os.chdir(dname)
    spec = importlib.util.spec_from_file_location(name, fname)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    os.chdir(old_directory)


class ExampleTest(unittest.TestCase):
    outdir = "outdir"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(dir_path, os.path.pardir))

    def setUp(self):
        self.init_dir = os.getcwd()

    def tearDown(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning("{} not removed after tests".format(self.outdir))
        os.chdir(self.init_dir)

    @classmethod
    def setUpClass(cls):
        if os.path.isdir(cls.outdir):
            try:
                shutil.rmtree(cls.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(cls.outdir))

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.outdir):
            try:
                shutil.rmtree(cls.outdir)
            except OSError:
                logging.warning("{} not removed prior to tests".format(cls.outdir))

    @parameterized.parameterized.expand(core_args)
    def test_core_examples(self, name, fname):
        """ Loop over examples to check they run """
        bilby.core.utils.command_line_args.bilby_test_mode = False
        ignore = ["15d_gaussian"]
        if any([item in fname for item in ignore]):
            pytest.skip()
        _execute_file(name, fname)

    @parameterized.parameterized.expand(gw_args)
    @pytest.mark.gw
    def test_gw_examples(self, name, fname):
        """ Loop over examples to check they run """
        bilby.core.utils.command_line_args.bilby_test_mode = True
        _execute_file(name, fname)


if __name__ == "__main__":
    unittest.main()
