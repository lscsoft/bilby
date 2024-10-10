import datetime
import inspect
import json
import os
import shutil
from importlib import import_module
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

from .log import logger
from .introspection import infer_args_from_method


def check_directory_exists_and_if_not_mkdir(directory):
    """ Checks if the given directory exists and creates it if it does not exist

    Parameters
    ==========
    directory: str
        Name of the directory

    """
    Path(directory).mkdir(parents=True, exist_ok=True)


class BilbyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        from ..prior import BaseJointPriorDist, Prior, PriorDict
        from ...bilby_mcmc.proposals import ProposalCycle

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, PriorDict):
            return {"__prior_dict__": True, "content": obj._get_json_dict()}
        if isinstance(obj, (BaseJointPriorDist, Prior)):
            return {
                "__prior__": True,
                "__module__": obj.__module__,
                "__name__": obj.__class__.__name__,
                "kwargs": dict(obj.get_instantiation_dict()),
            }
        if isinstance(obj, ProposalCycle):
            return str(obj)
        try:
            from astropy import cosmology as cosmo, units

            if isinstance(obj, cosmo.FLRW):
                return encode_astropy_cosmology(obj)
            if isinstance(obj, units.Quantity):
                return encode_astropy_quantity(obj)
            if isinstance(obj, units.PrefixUnit):
                return str(obj)
        except ImportError:
            logger.debug("Cannot import astropy, cannot write cosmological priors")
        if isinstance(obj, np.ndarray):
            return {"__array__": True, "content": obj.tolist()}
        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        if isinstance(obj, pd.DataFrame):
            return {"__dataframe__": True, "content": obj.to_dict(orient="list")}
        if isinstance(obj, pd.Series):
            return {"__series__": True, "content": obj.to_dict()}
        if inspect.isfunction(obj):
            return {
                "__function__": True,
                "__module__": obj.__module__,
                "__name__": obj.__name__,
            }
        if inspect.isclass(obj):
            return {
                "__class__": True,
                "__module__": obj.__module__,
                "__name__": obj.__name__,
            }
        if isinstance(obj, (timedelta)):
            return {
                "__timedelta__": True,
                "__total_seconds__": obj.total_seconds()
            }
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def encode_astropy_cosmology(obj):
    cls_name = obj.__class__.__name__
    dct = {key: getattr(obj, key) for key in infer_args_from_method(obj.__init__)}
    dct["__cosmology__"] = True
    dct["__name__"] = cls_name
    return dct


def encode_astropy_quantity(dct):
    dct = dict(__astropy_quantity__=True, value=dct.value, unit=str(dct.unit))
    if isinstance(dct["value"], np.ndarray):
        dct["value"] = list(dct["value"])
    return dct


def decode_astropy_cosmology(dct):
    try:
        from astropy import cosmology as cosmo

        cosmo_cls = getattr(cosmo, dct["__name__"])
        del dct["__cosmology__"], dct["__name__"]
        return cosmo_cls(**dct)
    except ImportError:
        logger.debug(
            "Cannot import astropy, cosmological priors may not be " "properly loaded."
        )
        return dct


def decode_astropy_quantity(dct):
    try:
        from astropy import units

        if dct["value"] is None:
            return None
        else:
            del dct["__astropy_quantity__"]
            return units.Quantity(**dct)
    except ImportError:
        logger.debug(
            "Cannot import astropy, cosmological priors may not be " "properly loaded."
        )
        return dct


def load_json(filename, gzip):
    if gzip or os.path.splitext(filename)[1].lstrip(".") == "gz":
        import gzip

        with gzip.GzipFile(filename, "r") as file:
            json_str = file.read().decode("utf-8")
        dictionary = json.loads(json_str, object_hook=decode_bilby_json)
    else:
        with open(filename, "r") as file:
            dictionary = json.load(file, object_hook=decode_bilby_json)
    return dictionary


def decode_bilby_json(dct):
    if dct.get("__prior_dict__", False):
        cls = getattr(import_module(dct["__module__"]), dct["__name__"])
        obj = cls._get_from_json_dict(dct)
        return obj
    if dct.get("__prior__", False):
        try:
            cls = getattr(import_module(dct["__module__"]), dct["__name__"])
        except AttributeError:
            logger.warning(
                "Unknown prior class for parameter {}, defaulting to base Prior object".format(
                    dct["kwargs"]["name"]
                )
            )
            from ..prior import Prior

            for key in list(dct["kwargs"].keys()):
                if key not in ["name", "latex_label", "unit", "minimum", "maximum", "boundary"]:
                    dct["kwargs"].pop(key)
            cls = Prior
        obj = cls(**dct["kwargs"])
        return obj
    if dct.get("__cosmology__", False):
        return decode_astropy_cosmology(dct)
    if dct.get("__astropy_quantity__", False):
        return decode_astropy_quantity(dct)
    if dct.get("__array__", False):
        return np.asarray(dct["content"])
    if dct.get("__complex__", False):
        return complex(dct["real"], dct["imag"])
    if dct.get("__dataframe__", False):
        return pd.DataFrame(dct["content"])
    if dct.get("__series__", False):
        return pd.Series(dct["content"])
    if dct.get("__function__", False) or dct.get("__class__", False):
        default = ".".join([dct["__module__"], dct["__name__"]])
        return getattr(import_module(dct["__module__"]), dct["__name__"], default)
    if dct.get("__timedelta__", False):
        return timedelta(seconds=dct["__total_seconds__"])
    return dct


def recursively_decode_bilby_json(dct):
    """
    Recursively call `bilby_decode_json`

    Parameters
    ----------
    dct: dict
        The dictionary to decode

    Returns
    -------
    dct: dict
        The original dictionary with all the elements decode if possible
    """
    dct = decode_bilby_json(dct)
    if isinstance(dct, dict):
        for key in dct:
            if isinstance(dct[key], dict):
                dct[key] = recursively_decode_bilby_json(dct[key])
    return dct


def decode_from_hdf5(item):
    """
    Decode an item from HDF5 format to python type.

    This currently just converts __none__ to None and some arrays to lists

    .. versionadded:: 1.0.0

    Parameters
    ----------
    item: object
        Item to be decoded

    Returns
    -------
    output: object
        Converted input item
    """
    if isinstance(item, str) and item == "__none__":
        output = None
    elif isinstance(item, bytes) and item == b"__none__":
        output = None
    elif isinstance(item, (bytes, bytearray)):
        output = item.decode()
    elif isinstance(item, np.ndarray):
        if item.size == 0:
            output = item
        elif "|S" in str(item.dtype) or isinstance(item[0], bytes):
            output = [it.decode() for it in item]
        else:
            output = item
    elif isinstance(item, np.bool_):
        output = bool(item)
    else:
        output = item
    return output


def encode_for_hdf5(key, item):
    """
    Encode an item to a HDF5 saveable format.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    item: object
        Object to be encoded, specific options are provided for Bilby types

    Returns
    -------
    output: object
        Input item converted into HDF5 saveable format
    """
    from ..prior.dict import PriorDict

    if isinstance(item, np.int_):
        item = int(item)
    elif isinstance(item, np.float64):
        item = float(item)
    elif isinstance(item, np.complex128):
        item = complex(item)
    if isinstance(item, np.ndarray):
        # Numpy's wide unicode strings are not supported by hdf5
        if item.dtype.kind == 'U':
            logger.debug(f'converting dtype {item.dtype} for hdf5')
            item = np.array(item, dtype='S')
    if isinstance(item, (np.ndarray, int, float, complex, str, bytes)):
        output = item
    elif item is None:
        output = "__none__"
    elif isinstance(item, list):
        item_array = np.array(item)
        if len(item) == 0:
            output = item
        elif np.issubdtype(item_array.dtype, np.number):
            output = np.array(item)
        elif issubclass(item_array.dtype.type, str) or None in item:
            output = list()
            for value in item:
                if isinstance(value, str):
                    output.append(value.encode("utf-8"))
                elif isinstance(value, bytes):
                    output.append(value)
                elif value is None:
                    output.append(b"__none__")
                else:
                    output.append(str(value).encode("utf-8"))
        else:
            raise ValueError(f'Cannot save {key}: {type(item)} type')
    elif isinstance(item, PriorDict):
        output = json.dumps(item._get_json_dict())
    elif isinstance(item, pd.DataFrame):
        output = item.to_dict(orient="list")
    elif inspect.isfunction(item) or inspect.isclass(item):
        output = dict(
            __module__=item.__module__, __name__=item.__name__, __class__=True
        )
    elif isinstance(item, dict):
        output = item.copy()
    elif isinstance(item, tuple):
        output = {str(ii): elem for ii, elem in enumerate(item)}
    elif isinstance(item, datetime.timedelta):
        output = item.total_seconds()
    else:
        raise ValueError(f'Cannot save {key}: {type(item)} type')
    return output


def recursively_load_dict_contents_from_group(h5file, path):
    """
    Recursively load a HDF5 file into a dictionary

    .. versionadded:: 1.1.0

    Parameters
    ----------
    h5file: h5py.File
        Open h5py file object
    path: str
        Path within the HDF5 file

    Returns
    -------
    output: dict
        The contents of the HDF5 file unpacked into the dictionary.
    """
    import h5py

    output = dict()
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            output[key] = decode_from_hdf5(item[()])
        elif isinstance(item, h5py.Group):
            output[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return output


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Recursively save a dictionary to a HDF5 group

    .. versionadded:: 1.1.0

    Parameters
    ----------
    h5file: h5py.File
        Open HDF5 file
    path: str
        Path inside the HDF5 file
    dic: dict
        The dictionary containing the data
    """
    for key, item in dic.items():
        item = encode_for_hdf5(key, item)
        if isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
        else:
            h5file[path + key] = item


def safe_file_dump(data, filename, module):
    """ Safely dump data to a .pickle file

    Parameters
    ==========
    data:
        data to dump
    filename: str
        The file to dump to
    module: pickle, dill, str
        The python module to use. If a string, the module will be imported
    """
    if isinstance(module, str):
        module = import_module(module)
    temp_filename = filename + ".temp"
    with open(temp_filename, "wb") as file:
        module.dump(data, file)
    shutil.move(temp_filename, filename)


def move_old_file(filename, overwrite=False):
    """ Moves or removes an old file.

    Parameters
    ==========
    filename: str
        Name of the file to be move
    overwrite: bool, optional
        Whether or not to remove the file or to change the name
        to filename + '.old'
    """
    if os.path.isfile(filename):
        if overwrite:
            logger.debug("Removing existing file {}".format(filename))
            os.remove(filename)
        else:
            logger.debug(
                "Renaming existing file {} to {}.old".format(filename, filename)
            )
            shutil.move(filename, filename + ".old")
    logger.debug("Saving result to {}".format(filename))


def safe_save_figure(fig, filename, **kwargs):
    check_directory_exists_and_if_not_mkdir(os.path.dirname(filename))
    from matplotlib import rcParams

    try:
        fig.savefig(fname=filename, **kwargs)
    except RuntimeError:
        logger.debug("Failed to save plot with tex labels turning off tex.")
        rcParams["text.usetex"] = False
        fig.savefig(fname=filename, **kwargs)
