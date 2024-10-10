"""
Wrappers to :code:`Astropy` functionality for specifying the cosmology to use.
"""

DEFAULT_COSMOLOGY = None
COSMOLOGY = [None, str(None)]


def _set_default_cosmology():
    from astropy import cosmology as cosmo
    global DEFAULT_COSMOLOGY, COSMOLOGY
    if DEFAULT_COSMOLOGY is None:
        DEFAULT_COSMOLOGY = cosmo.Planck15
        COSMOLOGY = [DEFAULT_COSMOLOGY, DEFAULT_COSMOLOGY.name]


def get_cosmology(cosmology=None):
    """
    Get an instance of a astropy.cosmology.FLRW subclass.

    To avoid repeatedly instantiating the same class, test if it is the same
    as the last used cosmology.

    Parameters
    ==========
    cosmology: astropy.cosmology.FLRW, str, dict
        Description of cosmology, one of:
            None - Use DEFAULT_COSMOLOGY
            Instance of astropy.cosmology.FLRW subclass
            String with name of known Astropy cosmology, e.g., "Planck13"

    Returns
    =======
    cosmo: astropy.cosmology.FLRW
        Cosmology instance
    """
    from astropy import cosmology as cosmo
    _set_default_cosmology()
    if cosmology is None:
        cosmology = DEFAULT_COSMOLOGY
    elif isinstance(cosmology, cosmo.FLRW):
        cosmology = cosmology
    elif isinstance(cosmology, str):
        cosmology = getattr(cosmo, cosmology)
    elif isinstance(cosmology, dict):
        if 'Ode0' in cosmology.keys():
            if 'w0' in cosmology.keys():
                cosmology = cosmo.wCDM(**cosmology)
            else:
                cosmology = cosmo.LambdaCDM(**cosmology)
        else:
            cosmology = cosmo.FlatLambdaCDM(**cosmology)
    return cosmology


def set_cosmology(cosmology=None):
    """
    Set an instance of a astropy.cosmology.FLRW subclass as the default
    cosmology.

    Parameters
    ==========
    cosmology: astropy.cosmology.FLRW, str, dict
        Description of cosmology, one of:
            None - Use DEFAULT_COSMOLOGY
            Instance of astropy.cosmology.FLRW subclass
            String with name of known Astropy cosmology, e.g., "Planck13"
            Dictionary with arguments required to instantiate the cosmology
            class.
    """
    cosmology = get_cosmology(cosmology)
    COSMOLOGY[0] = cosmology
    if cosmology.name is not None:
        COSMOLOGY[1] = cosmology.name
    else:
        COSMOLOGY[1] = repr(cosmology)


def z_at_value(func, fval, **kwargs):
    """
    Wrapped version of :code:`astropy.cosmology.z_at_value` to return float
    rather than an :code:`astropy Quantity` as returned for :code:`astropy>=5`.

    See https://docs.astropy.org/en/stable/api/astropy.cosmology.z_at_value.html#astropy.cosmology.z_at_value
    for detailed documentation.
    """
    from astropy.cosmology import z_at_value
    return z_at_value(func=func, fval=fval, **kwargs).value
