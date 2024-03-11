import os
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

from .tov_solver import IntegrateTOV
from ...core import utils

C_SI = utils.speed_of_light  # m/s
C_CGS = C_SI * 100.
G_SI = utils.gravitational_constant  # m^3 kg^-1 s^-2
MSUN_SI = utils.solar_mass  # Kg

# Stores conversions from geometerized to cgs or si unit systems
conversion_dict = {'pressure': {'cgs': C_SI ** 4. / G_SI * 10., 'si': C_SI ** 4. / G_SI, 'geom': 1.},
                   'energy_density': {'cgs': C_SI ** 4. / G_SI * 10., 'si': C_SI ** 4. / G_SI, 'geom': 1.},
                   'density': {'cgs': C_SI ** 2. / G_SI / 1000., 'si': C_SI ** 2. / G_SI, 'geom': 1.},
                   'pseudo_enthalpy': {'dimensionless': 1.},
                   'mass': {'g': C_SI ** 2. / G_SI * 1000, 'kg': C_SI ** 2. / G_SI, 'geom': 1.,
                            'm_sol': C_SI ** 2. / G_SI / MSUN_SI},
                   'radius': {'cm': 100., 'm': 1., 'km': .001},
                   'tidal_deformability': {'geom': 1.}}


# construct dictionary of pre-shipped EOS pressure denstity table
path_to_eos_tables = os.path.join(os.path.dirname(__file__), 'eos_tables')
list_of_eos_tables = os.listdir(path_to_eos_tables)
valid_eos_files = [i for i in list_of_eos_tables if 'LAL' in i]
valid_eos_file_paths = [os.path.join(path_to_eos_tables, filename) for filename in valid_eos_files]
valid_eos_names = [i.split('_', maxsplit=1)[-1].strip('.dat') for i in valid_eos_files]
valid_eos_dict = dict(zip(valid_eos_names, valid_eos_file_paths))


class TabularEOS(object):
    """
    Given a valid eos input format, such as 2-D array, an ascii file, or a string, parse, and interpolate

    Parameters
    ==========
    eos: (numpy.ndarray, str, ASCII TABLE)
        if `numpy.ndarray` then user supplied pressure-density 2D numpy array.
        if `str` then given a valid eos name, relevant preshipped ASCII table will be loaded
        if ASCII TABLE then given viable file extensions, which include .txt,.dat, etc (np.loadtxt used),
        read in pressure density from file.
    sampling_flag: bool
        Do you plan on sampling the parameterized EOS? Highly recommended. Defaults to False.
    warning_flag: bool
        Keeps track of status of various physical checks on EoS.

    Attributes
    ==========
    msg: str
        Human readable string describing the exception.
    code: int
        Exception error code.
    """

    def __init__(self, eos, sampling_flag=False, warning_flag=False):
        from scipy.integrate import cumulative_trapezoid

        self.sampling_flag = sampling_flag
        self.warning_flag = warning_flag

        if isinstance(eos, str):
            if eos in valid_eos_dict.keys():
                table = np.loadtxt(valid_eos_dict[eos])
            else:
                table = np.loadtxt(eos)
        elif isinstance(eos, np.ndarray):
            table = eos
        else:
            raise ValueError("eos provided is invalid type please supply a str name, str path to ASCII file, "
                             "or a numpy array")

        table = self.__remove_leading_zero(table)

        # all have units of m^-2
        self.pressure = table[:, 0]
        self.energy_density = table[:, 1]

        self.minimum_pressure = min(self.pressure)
        self.minimum_energy_density = min(self.energy_density)
        if (not self.check_monotonicity() and self.sampling_flag) or self.warning_flag:
            self.warning_flag = True
        else:
            integrand = self.pressure / (self.energy_density + self.pressure)
            self.pseudo_enthalpy = cumulative_trapezoid(integrand, np.log(self.pressure), initial=0) + integrand[0]

            self.interp_energy_density_from_pressure = CubicSpline(np.log10(self.pressure),
                                                                   np.log10(self.energy_density),
                                                                   )

            self.interp_energy_density_from_pseudo_enthalpy = CubicSpline(np.log10(self.pseudo_enthalpy),
                                                                          np.log10(self.energy_density))

            self.interp_pressure_from_pseudo_enthalpy = CubicSpline(np.log10(self.pseudo_enthalpy),
                                                                    np.log10(self.pressure))

            self.interp_pseudo_enthalpy_from_energy_density = CubicSpline(np.log10(self.energy_density),
                                                                          np.log10(self.pseudo_enthalpy))

            self.__construct_all_tables()

            self.minimum_pseudo_enthalpy = min(self.pseudo_enthalpy)
            if not self.check_causality() and self.sampling_flag:
                self.warning_flag = True

    def __remove_leading_zero(self, table):
        """
        For interpolation of lalsimulation tables;
        loglog interpolation breaks if the first entries are 0s
        """

        if table[0, 0] == 0. or table[0, 1] == 0.:
            return table[1:, :]

        else:
            return table

    def energy_from_pressure(self, pressure, interp_type='CubicSpline'):
        """
        Find value of energy_from_pressure
        as in lalsimulation, return e = K * p**(3./5.) below min pressure

        Parameters
        ==========
        pressure: float
            pressure in geometerized units.
        interp_type: str
            String specifying which interpolation type to use.
            Currently implemented: 'CubicSpline', 'linear'.
        energy_density: float
            energy-density in geometerized units.
        """
        pressure = np.atleast_1d(pressure)
        energy_returned = np.zeros(pressure.size)
        indices_less_than_min = np.nonzero(pressure < self.minimum_pressure)
        indices_greater_than_min = np.nonzero(pressure >= self.minimum_pressure)

        # We do this special for less than min pressure
        energy_returned[indices_less_than_min] = 10 ** (np.log10(self.energy_density[0]) +
                                                        (3. / 5.) * (np.log10(pressure[indices_less_than_min]) -
                                                                     np.log10(self.pressure[0])))

        if interp_type == 'CubicSpline':
            energy_returned[indices_greater_than_min] = (
                10. ** self.interp_energy_density_from_pressure(np.log10(pressure[indices_greater_than_min])))
        elif interp_type == 'linear':
            energy_returned[indices_greater_than_min] = np.interp(pressure[indices_greater_than_min],
                                                                  self.pressure,
                                                                  self.energy_density)
        else:
            raise ValueError('Interpolation scheme must be linear or CubicSpline')

        if energy_returned.size == 1:
            return energy_returned[0]
        else:
            return energy_returned

    def pressure_from_pseudo_enthalpy(self, pseudo_enthalpy, interp_type='CubicSpline'):
        """
        Find p(h)
        as in lalsimulation, return p = K * h**(5./2.) below min enthalpy

        :param pseudo_enthalpy (`float`): Dimensionless pseudo-enthalpy.
        :interp_type (`str`): String specifying interpolation type.
                              Current implementations are 'CubicSpline', 'linear'.

        :return pressure (`float`): pressure in geometerized units.
        """
        pseudo_enthalpy = np.atleast_1d(pseudo_enthalpy)
        pressure_returned = np.zeros(pseudo_enthalpy.size)
        indices_less_than_min = np.nonzero(pseudo_enthalpy < self.minimum_pseudo_enthalpy)
        indices_greater_than_min = np.nonzero(pseudo_enthalpy >= self.minimum_pseudo_enthalpy)

        pressure_returned[indices_less_than_min] = 10. ** (np.log10(self.pressure[0]) +
                                                           2.5 * (np.log10(pseudo_enthalpy[indices_less_than_min]) -
                                                                  np.log10(self.pseudo_enthalpy[0])))

        if interp_type == 'CubicSpline':
            pressure_returned[indices_greater_than_min] = (
                10. ** self.interp_pressure_from_pseudo_enthalpy(np.log10(pseudo_enthalpy[indices_greater_than_min])))
        elif interp_type == 'linear':
            pressure_returned[indices_greater_than_min] = np.interp(pseudo_enthalpy[indices_greater_than_min],
                                                                    self.pseudo_enthalpy,
                                                                    self.pressure)
        else:
            raise ValueError('Interpolation scheme must be linear or CubicSpline')

        if pressure_returned.size == 1:
            return pressure_returned[0]
        else:
            return pressure_returned

    def energy_density_from_pseudo_enthalpy(self, pseudo_enthalpy, interp_type='CubicSpline'):
        """
        Find energy_density_from_pseudo_enthalpy(pseudo_enthalpy)
        as in lalsimulation, return e = K * h**(3./2.) below min enthalpy

        :param pseudo_enthalpy (`float`): Dimensionless pseudo-enthalpy.
        :param interp_type (`str`): String specifying interpolation type.
                                    Current implementations are 'CubicSpline', 'linear'.

        :return energy_density (`float`): energy-density in geometerized units.
        """
        pseudo_enthalpy = np.atleast_1d(pseudo_enthalpy)
        energy_returned = np.zeros(pseudo_enthalpy.size)
        indices_less_than_min = np.nonzero(pseudo_enthalpy < self.minimum_pseudo_enthalpy)
        indices_greater_than_min = np.nonzero(pseudo_enthalpy >= self.minimum_pseudo_enthalpy)

        energy_returned[indices_less_than_min] = 10 ** (np.log10(self.energy_density[0]) +
                                                        1.5 * (np.log10(pseudo_enthalpy[indices_less_than_min]) -
                                                               np.log10(self.pseudo_enthalpy[0])))
        if interp_type == 'CubicSpline':
            x = np.log10(pseudo_enthalpy[indices_greater_than_min])
            energy_returned[indices_greater_than_min] = 10 ** self.interp_energy_density_from_pseudo_enthalpy(x)
        elif interp_type == 'linear':
            energy_returned[indices_greater_than_min] = np.interp(pseudo_enthalpy[indices_greater_than_min],
                                                                  self.pseudo_enthalpy,
                                                                  self.energy_density)
        else:
            raise ValueError('Interpolation scheme must be linear or CubicSpline')

        if energy_returned.size == 1:
            return energy_returned[0]
        else:
            return energy_returned

    def pseudo_enthalpy_from_energy_density(self, energy_density, interp_type='CubicSpline'):
        """
        Find h(epsilon)
        as in lalsimulation, return h = K * e**(2./3.) below min enthalpy

        :param energy_density (`float`): energy-density in geometerized units.
        :param interp_type (`str`): String specifying interpolation type.
                                    Current implementations are 'CubicSpline', 'linear'.

        :return pseudo_enthalpy (`float`): Dimensionless pseudo-enthalpy.
        """
        energy_density = np.atleast_1d(energy_density)
        pseudo_enthalpy_returned = np.zeros(energy_density.size)
        indices_less_than_min = np.nonzero(energy_density < self.minimum_energy_density)
        indices_greater_than_min = np.nonzero(energy_density >= self.minimum_energy_density)

        pseudo_enthalpy_returned[indices_less_than_min] = 10 ** (np.log10(self.pseudo_enthalpy[0]) + (2. / 3.) *
                                                                 (np.log10(energy_density[indices_less_than_min]) -
                                                                  np.log10(self.energy_density[0])))

        if interp_type == 'CubicSpline':
            x = np.log10(energy_density[indices_greater_than_min])
            pseudo_enthalpy_returned[indices_greater_than_min] = 10**self.interp_pseudo_enthalpy_from_energy_density(x)
        elif interp_type == 'linear':
            pseudo_enthalpy_returned[indices_greater_than_min] = np.interp(energy_density[indices_greater_than_min],
                                                                           self.energy_density,
                                                                           self.pseudo_enthalpy)
        else:
            raise ValueError('Interpolation scheme must be linear or CubicSpline')

        if pseudo_enthalpy_returned.size == 1:
            return pseudo_enthalpy_returned[0]
        else:
            return pseudo_enthalpy_returned

    def dedh(self, pseudo_enthalpy, rel_dh=1e-5, interp_type='CubicSpline'):
        """
        Value of [depsilon/dh](p)

        :param pseudo_enthalpy (`float`): Dimensionless pseudo-enthalpy.
        :param interp_type (`str`): String specifying interpolation type.
                                    Current implementations are 'CubicSpline', 'linear'.
        :param rel_dh (`float`): Relative step size in pseudo-enthalpy space.

        :return dedh (`float`): Derivative of energy-density with respect to pseudo-enthalpy
                                evaluated at `pseudo_enthalpy` in geometerized units.
        """

        # step size=fraction of value
        dh = pseudo_enthalpy * rel_dh

        eps_upper = self.energy_density_from_pseudo_enthalpy(pseudo_enthalpy + dh, interp_type=interp_type)
        eps_lower = self.energy_density_from_pseudo_enthalpy(pseudo_enthalpy - dh, interp_type=interp_type)

        return (eps_upper - eps_lower) / (2. * dh)

    def dedp(self, pressure, rel_dp=1e-5, interp_type='CubicSpline'):
        """
        Find value of [depsilon/dp](p)

        :param pressure (`float`): pressure in geometerized units.
        :param rel_dp (`float`): Relative step size in pressure space.
        :param interp_type (`float`): String specifying interpolation type.
                                      Current implementations are 'CubicSpline', 'linear'.

        :return dedp (`float`): Derivative of energy-density with respect to pressure
                                evaluated at `pressure`.
        """

        # step size=fraction of value
        dp = pressure * rel_dp

        eps_upper = self.energy_from_pressure(pressure + dp, interp_type=interp_type)
        eps_lower = self.energy_from_pressure(pressure - dp, interp_type=interp_type)

        return (eps_upper - eps_lower) / (2. * dp)

    def velocity_from_pseudo_enthalpy(self, pseudo_enthalpy, interp_type='CubicSpline'):
        """
        Returns the speed of sound in geometerized units in the
        neutron star at the specified pressure.

        Assumes the equation
        vs = c (de/dp)^{-1/2}

        :param pseudo_enthalpy (`float`): Dimensionless pseudo-enthalpy.
        :param interp_type (`str`): String specifying interpolation type.
                                    Current implementations are 'CubicSpline', 'linear'.

        :return v_s (`float`): Speed of sound at `pseudo-enthalpy` in geometerized units.
        """
        pressure = self.pressure_from_pseudo_enthalpy(pseudo_enthalpy, interp_type=interp_type)
        return self.dedp(pressure, interp_type=interp_type) ** -0.5

    def check_causality(self):
        """
        Checks to see if the equation of state is causal i.e. the speed
        of sound in the star is less than the speed of light.
        Returns True if causal, False if not.
        """
        pmax = self.pressure[-1]
        emax = self.energy_from_pressure(pmax)
        hmax = self.pseudo_enthalpy_from_energy_density(emax)
        vsmax = self.velocity_from_pseudo_enthalpy(hmax)
        if vsmax < 1.1:
            return True
        else:
            return False

    def check_monotonicity(self):
        """
        Checks to see if the equation of state is monotonically increasing
        in energy density-pressure space. Returns True if monotonic, False if not.
        """
        e1 = self.energy_density[1:]
        e2 = self.energy_density[:-1]
        ediff = e1 - e2
        e_negatives = len(np.where(ediff < 0))

        p1 = self.pressure[1:]
        p2 = self.pressure[:-1]
        pdiff = p1 - p2
        p_negatives = len(np.where(pdiff < 0))
        if e_negatives > 1 or p_negatives > 1:
            return False
        else:
            return True

    def __get_plot_range(self, data):
        """
        Determines default plot range based on data provided.
        """
        low = np.amin(data)
        high = np.amax(data)
        dx = 0.05 * (high - low)

        xmin = low - dx
        xmax = high + dx
        xlim = [xmin, xmax]

        return xlim

    def __construct_all_tables(self):
        """Pressure and epsilon already tabular, now create array of enthalpies"""
        edat = self.energy_density
        hdat = [self.pseudo_enthalpy_from_energy_density(e) for e in edat]
        self.pseudo_enthalpy = np.array(hdat)

    def plot(self, rep, xlim=None, ylim=None, units=None):
        """
        Given a representation in the form 'energy_density-pressure', plot the EoS in that space.

        Parameters
        ==========
        rep: str
            Representation to plot. For example, plotting in energy_density-pressure space,
            specify 'energy_density-pressure'
        xlim: list
            Plotting bounds for x-axis in the form [low, high].
            Defaults to 'None' which will plot from 10% below min x value to 10% above max x value
        ylim: list
            Plotting bounds for y-axis in the form [low, high].
            Defaults to 'None' which will plot from 10% below min y value to 10% above max y value
        units: str
            Specifies unit system to plot. Currently can plot in CGS:'cgs', SI:'si', or geometerized:'geom'

        Returns
        =======
        fig: matplotlib.figure.Figure
            EOS plot.
        """
        import matplotlib.pyplot as plt

        # Set data based on specified representation
        varnames = rep.split('-')

        assert varnames[0] != varnames[
            1], 'Cannot plot the same variable against itself. Please choose another representation'

        # Correspondence of rep parameter, data, and latex symbol
        # rep_dict = {'energy_density': [self.epsilon, r'$\epsilon$'],
        # 'pressure': [self.p, r'$p$'], 'pseudo_enthalpy': [pseudo_enthalpy, r'$h$']}

        # FIXME: The second element in these arrays should be tex labels, but tex's not working rn
        rep_dict = {'energy_density': [self.energy_density, 'energy_density'],
                    'pressure': [self.pressure, 'pressure'],
                    'pseudo_enthalpy': [self.pseudo_enthalpy, 'pseudo_enthalpy']}

        xname = varnames[1]
        yname = varnames[0]

        # Set units
        eos_default_units = {'pressure': 'cgs', 'energy_density': 'cgs', 'density': 'cgs',
                             'pseudo_enthalpy': 'dimensionless'}
        if units is None:
            units = [eos_default_units[yname], eos_default_units[xname]]          # Default unit system is cgs
        elif isinstance(units, str):
            units = [units, units]          # If only one unit system given, use for both

        xunits = units[1]
        yunits = units[0]

        # Ensure valid units
        if xunits not in list(conversion_dict[xname].keys()) or yunits not in list(conversion_dict[yname].keys()):
            s = '''
                Invalid unit system. Valid variable-unit pairs are:
                p: {p_units}
                e: {e_units}
                rho: {rho_units}
                h: {h_units}.
                '''.format(p_units=list(conversion_dict['pressure'].keys()),
                           e_units=list(conversion_dict['energy_density'].keys()),
                           rho_units=list(conversion_dict['density'].keys()),
                           h_units=list(conversion_dict['pseudo_enthalpy'].keys()))
            raise ValueError(s)

        xdat = rep_dict[xname][0] * conversion_dict[xname][xunits]
        ydat = rep_dict[yname][0] * conversion_dict[yname][yunits]

        xlabel = rep_dict[varnames[1]][1].replace('_', ' ')
        ylabel = rep_dict[varnames[0]][1].replace('_', ' ') + '(' + xlabel + ')'

        # Determine plot ranges. Currently shows 10% wider than actual data range.
        if xlim is None:
            xlim = self.__get_plot_range(xdat)

        if ylim is None:
            ylim = self.__get_plot_range(ydat)

        fig, ax = plt.subplots()

        ax.loglog(xdat, ydat)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig


def spectral_adiabatic_index(gammas, x):
    arg = 0
    for i in range(len(gammas)):
        arg += gammas[i] * x ** i

    return np.exp(arg)


class SpectralDecompositionEOS(TabularEOS):
    """
    Parameterized EOS using a spectral
    decomposition per Lindblom
    arXiv: 1009.0738v2. Inherits from TabularEOS.

    Parameters
    ==========
    gammas: list
        List of adiabatic expansion parameters used
        to construct the equation of state in various
        spaces.
    p0: float
        The starting point in pressure of the high-density EoS. This is stitched to
        the low-density portion of the SLY EoS model. The default value chosen is set to
        a sufficiently low pressure so that the high-density EoS will never be
        overconstrained.
    e0/c**2: float
        The starting point in energy-density of the high-density EoS. This is stitched to
        the low-density portion of the SLY EoS model. The default value chosen is set to
        a sufficiently low energy density so that the high-density EoS will never be
        overconstrained.
    xmax: float
        highest dimensionless pressure value in EoS
    npts: float (optional)
        number of points in pressure-energy density data.
    """

    def __init__(self, gammas, p0=3.01e33, e0=2.03e14, xmax=None, npts=100, sampling_flag=False, warning_flag=False):
        self.warning_flag = warning_flag
        self.gammas = gammas
        self.p0 = p0
        self.e0 = e0
        self.xmax = xmax
        self.npts = npts
        if self.xmax is None:
            self.xmax = self.__determine_xmax()
        self.sampling_flag = sampling_flag
        self.__construct_a_of_x_table()

        # Construct pressure-energy density table and
        # set up interpolating functions.

        if self.warning_flag and self.sampling_flag:
            # If sampling prior is enabled and adiabatic check
            # has failed, empty the array values
            self.e_pdat = np.zeros((2, 2))
        else:
            self.e_pdat = self.__construct_e_of_p_table()

        super().__init__(self.e_pdat, sampling_flag=self.sampling_flag, warning_flag=self.warning_flag)

    def __determine_xmax(self, a_max=6.0):
        highest_order_gamma = np.abs(self.gammas[-1])[0]
        expansion_order = float(len(self.gammas) - 1)

        xmax = (np.log(a_max) / highest_order_gamma) ** (1. / expansion_order)
        return xmax

    def __mu_integrand(self, x):

        return 1. / spectral_adiabatic_index(self.gammas, x)

    def mu(self, x):
        from scipy.integrate import quad
        return np.exp(-quad(self.__mu_integrand, 0, x)[0])

    def __eps_integrand(self, x):

        return np.exp(x) * self.mu(x) / spectral_adiabatic_index(self.gammas, x)

    def energy_density(self, x, eps0):
        from scipy.integrate import quad
        quad_result, quad_err = quad(self.__eps_integrand, 0, x)
        eps_of_x = (eps0 * C_CGS ** 2.) / self.mu(x) + self.p0 / self.mu(x) * quad_result
        return eps_of_x

    def __construct_a_of_x_table(self):

        xdat = np.linspace(0, self.xmax, num=self.npts)

        # Generate adiabatic index points until a point is out of prior range
        if self.sampling_flag:
            adat = np.empty(self.npts)
            for i in range(self.npts):
                if 0.6 < spectral_adiabatic_index(self.gammas, xdat[i]) < 4.6:
                    adat[i] = spectral_adiabatic_index(self.gammas, xdat[i])
                else:
                    break

            # Truncate arrays to last point within range
            adat = adat[:i]
            xmax_new = xdat[i - 1]

            # If EOS is too short, set prior to 0, else resample the function and set new xmax
            if xmax_new < 4. or i == 0:
                self.warning_flag = True
            else:
                xdat = np.linspace(0, xmax_new, num=self.npts)
                adat = spectral_adiabatic_index(self.gammas, xdat)
                self.xmax = xmax_new
        else:
            adat = spectral_adiabatic_index(self.gammas, xdat)

        self.adat = adat

    def __construct_e_of_p_table(self):

        """
        Creates p, epsilon table for a given set of spectral parameters
        """

        # make p range
        # to match lalsimulation tables: array = [pressure, density]
        x_range = np.linspace(0, self.xmax, self.npts)
        p_range = self.p0 * np.exp(x_range)

        eos_vals = np.zeros((self.npts, 2))
        eos_vals[:, 0] = p_range

        for i in range(0, len(x_range)):
            eos_vals[i, 1] = self.energy_density(x_range[i], self.e0)

        # convert eos to geometrized units in *m^-2*
        # IMPORTANT
        eos_vals = eos_vals * 0.1 * G_SI / C_SI ** 4

        # doing as those before me have done and using SLY4 as low density region
        # SLY4 in geometrized units
        low_density_path = os.path.join(os.path.dirname(__file__), 'eos_tables', 'LALSimNeutronStarEOS_SLY4.dat')
        low_density = np.loadtxt(low_density_path)

        cutoff = eos_vals[0, :]

        # Then find overlap point
        break_pt = len(low_density)
        for i in range(1, len(low_density)):
            if low_density[-i, 0] < cutoff[0] and low_density[-i, 1] < cutoff[1]:
                break_pt = len(low_density) - i + 1
                break

        # stack EOS arrays
        eos_vals = np.vstack((low_density[0:break_pt, :], eos_vals))

        return eos_vals


class EOSFamily(object):
    """
    Create a EOS family and get mass-radius information

    Parameters
    ==========
    eos: object
        Supply a `TabularEOS` class (or subclass)
    npts: float
        Number of points to calculate for mass-radius relation. Default is 500.

    Notes
    =====
    The mass-radius and mass-k2 data should be
    populated here via the TOV solver upon object construction.
    """
    def __init__(self, eos, npts=500):
        from scipy.optimize import minimize_scalar
        self.eos = eos

        # FIXME: starting_energy_density is set somewhat arbitrarily
        starting_energy_density = 1.6e-10
        ending_energy_density = max(self.eos.energy_density)
        log_starting_energy_density = np.log(starting_energy_density)
        log_ending_energy_density = np.log(ending_energy_density)
        log_energy_density_grid = np.linspace(log_starting_energy_density,
                                              log_ending_energy_density,
                                              num=npts)
        energy_density_grid = np.exp(log_energy_density_grid)

        # Generate m, r, and k2 lists
        mass = []
        radius = []
        k2love_number = []
        for i in range(len(energy_density_grid)):
            tov_solver = IntegrateTOV(self.eos, energy_density_grid[i])
            m, r, k2 = tov_solver.integrate_TOV()
            mass.append(m)
            radius.append(r)
            k2love_number.append(k2)

            # Check if maximum mass has been found
            if i > 0 and mass[i] <= mass[i - 1]:
                break

        # If we're not at the end of the array, determine actual maximum mass. Else, assume
        # last point is the maximum mass and proceed.
        if i < (npts - 1):
            # Now replace with point with interpolated maximum mass
            # This is done by interpolating the last three points and then
            # minimizing the negative of the interpolated function
            x = [energy_density_grid[i - 2], energy_density_grid[i - 1], energy_density_grid[i]]
            y = [mass[i - 2], mass[i - 1], mass[i]]

            f = interp1d(x, y, kind='quadratic', bounds_error=False, fill_value='extrapolate')

            res = minimize_scalar(lambda x: -f(x))

            # Integrate max energy density to get maximum mass
            tov_solver = IntegrateTOV(self.eos, res.x)
            mfin, rfin, k2fin = tov_solver.integrate_TOV()

            mass[-1] = mfin
            radius[-1] = rfin
            k2love_number[-1] = k2fin

        # Currently, everything is in geometerized units.
        # The mass variables have dimensions of length, k2 is dimensionless
        # and radii have dimensions of length. Calculate dimensionless lambda
        # with these quantities, then convert to SI.

        # Calculating dimensionless lambda values from k2, radii, and mass
        tidal_deformability = [2. / 3. * k2 * r ** 5. / m ** 5. for k2, r, m in
                               zip(k2love_number, radius, mass)]

        # As a last resort, if highest mass is still smaller than second
        # to last point, remove the last point from each array
        if mass[-1] < mass[-2]:
            mass = mass[:-1]
            radius = radius[:-1]
            k2love_number = k2love_number[:-1]
            tidal_deformability = tidal_deformability[:-1]

        self.mass = np.array(mass)
        self.radius = np.array(radius)
        self.k2love_number = np.array(k2love_number)
        self.tidal_deformability = np.array(tidal_deformability)
        self.maximum_mass = mass[-1] * conversion_dict['mass']['m_sol']

    def radius_from_mass(self, m):
        """
        :param m: mass of neutron star in solar masses
        :return: radius of neutron star in meters
        """
        f = CubicSpline(self.mass, self.radius, bc_type='natural', extrapolate=True)

        mass_converted_to_geom = m * MSUN_SI * G_SI / C_SI ** 2.
        return f(mass_converted_to_geom)

    def k2_from_mass(self, m):
        """
        :param m: mass of neutron star in solar masses.
        :return: dimensionless second tidal love number.
        """
        f = CubicSpline(self.mass, self.k2love_number, bc_type='natural', extrapolate=True)

        m_geom = m * MSUN_SI * G_SI / C_SI ** 2.
        return f(m_geom)

    def lambda_from_mass(self, m):
        """
        Convert from equation of state model parameters to
        component tidal parameters.

        :param m: Mass of neutron star in solar masses.
        :return: Tidal parameter of neutron star of mass m.
        """

        # Get lambda from mass and equation of state

        r = self.radius_from_mass(m)
        k = self.k2_from_mass(m)
        m_geom = m * MSUN_SI * G_SI / C_SI ** 2.
        c = m_geom / r
        lmbda = (2. / 3.) * k / c ** 5.

        return lmbda

    def __get_plot_range(self, data):
        low = np.amin(data)
        high = np.amax(data)
        dx = 0.05 * (high - low)

        xmin = low - dx
        xmax = high + dx
        xlim = [xmin, xmax]

        return xlim

    def plot(self, rep, xlim=None, ylim=None, units=None):
        """
        Given a representation in the form 'm-r', plot the family in that space.

        Parameters
        ==========
        rep: str
            Representation to plot. For example, plotting in mass-radius space, specify 'm-r'
        xlim: list
            Plotting bounds for x-axis in the form [low, high].
            Defaults to 'None' which will plot from 10% below min x value to 10% above max x value
        ylim: list
            Plotting bounds for y-axis in the form [low, high].
            Defaults to 'None' which will plot from 10% below min y value to 10% above max y value
        units: str
            Specifies unit system to plot. Currently can plot in CGS:'cgs', SI:'si', or geometerized:'geom'

        Returns
        =======
        fig: matplotlib.figure.Figure
            EOS Family plot.
        """
        import matplotlib.pyplot as plt

        # Set data based on specified representation
        varnames = rep.split('-')

        assert varnames[0] != varnames[
            1], 'Cannot plot the same variable against itself. Please choose another representation'

        # Correspondence of rep parameter, data, and latex symbol
        rep_dict = {'mass': [self.mass, r'$M$'], 'radius': [self.radius, r'$R$'], 'k2': [self.k2love_number, r'$k_2$'],
                    'tidal_deformability': [self.tidal_deformability, r'$l$']}

        xname = varnames[1]
        yname = varnames[0]

        # Set units
        fam_default_units = {'mass': 'm_sol', 'radius': 'km', 'tidal_deformability': 'geom'}
        if units is None:
            units = [fam_default_units[yname], fam_default_units[xname]]  # Default unit system is cgs
        elif isinstance(units, str):
            units = [units, units]  # If only one unit system given, use for both

        xunits = units[1]
        yunits = units[0]

        # Ensure valid units
        if xunits not in list(conversion_dict[xname].keys()) or yunits not in list(conversion_dict[yname].keys()):
            s = '''
                        Invalid unit system. Valid variable-unit pairs are:
                        m: {m_units}
                        r: {r_units}
                        l: {l_units}.
                        '''.format(m_units=list(conversion_dict['mass'].keys()),
                                   r_units=list(conversion_dict['radius'].keys()),
                                   l_units=list(conversion_dict['tidal_deformability'].keys()))
            raise ValueError(s)

        xdat = rep_dict[varnames[1]][0] * conversion_dict[xname][xunits]
        ydat = rep_dict[varnames[0]][0] * conversion_dict[yname][yunits]

        xlabel = rep_dict[varnames[1]][1].replace('_', ' ')
        ylabel = rep_dict[varnames[0]][1].replace('_', ' ') + '(' + xlabel + ')'

        # Determine plot ranges. Currently shows 10% wider than actual data range.
        if xlim is None:
            xlim = self.__get_plot_range(xdat)

        if ylim is None:
            ylim = self.__get_plot_range(ydat)

        # Plot the data with appropriate labels
        fig, ax = plt.subplots()

        ax.loglog(xdat, ydat)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig
