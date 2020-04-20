from dataclasses import dataclass
from munch import Munch
from openamundsen import errors
from openamundsen.util import create_empty_array


class StateVariableManager:
    """
    Class for managing state variables of a Model instance.
    State variables are organized into categories (e.g., "base", "meteo",
    "snow").  For each category, a StateVariableContainer is created containing
    the variables and their respective metadata.

    Parameters
    ----------
    rows, cols : int
        Number of rows and columns of the model grid.
    """
    def __init__(self, rows, cols):
        self._rows = rows
        self._cols = cols
        self._categories = []

    @property
    def categories(self):
        """
        Return the currently assigned categories.
        """
        return tuple(self._categories)

    def add_category(self, category):
        """
        Add a category.

        Parameters
        ----------
        category : str
            Category name.
        """
        if category in self.categories:
            raise errors.CategoryError(f'Category {category} already exists')

        if not category.isidentifier():
            raise errors.CategoryError(f'Category name must be a valid Python identifier')

        self._categories.append(category)
        self._categories.sort()
        svc = StateVariableContainer(self)
        setattr(self, category, svc)

        return svc

    def parse(self, s):
        """
        Parse a string in the form "<category>[.<variable>]" and return the
        respective category and variable name (or None for the latter, it not
        specified).

        Returns
        -------
        (category, var_name): (str, str or None) tuple
        """
        if '.' in s:
            category, var_name = s.split('.')
        else:
            category = s
            var_name = None

        return category, var_name

    def __getitem__(self, key):
        """
        Return either the StateVariableContainer for a category (if key is a
        category identifier) or a variable itself (if key is a string in the
        form "<category>.<variable>".

        Parameters
        ----------
        key : str

        Returns
        -------
        StateVariableContainer or ndarray
        """
        category, var_name = self.parse(key)

        if var_name is None:
            return getattr(self, category)
        else:
            return getattr(self, category)[var_name]

    def initialize(self):
        """
        Initialize the fields (i.e., create the arrays) for all variables of
        all categories.
        """
        for category in self.categories:
            svc = self[category]

            for var_name, var_def in svc._meta.items():
                svc[var_name] = create_empty_array((self._rows, self._cols), var_def.dtype)


class StateVariableContainer(Munch):
    """
    Container for storing state variables. This class inherits from `Munch` so
    that attributes are accessible both using dict notation (`svc['temp']`)
    as well as dot notation (`svc.temp`).

    Parameters
    ----------
    manager : StateVariableManager
        StateVariableManager instance to which this container is associated.
    """
    def __init__(self, manager):
        self._manager = manager
        self._meta = {}

    def add_variable(self, name, dtype=float, standard_name=None, long_name=None, units=None):
        """
        Add a variable along with optional metadata.
        Note that the variable itself is not yet created here (this is done in
        StateVariableManager.iniitalize); only its name and attribute are
        stored.

        Parameters
        ----------
        name : str
            Name of the variable to be created. Must be a valid Python identifier.

        dtype : type, default float
            Data type for the variable values.

        standard_name : str, optional
            Standard name of the variable in CF notation
            (http://cfconventions.org/standard-names.html).

        long_name : str, optional
            Long descriptive name of the variable.

        units : str, optional
            udunits-compatible definition of the units of the given variable.
        """
        definition = StateVariableDefinition(
            dtype=dtype,
            standard_name=standard_name,
            long_name=long_name,
            units=units,
        )

        self._meta[name] = definition


@dataclass(frozen=True)
class StateVariableDefinition:
    """
    Class for describing metadata of a state variable.

    Parameters
    ----------
    dtype : type, default float
        Data type for the variable values.

    standard_name : str, optional
        Standard name of the variable in CF notation
        (http://cfconventions.org/standard-names.html).

    long_name : str, optional
        Long descriptive name of the variable.

    units : str, optional
        udunits-compatible definition of the units of the given variable.

    Examples
    --------
    >>> StateVariableDefinition(
            standard_name='precipitation_flux',
            long_name='Total precipitation flux',
            units='kg m-2 s-1')
    """
    dtype: type = float
    standard_name: str = None
    long_name: str = None
    units: str = None


def add_default_state_variables(model):
    """
    Add all state variables to a Model instance which are required for any
    model run. Depending on which submodules are activated in the run
    configuration, further state variables might be added in other locations.
    """
    state = model.state

    # Base variables
    base = state.add_category('base')
    base.add_variable('dem', standard_name='surface_altitude', units='m')
    base.add_variable('slope')
    base.add_variable('aspect')
    base.add_variable('roi', dtype=bool)

    # Meteorological variables
    meteo = state.add_category('meteo')
    meteo.add_variable(
        'temp',
        standard_name='air_temperature',
        units='K',
        long_name='Air Temperature')
    meteo.add_variable(
        'precip',
        standard_name='precipitation_flux',
        units='kg m-2 s-1',
        long_name='Precipitation Flux',
    )
    meteo.add_variable(
        'rel_hum',
        standard_name='relative_humidity',
        units='%',
        long_name='Relative Humidity',
    )
    meteo.add_variable(
        'short_in',
        standard_name='surface_downwelling_shortwave_flux_in_air',
        units='W m-2',
        long_name='Downwelling Shortwave Radiation',
    )
    meteo.add_variable(
        'wind_speed',
        standard_name='wind_speed',
        units='m s-1',
        long_name='Wind Speed',
    )
