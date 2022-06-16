from dataclasses import dataclass
from munch import Munch
import numpy as np
from openamundsen import errors

_DTYPE_INIT_VALS = {
    float: np.nan,
    int: 0,
    bool: False,
}
_DTYPE_INIT_VALS[np.dtype('float64')] = _DTYPE_INIT_VALS[float]
_DTYPE_INIT_VALS[np.dtype('int64')] = _DTYPE_INIT_VALS[int]


class StateVariableManager:
    """
    Class for managing state variables of a OpenAmundsen instance.
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
                if var_def.dim3 == 0:
                    arr = create_empty_array((self._rows, self._cols), var_def.dtype)
                else:
                    arr = create_empty_array((var_def.dim3, self._rows, self._cols), var_def.dtype)

                svc[var_name] = arr

    def reset(self):
        """
        Fill all state variables with their default "no data" value, except
        those for which the `retain` keyword is set.
        """
        for category in self.categories:
            svc = self[category]

            for var_name, var_def in svc._meta.items():
                if not var_def.retain:
                    arr = svc[var_name]
                    arr.fill(_DTYPE_INIT_VALS[arr.dtype])

    def meta(self, var):
        """
        Return metadata of a state variable.

        Parameters
        ----------
        var : str
            Variable name, e.g. "meteo.temp".

        Returns
        -------
        StateVariableDefinition
        """
        category, var_name = self.parse(var)
        return self[category]._meta[var_name]


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

    def add_variable(
            self,
            name,
            units=None,
            long_name=None,
            standard_name=None,
            dtype=float,
            dim3=0,
            retain=False,
    ):
        """
        Add a variable along with optional metadata.
        Note that the variable itself is not yet created here (this is done in
        StateVariableManager.iniitalize); only its name and attribute are
        stored.

        Parameters
        ----------
        name : str
            Name of the variable to be created. Must be a valid Python identifier.

        units : str, optional
            udunits-compatible definition of the units of the given variable.

        long_name : str, optional
            Long descriptive name of the variable.

        standard_name : str, optional
            Standard name of the variable in CF notation
            (http://cfconventions.org/standard-names.html).

        dtype : type, default float
            Data type for the variable values.

        dim3 : int, default 0
            Size of an optional third dimension of the field.

        retain : bool, default False
            Whether the variable contents should be retained or it is safe to
            reset the variable in every timestep.
        """
        definition = StateVariableDefinition(
            units=units,
            long_name=long_name,
            standard_name=standard_name,
            dtype=dtype,
            dim3=dim3,
            retain=retain,
        )

        self._meta[name] = definition


@dataclass(frozen=True)
class StateVariableDefinition:
    """
    Class for describing metadata of a state variable.

    Parameters
    ----------
    units : str, optional
        udunits-compatible definition of the units of the given variable.

    long_name : str, optional
        Long descriptive name of the variable.

    standard_name : str, optional
        Standard name of the variable in CF notation
        (http://cfconventions.org/standard-names.html).

    dtype : type, default float
        Data type for the variable values.

    dim3 : int, default 0
        Size of an optional third dimension of the field.

    retain : bool, default False
        Whether the variable contents should be retained or it is safe to reset
        the variable in every timestep.

    Examples
    --------
    >>> StateVariableDefinition(
            standard_name='precipitation_flux',
            long_name='Total precipitation flux',
            units='kg m-2 s-1')
    """
    units: str = None
    long_name: str = None
    standard_name: str = None
    dtype: type = float
    dim3: int = 0
    retain: bool = False


def add_default_state_variables(model):
    """
    Add all state variables to an OpenAmundsen instance which are required for any
    model run. Depending on which submodules are activated in the run
    configuration, further state variables might be added in other locations.
    """
    state = model.state

    # Base variables
    base = state.add_category('base')
    base.add_variable('dem', 'm', 'Surface altitude', 'surface_altitude', retain=True)
    base.add_variable('slope', 'degree', 'Terrain slope', retain=True)
    base.add_variable('aspect', 'degree', 'Terrain aspect', retain=True)
    base.add_variable('normal_vec', long_name='Vector normal to the surface', dim3=3, retain=True)
    base.add_variable('svf', long_name='Sky-view factor', retain=True)

    # Meteorological variables
    meteo = state.add_category('meteo')
    meteo.add_variable('temp', 'K', 'Air temperature', 'air_temperature')
    meteo.add_variable('precip', 'kg m-2', 'Precipitation amount', 'precipitation_amount')
    meteo.add_variable('snowfall', 'kg m-2', 'Snowfall amount', 'snowfall_amount')
    meteo.add_variable('rainfall', 'kg m-2', 'Rainfall amount', 'rainfall_amount')
    meteo.add_variable('rel_hum', '%', 'Relative humidity', 'relative_humidity')
    meteo.add_variable('wind_speed', 'm s-1', 'Wind speed', 'wind_speed')
    meteo.add_variable('wind_dir', 'degree', 'Wind direction', 'wind_from_direction')
    meteo.add_variable('sw_in', 'W m-2', 'Incoming shortwave radiation', 'surface_downwelling_shortwave_flux_in_air')
    meteo.add_variable('sw_out', 'W m-2', 'Outgoing shortwave radiation', 'surface_upwelling_shortwave_flux_in_air')
    meteo.add_variable('lw_in', 'W m-2', 'Incoming longwave radiation', 'downwelling_longwave_flux_in_air')
    meteo.add_variable('lw_out', 'W m-2', 'Outgoing longwave radiation', 'surface_upwelling_longwave_flux_in_air')
    meteo.add_variable('net_radiation', 'W m-2', 'Net radiation', 'surface_net_downward_radiative_flux')
    meteo.add_variable('sw_in_clearsky', 'W m-2', 'Clear-sky incoming shortwave radiation', 'surface_downwelling_shortwave_flux_in_air_assuming_clear_sky')
    meteo.add_variable('dir_in_clearsky', 'W m-2', 'Clear-sky direct incoming shortwave radiation')
    meteo.add_variable('diff_in_clearsky', 'W m-2', 'Clear-sky diffuse incoming shortwave radiation', 'surface_diffuse_downwelling_shortwave_flux_in_air_assuming_clear_sky')
    meteo.add_variable('cloud_factor', '1', 'Cloud factor', retain=True)
    meteo.add_variable('cloud_fraction', '1', 'Cloud fraction', 'cloud_area_fraction')
    meteo.add_variable('wet_bulb_temp', 'K', 'Wet-bulb temperature', 'wet_bulb_temperature')
    meteo.add_variable('dew_point_temp', 'K', 'Dew point temperature', 'dew_point_temperature')
    meteo.add_variable('atmos_press', 'Pa', 'Atmospheric pressure', 'air_pressure')
    meteo.add_variable('sat_vap_press', 'Pa', 'Saturation vapor pressure')
    meteo.add_variable('vap_press', 'Pa', 'Vapor pressure', 'water_vapor_partial_pressure_in_air')
    meteo.add_variable('spec_hum', 'kg kg-1', 'Specific humidity', 'specific_humidity')
    meteo.add_variable('spec_heat_cap_moist_air', 'J kg-1 K-1', 'Specific heat capacity of moist air')
    meteo.add_variable('psych_const', 'Pa K-1', 'Psychrometric constant')
    meteo.add_variable('lat_heat_vap', 'J kg-1', 'Latent heat of vaporization')
    meteo.add_variable('precipitable_water', 'kg m-2', 'Precipitable water')
    meteo.add_variable('dry_air_density', 'kg m-3', 'Dry air density')
    if model._require_evapotranspiration or model._require_canopy:
        meteo.add_variable('top_canopy_temp', 'K', 'Above-canopy air temperature')
        meteo.add_variable('top_canopy_rel_hum', '%', 'Above-canopy relative humidity')
        meteo.add_variable('top_canopy_wind_speed', 'm s-1', 'Above-canopy wind speed')
        meteo.add_variable('top_canopy_sw_in', 'W m-2', 'Above-canopy incoming shortwave radiation')
        meteo.add_variable('top_canopy_lw_in', 'W m-2', 'Above-canopy incoming longwave radiation')

    # Surface variables
    surf = state.add_category('surface')
    surf.add_variable('temp', 'K', 'Surface temperature', 'surface_temperature', retain=True)
    surf.add_variable('albedo', '1', 'Surface albedo', 'surface_albedo', retain=True)
    surf.add_variable('heat_flux', 'W m-2', 'Surface heat flux')
    surf.add_variable('sens_heat_flux', 'W m-2', 'Sensible heat flux', 'surface_downward_sensible_heat_flux')
    surf.add_variable('lat_heat_flux', 'W m-2', 'Latent heat flux', 'surface_downward_latent_heat_flux')
    surf.add_variable('moisture_flux', 'kg m-2 s-1', 'Moisture flux')
    surf.add_variable('advective_heat_flux', 'W m-2', 'Heat advected by precipitation')
    surf.add_variable('lat_heat', 'J kg-1', 'Latent heat')
    surf.add_variable('sat_spec_hum', 'kg kg-1', 'Saturation specific humidity at surface temperature')
    surf.add_variable('moisture_availability', '1', 'Moisture availability factor')
    surf.add_variable('roughness_length', 'm', 'Surface roughness length', 'surface_roughness_length')
    surf.add_variable('turbulent_exchange_coeff', '1', 'Transfer coefficient for heat and moisture')
    surf.add_variable('conductance', 'm s-1', 'Surface conductance')
    if model.config.snow.model == 'multilayer':
        surf.add_variable('layer_temp', 'K', 'Surface layer temperature', retain=True)
        surf.add_variable('thickness', 'm', 'Surface layer thickness', retain=True)
        surf.add_variable('therm_cond', 'W m-1 K-1', 'Surface thermal conductivity')
    elif model.config.snow.model == 'cryolayers':
        surf.add_variable('layer_type', '1', 'Surface layer type', dtype=int)

    # Snow variables (shared by all snow models, additional ones might be added from the individual models)
    snow = state.add_category('snow')
    snow.add_variable('swe', 'kg m-2', 'Snow water equivalent', 'surface_snow_amount', retain=True)
    snow.add_variable('depth', 'm', 'Snow depth', 'surface_snow_thickness')
    snow.add_variable('melt', 'kg m-2', 'Snow melt', 'surface_snow_melt_amount')
    snow.add_variable('sublimation', 'kg m-2', 'Snow sublimation', 'surface_snow_sublimation_amount')
    snow.add_variable('refreezing', 'kg m-2', 'Liquid water refreezing')
    snow.add_variable('runoff', 'kg m-2', 'Snow runoff')
    snow.add_variable('albedo', '1', 'Snow albedo', retain=True)
    snow.add_variable('area_fraction', '1', 'Snow cover fraction', 'surface_snow_area_fraction')

    # Soil variables
    soil = state.add_category('soil')
    num_soil_layers = len(model.config.soil.thickness)
    soil.add_variable('heat_flux', 'W m-2', 'Soil heat flux', 'downward_heat_flux_in_soil')
    soil.add_variable('vol_heat_cap_dry', 'J K-1 m-3', 'Volumetric heat capacity of dry soil', retain=True)
    soil.add_variable('sat_water_pressure', 'm', 'Saturated soil water pressure', retain=True)
    soil.add_variable('vol_moisture_content_sat', 'm3 m-3', 'Volumetric soil moisture content at saturation', retain=True)
    soil.add_variable('vol_moisture_content_crit', 'm3 m-3', 'Volumetric soil moisture content at critical point', retain=True)
    soil.add_variable('frac_frozen_moisture_content', '1', 'Fractional frozen soil moisture content', dim3=num_soil_layers, retain=True)
    soil.add_variable('frac_unfrozen_moisture_content', '1', 'Fractional unfrozen soil moisture content', dim3=num_soil_layers, retain=True)
    soil.add_variable('clapp_hornberger', '1', 'Clapp-Hornberger exponent', retain=True)
    soil.add_variable('vol_moisture_content', 'm3 m-3', 'Volumetric soil moisture content', dim3=num_soil_layers, retain=True)
    soil.add_variable('temp', 'K', 'Soil temperature', dim3=num_soil_layers, retain=True)
    soil.add_variable('thickness', 'm', 'Soil thickness', dim3=num_soil_layers, retain=True)
    soil.add_variable('heat_cap', 'J K-1 m-2', 'Areal heat capacity of soil', dim3=num_soil_layers)
    soil.add_variable('therm_cond', 'W m-1 K-1', 'Thermal conductivity of soil', dim3=num_soil_layers)
    soil.add_variable('therm_cond_minerals', 'W m-1 K-1', 'Thermal conductivity of soil minerals', retain=True)
    soil.add_variable('therm_cond_dry', 'W m-1 K-1', 'Thermal conductivity of dry soil', retain=True)


def create_empty_array(shape, dtype):
    """
    Create an empty array with a given shape and dtype initialized to "no
    data". The value of "no data" depends on the dtype and is e.g. NaN for
    float, 0 for int and False for float.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g. (2, 3).

    dtype : type
        The desired data type for the array.

    Returns
    -------
    out : ndarray
    """
    return np.full(shape, _DTYPE_INIT_VALS[dtype], dtype=dtype)
