from munch import Munch
from openamundsen.util import create_empty_array


class StateVariableContainer(Munch):
    """
    Container for storing state variables. This class inherits from `Munch` so
    that attributes are accessible both using dict notation (`state['temp']`)
    as well as dot notation (`state.temp`).
    """
    pass


def initialize_state_variables(model):
    """
    Initialize the default state variables (i.e., create empty arrays) of a
    Model instance.
    """
    model.logger.info('Initializing state variables')

    rows = model.grid['rows']
    cols = model.grid['cols']

    def field(dtype=float):
        return create_empty_array((rows, cols), dtype)

    model.state = StateVariableContainer()

    # Base variables
    base = StateVariableContainer()
    base.dem = field()  # terrain elevation (m)
    base.slope = field()  # terrain slope
    base.aspect = field()  # terrain aspect
    base.roi = field(bool)  # region of interest

    # Meteorological variables
    meteo = StateVariableContainer()
    meteo.temp = field()  # air temperature (K)
    meteo.precip = field()  # precipitation (kg m-2 s-1)
    meteo.rel_hum = field()  # relative humidity (%)
    meteo.shortwave_in = field()  # shortwave incoming radiation (W m-2)
    meteo.wind_speed = field()  # wind speed (m s-1)

    # Snow variables
    snow = StateVariableContainer()
    snow.swe = field()
    snow.depth = field()

    model.state.base = base
    model.state.meteo = meteo
    model.state.snow = snow
