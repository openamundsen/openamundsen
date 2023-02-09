DAYS_PER_YEAR = 365.25
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600

STANDARD_TIMEZONE_WIDTH = 15
SUN_DEGREES_PER_HOUR = 15  # 360/24
MINUTES_PER_DEGREE_OF_EARTH_ROTATION = 4  # earth rotates 1 degree every 4 minutes
SOLAR_CONSTANT = 1366.1  # W m-2

STANDARD_ATMOSPHERE = 101_325  # Pa
ATMOSPHERIC_LAPSE_RATE = 6.5e-3  # K m-1
GRAVITATIONAL_ACCELERATION = 9.80665  # m s-2

T0 = 273.15  # K

MOLAR_MASS_DRY_AIR = 0.02896968  # kg mol-1
MOLAR_MASS_WATER = 0.0180153  # kg mol-1
UNIVERSAL_GAS_CONSTANT = 8.314462618  # J mol-1 K-1
GAS_CONSTANT_DRY_AIR = 287.058  # J kg-1 K-1
SPEC_GAS_CONSTANT_WATER_VAPOR = 461.52  # J kg-1 K-1
STANDARD_SEA_LEVEL_TEMPERATURE = 15 + T0  # K

SPEC_HEAT_CAP_DRY_AIR = 1004.68506  # specific heat capacity of dry air (J kg-1 K-1)
SPEC_HEAT_CAP_ICE = 2100.  # specific heat capacity of ice (J kg-1 K-1)
SPEC_HEAT_CAP_WATER = 4180.  # specific heat capacity of water (J kg-1 K-1)

LATENT_HEAT_OF_VAPORIZATION = 2.501e6  # J kg-1 (= latent heat of condensation)
LATENT_HEAT_OF_FUSION = 0.334e6  # J kg-1
LATENT_HEAT_OF_SUBLIMATION = LATENT_HEAT_OF_VAPORIZATION + LATENT_HEAT_OF_FUSION  # J kg-1

STEFAN_BOLTZMANN = 5.670374419e-8  # Stefan-Boltzmann constant (W m-2 K-4)
VON_KARMAN = 0.40  # von Karman constant

WATER_DENSITY = 1000.  # density of water (kg m-3)
ICE_DENSITY = 917.  # density of ice (kg m-3)

# Soil constants
VOL_HEAT_CAP_SAND = 2.128e6  # volumetric heat capacity of sand (J m-3 K-1)
VOL_HEAT_CAP_CLAY = 2.385e6  # volumetric heat capacity of clay (J m-3 K-1)

# Thermal conductivities
THERM_COND_AIR = 0.025  # thermal conductivity of air (W m-1 K-1)
THERM_COND_CLAY = 1.16  # thermal conductivity of clay (W m-1 K-1)
THERM_COND_ICE = 2.24  # thermal conducivity of ice (W m-1 K-1)
THERM_COND_SAND = 1.57  # thermal conductivity of sand (W m-1 K-1)
THERM_COND_WATER = 0.56  # thermal conductivity of water (W m-1 K-1)

CRS_WGS84 = 'epsg:4326'

METEO_VAR_METADATA = {
    'temp': {
        'standard_name': 'air_temperature',
        'units': 'K',
    },
    'precip': {
        'standard_name': 'precipitation_amount',
        'units': 'kg m-2',
    },
    'rel_hum': {
        'standard_name': 'relative_humidity',
        'units': '%',
    },
    'sw_in': {
        'standard_name': 'surface_downwelling_shortwave_flux_in_air',
        'units': 'W m-2',
    },
    'wind_speed': {
        'standard_name': 'wind_speed',
        'units': 'm s-1',
    },
    'wind_dir': {
        'standard_name': 'wind_from_direction',
        'units': 'degree',
    },
    'cloud_cover': {
        'standard_name': 'cloud_area_fraction',
        'units': '%',
    },
}
MINIMUM_REQUIRED_METEO_VARS = ['temp', 'precip', 'rel_hum', 'sw_in', 'wind_speed']

NETCDF_VAR_MAPPINGS = {
    'tas': 'temp',
    'pr': 'precip',
    'hurs': 'rel_hum',
    'rsds': 'sw_in',
    'wss': 'wind_speed',
    'wind_dir': 'wind_dir',
}

# Mappings of internal variable names to interpolation config keys (e.g.
# config['meteo']['interpolation']['temperature'])
INTERPOLATION_CONFIG_PARAM_MAPPINGS = {
    'temp': 'temperature',
    'precip': 'precipitation',
    'rel_hum': 'humidity',
}

ALLOWED_METEO_VAR_RANGES = {
    'temp': (0, None),
    'precip': (0, None),
    'rel_hum': (0, 100),
    'sw_in': (0, None),
    'wind_speed': (0.1, None),
    'cloud_cover': (0, 100),
}
