DAYS_PER_YEAR = 365.25
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600

STANDARD_TIME_ZONE_WIDTH = 15
SUN_DEGREES_PER_HOUR = 15  # 360/24
MINUTES_PER_DEGREE_OF_EARTH_ROTATION = 4  # earth rotates 1 degree every 4 minutes
SOLAR_CONSTANT = 1366.1  # W m-2

METEO_VAR_METADATA = {
    'temp': {
        'standard_name': 'air_temperature',
        'units': 'K',
    },
    'precip': {
        'standard_name': 'precipitation_flux',
        'units': 'kg m-2 s-1',
    },
    'rel_hum': {
        'standard_name': 'relative_humidity',
        'units': '%',
    },
    'shortwave_in': {
        'standard_name': 'surface_downwelling_shortwave_flux_in_air',
        'units': 'W m-2',
    },
    'wind_speed': {
        'standard_name': 'wind_speed',
        'units': 'm s-1',
    },
}

NETCDF_VAR_MAPPINGS = {
    'tas': 'temp',
    'pr': 'precip',
    'hurs': 'rel_hum',
    'rsds': 'shortwave_in',
    'wss': 'wind_speed',
}

ALLOWED_METEO_VAR_RANGES = {
    'temp': (-273.15, None),
    'precip': (0, None),
    'rel_hum': (0, 100),
    'shortwave_in': (0, None),
    'wind_speed': (0, None),
}
