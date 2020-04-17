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
