import numpy as np
from openamundsen import constants, interpolation
import scipy.stats


def _param_station_data(ds, param, date):
    data = ds[param].sel(time=date).dropna(dim='station')
    xs = ds['x'].sel(station=data['station'])
    ys = ds['y'].sel(station=data['station'])
    zs = ds['alt'].sel(station=data['station'])
    return data.values, xs.values, ys.values, zs.values


def _linear_fit(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return slope


def _detrend(data, elevs, factor, method='linear'):
    if method == 'linear':
        data_detrended = data - factor * elevs
    else:
        raise NotImplementedError

    return data_detrended


def _apply_trend(data, elevs, factor, method='linear'):
    if method == 'linear':
        data_trended = data + factor * elevs
    else:
        raise NotImplementedError

    return data_trended


def interpolate_station_data(model, date):
    model.logger.debug('Interpolating station data')

    roi = model.state.base.roi

    for param in ('temp', 'precip', 'rel_hum', 'wind_speed'):
        data, xs, ys, zs = _param_station_data(model.meteo, param, date)
        slope = _linear_fit(zs, data)
        data_detrended = _detrend(data, zs, slope)
        data_detrended_interpol = interpolation.idw(
            xs,
            ys,
            data_detrended,
            model.grid.roi_points[:, 0],
            model.grid.roi_points[:, 1],
        )
        data_interpol = _apply_trend(
            data_detrended_interpol,
            model.state.base.dem[model.state.base.roi],
            slope,
        )

        min_range, max_range = constants.ALLOWED_METEO_VAR_RANGES[param]
        data_interpol_clipped = np.clip(data_interpol, min_range, max_range)

        model.state.meteo[param][roi] = data_interpol_clipped[:]


def process_meteo_data(model):
    model.logger.debug('Processing meteorological fields')
