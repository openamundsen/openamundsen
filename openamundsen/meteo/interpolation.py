import numpy as np
from openamundsen import constants, interpolation, meteo


def _param_station_data(ds, param, date):
    """
    Return station measurements including station x, y and z coordinates for a
    given parameter and date.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the station measurements (i.e. the model.meteo
        variable)

    param : str
        Parameter for which the measurements should be returned.

    date : datetime-like
        Date for which the measurements should be returned.

    Returns
    -------
    data, x, y, z : ndarrays
        Measurements (excluding nodata values) and corresponding x, y and z
        coordinates.
    """
    data = ds[param].sel(time=date).dropna(dim='station')
    xs = ds['x'].sel(station=data['station'])
    ys = ds['y'].sel(station=data['station'])
    zs = ds['alt'].sel(station=data['station'])
    return data.values, xs.values, ys.values, zs.values


def _linear_fit(x, y):
    """
    Calculate the linear regression for a set of sample points.

    Parameters
    ----------
    x, y : ndarray
        x and y coordinates of the sample points.

    Returns
    -------
    slope, intercept : float
    """
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def _detrend(data, elevs, factor, method='linear'):
    """
    Perform elevation detrending for a set of measurements.

    Parameters
    ----------
    data : ndarray
        Values to be detrended.

    elevs : ndarray
        Elevations corresponding to the data points.

    factor : float
        Lapse rate.

    method : str, default "linear"
        Detrending method.

    Returns
    -------
    data_detrended : ndarray
        Detrended data values.
    """
    if method == 'linear':
        data_detrended = data - factor * elevs
    else:
        raise NotImplementedError

    return data_detrended


def _apply_trend(data, elevs, factor, method='linear'):
    """
    Reapply a trend to a detrended set of data points.

    Parameters
    ----------
    data : ndarray
        Detrended values.

    elevs : ndarray
        Elevations corresponding to the data points.

    factor : float
        Lapse rate.

    method : str, default "linear"
        Detrending method.

    Returns
    -------
    data_trend : ndarray
        Data points with reapplied trend.
    """
    if method == 'linear':
        data_trend = data + factor * elevs
    else:
        raise NotImplementedError

    return data_trend


def _detrend_and_interpolate(xs, ys, zs, data, target_xs, target_ys, target_zs):
    slope, _ = _linear_fit(zs, data)
    data_detrended = _detrend(data, zs, slope)
    data_detrended_interpol = interpolation.idw(
        xs,
        ys,
        data_detrended,
        target_xs,
        target_ys,
    )
    data_interpol = _apply_trend(
        data_detrended_interpol,
        target_zs,
        slope,
    )
    return data_interpol


def interpolate_station_data(model, date):
    """
    Interpolate station measurements to the model grid.

    Parameters
    ----------
    model : Model
        Model instance.

    date : datetime-like
        Date for which to perform the interpolation.
    """
    model.logger.debug('Interpolating station data')

    roi = model.grid.roi

    target_xs = model.grid.roi_points[:, 0]
    target_ys = model.grid.roi_points[:, 1]
    target_zs = model.state.base.dem[roi]

    for param in ('temp', 'precip', 'wind_speed'):
        data, xs, ys, zs = _param_station_data(model.meteo, param, date)
        data_interpol = _detrend_and_interpolate(xs, ys, zs, data, target_xs, target_ys, target_zs)
        model.state.meteo[param][roi] = data_interpol[:]

    # Interpolate absolute humidity, convert back to relative humidity
    ds = model.meteo[['temp', 'rel_hum', 'x', 'y', 'alt']].sel(time=date).dropna(dim='station')
    xs = ds.x.values
    ys = ds.y.values
    zs = ds.alt.values
    temps = ds.temp.values
    rel_hums = ds.rel_hum.values
    vapor_pressures = meteo.vapor_pressure(temps, rel_hums)
    abs_hums = meteo.absolute_humidity(temps, vapor_pressures)
    abs_hum_interpol = _detrend_and_interpolate(xs, ys, zs, abs_hums, target_xs, target_ys, target_zs)
    rel_hum_interpol = meteo.relative_humidity(model.state.meteo['temp'][roi], abs_hum_interpol)
    model.state.meteo['rel_hum'][roi] = rel_hum_interpol[:]

    for param in ('temp', 'precip', 'rel_hum', 'wind_speed'):
        data = model.state.meteo[param]
        min_range, max_range = constants.ALLOWED_METEO_VAR_RANGES[param]
        data[roi] = np.clip(data[roi], min_range, max_range)
