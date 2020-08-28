import numpy as np
from openamundsen import constants, interpolation, meteo


def _param_station_data(ds, param, date):
    """
    Return station measurements including station x, y and z coordinates for one or more given
    parameter(s) and a specified date.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the station measurements (i.e. the model.meteo
        variable)

    param : str or list
        Parameter(s) for which the measurements should be returned.

    date : datetime-like
        Date for which the measurements should be returned.

    Returns
    -------
    data : ndarray
        Measurements (excluding nodata values). If param is a string, this is a 1-D array; if param
        is a list of strings it is a 2-D array with dimensions
        (len(param), <number of available measurements where all parameters are non-null>).
    x, y, z : ndarrays
        Corresponding x, y and z coordinates.
    """
    if isinstance(param, str):
        param_as_list = [param]
    else:
        param_as_list = param

    ds_param = ds[param_as_list + ['x', 'y', 'alt']].sel(time=date).dropna(dim='station')
    data = ds_param[param]
    xs = ds_param['x'].values
    ys = ds_param['y'].values
    zs = ds_param['alt'].values

    if isinstance(param, str):
        data_vals = data.values
    else:
        data_vals = data.to_array().values

    return data_vals, xs, ys, zs


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
    if len(x) < 2:
        slope = 0
        intercept = 0
    else:
        slope, intercept = np.polyfit(x, y, 1)

    return slope, intercept


def _apply_linear_trend(data, elevs, trend, direction):
    """
    Detrend or retrend a set of data points using a given trend value.

    Parameters
    ----------
    data : ndarray
        Detrended values.

    elevs : ndarray
        Elevations corresponding to the data points.

    trend : float
        Trend value.

    direction : str
        Either "detrend" or "retrend".

    Returns
    -------
    data_trend : ndarray
        De-/retrended data points.
    """
    if direction == 'detrend':
        return data - trend * elevs
    elif direction == 'retrend':
        return data + trend * elevs
    else:
        raise NotImplementedError(f'Unsupported direction: {direction}')


def _interpolate_with_trend(
    xs,
    ys,
    zs,
    data,
    target_xs,
    target_ys,
    target_zs,
    trend_method,
    lapse_rate,
):
    """
    Interpolate station measurements using inverse distance weighting with an
    elevation-dependent trend.

    Parameters
    ----------
    xs, ys, zs : ndarray
        x, y and z coordinates of the stations.

    data : ndarray
        Values to be interpolated.

    target_xs, target_ys, target_zs : ndarray
        x, y and z coordinates of the interpolation targets.

    trend_method : string
        Method to use for de-/retrending. Either 'regression' (to derive a
        trend from the data points using linear regression), 'fixed' (to use a
        prescribed lapse rate as linear trend), 'fractional' (to use a
        prescribed fractional lapse rate (in case of precipitation) as linear
        trend) or 'adjustment_factor' (to use a nonlinear precipitation
        adjustment factor following [1]).

    lapse_rate : float
        Trend value. If trend_method == 'regression', this value is ignored.
        If trend_method == 'fixed', the value is interpreted as absolute change
        by elevation (e.g. °C m-1).  If trend_method == 'fractional', the value
        is interpreted as a fractional change by elevation (e.g., 0.0005 for
        0.05 % m-1) If trend_method == 'adjustment_factor', the value is
        interpreted as a nonlinear precipitation adjustment factor following
        [1].

    Returns
    -------
    data_interpol : ndarray
        Interpolated values for the target locations.

    References
    ----------
    .. [1] Liston, G. E., & Elder, K. (2006). A Meteorological Distribution
       System for High-Resolution Terrestrial Modeling (MicroMet). Journal of
       Hydrometeorology, 7(2), 217–234. https://doi.org/10.1175/JHM486.1
    """
    if trend_method in ('regression', 'fixed', 'fractional'):
        if trend_method == 'regression':
            # When using the regression method, the passed lapse rate is overwritten
            lapse_rate, _ = _linear_fit(zs, data)
        elif trend_method == 'fixed':
            pass  # do nothing, i.e., use the passed lapse rate as is
        elif trend_method == 'fractional':
            lapse_rate *= np.nanmean(data)

        data_detrended = _apply_linear_trend(data, zs, lapse_rate, 'detrend')
        data_detrended_interpol = interpolation.idw(
            xs,
            ys,
            data_detrended,
            target_xs,
            target_ys,
        )
        data_interpol = _apply_linear_trend(data_detrended_interpol, target_zs, lapse_rate, 'retrend')
    elif trend_method == 'adjustment_factor':
        data_interpol_notrend = interpolation.idw(xs, ys, data, target_xs, target_ys)
        zs_interpol = interpolation.idw(xs, ys, zs, target_xs, target_ys)
        z_diffs = target_zs - zs_interpol
        data_interpol = data_interpol_notrend * (  # eq. (33) from [1]
            (1 + lapse_rate * z_diffs) / (1 - lapse_rate * z_diffs)
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

    # Mappings of config keys (e.g. meteo.interpolation.temperature) to
    # internal variable names
    param_name_mappings = {
        'temp': 'temperature',
        'precip': 'precipitation',
        'wind_speed': 'wind_speed',
    }

    for param in ('temp', 'precip', 'wind_speed'):
        if param == 'wind_speed':  # for wind speed fixed lapse rates are not supported
            trend_method = 'regression'
            lapse_rate = None
        else:
            param_config = model.config['meteo']['interpolation'][param_name_mappings[param]]
            trend_method = param_config['trend_method']
            lapse_rate = param_config['lapse_rate'][date.month - 1]

        data, xs, ys, zs = _param_station_data(model.meteo, param, date)
        data_interpol = _interpolate_with_trend(
            xs,
            ys,
            zs,
            data,
            target_xs,
            target_ys,
            target_zs,
            trend_method,
            lapse_rate,
        )
        model.state.meteo[param][roi] = data_interpol[:]

    # For humidity interpolate dewpoint temperature and convert back to relative humidity
    ds = model.meteo[['temp', 'rel_hum', 'x', 'y', 'alt']].sel(time=date).dropna(dim='station')
    xs = ds.x.values
    ys = ds.y.values
    zs = ds.alt.values
    temps = ds.temp.values
    rel_hums = ds.rel_hum.values
    dewpoint_temps = meteo.dew_point_temperature(temps, rel_hums)
    param_config = model.config['meteo']['interpolation']['humidity']
    trend_method = param_config['trend_method']
    lapse_rate = param_config['lapse_rate'][date.month - 1]
    dewpoint_temp_interpol = _interpolate_with_trend(
        xs,
        ys,
        zs,
        dewpoint_temps,
        target_xs,
        target_ys,
        target_zs,
        trend_method,
        lapse_rate,
    )
    vapor_press_roi = meteo.saturation_vapor_pressure(dewpoint_temp_interpol, 'water')
    sat_vapor_press_roi = meteo.saturation_vapor_pressure(model.state.meteo['temp'][roi], 'water')
    model.state.meteo['rel_hum'][roi] = 100 * vapor_press_roi / sat_vapor_press_roi

    for param in ('temp', 'precip', 'rel_hum', 'wind_speed'):
        data = model.state.meteo[param]
        min_range, max_range = constants.ALLOWED_METEO_VAR_RANGES[param]
        data[roi] = np.clip(data[roi], min_range, max_range)
