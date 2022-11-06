import numpy as np
from openamundsen import constants, interpolation, meteo, util


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
    data,
    xs,
    ys,
    zs,
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
    data : ndarray
        Values to be interpolated.

    xs, ys, zs : ndarray
        x, y and z coordinates of the stations.

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


def interpolate_station_data(model):
    """
    Interpolate station measurements to the model grid.

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.
    """
    model.logger.debug('Interpolating station data')

    date = model.date
    roi = model.grid.roi
    target_xs = model.grid.roi_points[:, 0]
    target_ys = model.grid.roi_points[:, 1]
    target_zs = model.state.base.dem[roi]

    for param in ('temp', 'precip'):
        param_config = model.config['meteo']['interpolation'][constants.INTERPOLATION_CONFIG_PARAM_MAPPINGS[param]]
        data, xs, ys, zs = _param_station_data(model.meteo, param, date)

        model.state.meteo[param][roi] = interpolate_param(
            param,
            model.date,
            param_config,
            data,
            xs,
            ys,
            zs,
            target_xs,
            target_ys,
            target_zs,
        )

    # For relative humidity dew point temperature is interpolated, so we also need station
    # temperatures and the interpolated temperature field
    param = 'rel_hum'
    param_config = model.config['meteo']['interpolation'][constants.INTERPOLATION_CONFIG_PARAM_MAPPINGS[param]]
    data, xs, ys, zs = _param_station_data(model.meteo, [param, 'temp'], date)
    rel_hums = data[0, :]
    temps = data[1, :]
    model.state.meteo[param][roi] = interpolate_param(
        param,
        model.date,
        param_config,
        rel_hums,
        xs,
        ys,
        zs,
        target_xs,
        target_ys,
        target_zs,
        temps=temps,
        target_temps=model.state.meteo['temp'][roi],
    )

    wind_config = model.config['meteo']['interpolation']['wind']
    if wind_config['method'] == 'idw':
        param = 'wind_speed'
        data, xs, ys, zs = _param_station_data(model.meteo, param, date)
        model.state.meteo[param][roi] = interpolate_param(
            param,
            model.date,
            wind_config,
            data,
            xs,
            ys,
            zs,
            target_xs,
            target_ys,
            target_zs,
        )
    elif wind_config['method'] == 'liston':
        data, xs, ys, zs = _param_station_data(model.meteo, ['wind_speed', 'wind_dir'], date)
        wind_speeds = data[0, :]
        wind_dirs = data[1, :]
        wind_us, wind_vs = meteo.wind_to_uv(wind_speeds, wind_dirs)
        wind_u_roi = interpolate_param(
            'wind_vec',
            model.date,
            wind_config,
            wind_us,
            xs,
            ys,
            zs,
            target_xs,
            target_ys,
            target_zs,
        )
        wind_v_roi = interpolate_param(
            'wind_vec',
            model.date,
            wind_config,
            wind_vs,
            xs,
            ys,
            zs,
            target_xs,
            target_ys,
            target_zs,
        )
        wind_speed_roi, wind_dir_roi = meteo.wind_from_uv(wind_u_roi, wind_v_roi)
        wind_dir_minus_aspect = np.deg2rad(wind_dir_roi - model.state.base.aspect[roi])
        wind_slope = (  # slope in the direction of the wind (eq. (15))
            np.deg2rad(model.state.base.slope[roi]) * np.cos(wind_dir_minus_aspect)
        )
        wind_slope_scaled = util.normalize_array(wind_slope, -0.5, 0.5)
        wind_weighting_factor = (  # eq. (16)
            1
            + wind_config.slope_weight * wind_slope_scaled
            + wind_config.curvature_weight * model.state.base.scaled_curvature[roi]
        )
        wind_dir_diverting_factor = (  # eq. (18)
            -0.5 * wind_slope_scaled * np.sin(-2 * wind_dir_minus_aspect)
        )
        wind_speed_corr = wind_speed_roi * wind_weighting_factor
        wind_min_range, wind_max_range = constants.ALLOWED_METEO_VAR_RANGES['wind_speed']
        wind_speed_corr = np.clip(wind_speed_corr, wind_min_range, wind_max_range)
        model.state.meteo.wind_speed[roi] = wind_speed_corr
        model.state.meteo.wind_dir[roi] = (wind_dir_roi + wind_dir_diverting_factor) % 360
    else:
        raise NotImplementedError(f'Unsupported method: {wind_config.method}')


def interpolate_param(
        param,
        date,
        param_config,
        data,
        xs,
        ys,
        zs,
        target_xs,
        target_ys,
        target_zs,
        temps=None,
        target_temps=None,
):
    """
    Interpolate a set of data points to a set of target points.

    Parameters
    ----------
    param : str
        Parameter to be interpolated. Can be one of 'temp', 'precip', 'rel_hum', 'wind_speed'.

    date : datetime-like
        Date (required in the case of seasonally specified lapse rates).

    param_config : dict
        Interpolation configuration for the respective parameter from the model run configuration
        (e.g., config['meteo']['interpolation'][‘temperature']).

    data : ndarray
        Values to be interpolated.

    xs, ys, zs : ndarray
        x, y and z coordinates of the stations.

    target_xs, target_ys, target_zs : ndarray
        x, y and z coordinates of the interpolation targets.

    temps : ndarray, default None
        Temperatures at the source locations (required for humidity interpolation).

    target_temps : ndarray, default None
        Temperatures at the target locations (required for humidity interpolation).

    Returns
    -------
    data_interpol : ndarray
        Interpolated values for the target locations.
    """
    # If there are no points to be interpolated, return an all-zero array for precipitation, and an
    # all-nan array for all other parameters
    if data.size == 0:
        if param == 'precip':
            fill_value = 0.
        else:
            fill_value = np.nan

        return np.full(target_xs.shape, fill_value)

    if param in ('temp', 'precip', 'rel_hum', 'wind_speed', 'wind_vec'):
        trend_method = param_config['trend_method']
        lapse_rate = param_config['lapse_rate'][date.month - 1]
    else:
        raise NotImplementedError(f'Unsupported parameter: {param}')

    # For relative humidity interpolate dew point temperature and convert back to humidity later
    if param == 'rel_hum':
        if temps is None or target_temps is None:
            raise Exception('Temperature must be provided for humidity interpolation')

        rel_hums = data
        dew_point_temps = meteo.dew_point_temperature(temps, rel_hums)
        data = dew_point_temps

    data_interpol = _interpolate_with_trend(
        data,
        xs,
        ys,
        zs,
        target_xs,
        target_ys,
        target_zs,
        trend_method,
        lapse_rate,
    )

    if param == 'rel_hum':
        target_dew_point_temps = data_interpol
        vapor_press = meteo.saturation_vapor_pressure(target_dew_point_temps, 'water')
        sat_vapor_press = meteo.saturation_vapor_pressure(target_temps, 'water')
        data_interpol = 100 * vapor_press / sat_vapor_press

    # Restrict interpolated values to the range of the point values if extrapolation is disabled in
    # the parameter config
    if not param_config['extrapolate'] and len(data) > 0:
        min_range = np.nanmin(data)
        max_range = np.nanmax(data)

        if np.isfinite(min_range):
            data_interpol = np.clip(data_interpol, min_range, max_range)

    if param in constants.ALLOWED_METEO_VAR_RANGES:
        min_range, max_range = constants.ALLOWED_METEO_VAR_RANGES[param]
        data_interpol = np.clip(data_interpol, min_range, max_range)

    return data_interpol
