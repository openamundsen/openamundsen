from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pwlf
from loguru import logger

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

    ds_param = ds[[*param_as_list, "x", "y", "alt"]].sel(time=date).dropna(dim="station")
    data = ds_param[param]
    xs = ds_param["x"].values
    ys = ds_param["y"].values
    zs = ds_param["alt"].values

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


def _piecewise_linear_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    max_segments: int,
    min_points_per_segment: int,
    delta_aic_threshold: float,
) -> dict | None:
    best = None

    for num_segments in range(1, max_segments + 1):
        # Stop if it is impossible to allocate min_points_per_segment to each segment
        if num_segments > 1 and len(xs) < num_segments * min_points_per_segment:
            break

        pw = pwlf.PiecewiseLinFit(xs, ys)
        try:
            pw.fit(
                num_segments,
                seed=42,
                maxiter=50,
                popsize=15,
            )
        except (ValueError, ZeroDivisionError):
            continue

        breaks = pw.fit_breaks

        # Ensure there are at least min_points_per_segment points in each segment (except for the
        # single-segment case)
        if num_segments > 1 and any(
            np.sum((xs >= lo) & (xs <= hi)) < min_points_per_segment
            for lo, hi in zip(breaks[:-1], breaks[1:])
        ):
            break

        preds = pw.predict(xs)
        rss = np.sum((ys - preds) ** 2)
        n = len(ys)
        k = 2 * num_segments
        aic = n * np.log(rss / n) + 2 * k

        model_params = {
            "model": pw,
            "breaks": breaks,
            "aic": aic,
        }

        if best is None:
            best = model_params
            continue

        model_has_improved = (best["aic"] - aic) >= delta_aic_threshold
        if model_has_improved:
            best = model_params
        else:
            # If the model has not improved compared to the earlier one we stop, since it is
            # unlikely in this case that the next one will improve the results
            break

    return best


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
    if direction == "detrend":
        return data - trend * elevs
    elif direction == "retrend":
        return data + trend * elevs
    else:
        raise NotImplementedError(f"Unsupported direction: {direction}")


def _apply_piecewise_trend(
    data: np.ndarray,
    elevs: np.ndarray,
    pw_model: pwlf.PiecewiseLinFit,
    direction: Literal["detrend", "retrend"],
) -> np.ndarray:
    """
    Add or remove a piecewise-linear elevation trend.

    Parameters
    ----------
    data
        Values to (de)trend.
    elevs
        Matching elevations.
    pw_model
        Fitted pwlf.PiecewiseLinFit.
    direction
        Either "detrend" or "retrend".

    Returns
    -------
    np.ndarray
        De-/retrended data points.
    """
    trend_vals = pw_model.predict(elevs)

    if direction == "detrend":
        return data - trend_vals
    elif direction == "retrend":
        return data + trend_vals
    else:
        raise ValueError(f"Unsupported direction: {direction}")


def interpolate_station_data(model):
    """
    Interpolate station measurements to the model grid.

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.
    """
    logger.debug("Interpolating station data")

    date = model.date
    roi = model.grid.roi
    target_xs = model.grid.roi_points[:, 0]
    target_ys = model.grid.roi_points[:, 1]
    target_zs = model.state.base.dem[roi]

    model._interpolation = {}

    for param in ("temp", "precip"):
        param_config = model.config["meteo"]["interpolation"][
            constants.INTERPOLATION_CONFIG_PARAM_MAPPINGS[param]
        ]
        data, xs, ys, zs = _param_station_data(model.meteo, param, date)

        model.state.meteo[param][roi], model._interpolation[param] = interpolate_param(
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
    param = "rel_hum"
    param_config = model.config["meteo"]["interpolation"][
        constants.INTERPOLATION_CONFIG_PARAM_MAPPINGS[param]
    ]
    data, xs, ys, zs = _param_station_data(model.meteo, [param, "temp"], date)
    rel_hums = data[0, :]
    temps = data[1, :]
    model.state.meteo[param][roi], _ = interpolate_param(
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
        target_temps=model.state.meteo["temp"][roi],
    )

    wind_config = model.config["meteo"]["interpolation"]["wind"]
    if wind_config["method"] == "idw":
        data, xs, ys, zs = _param_station_data(model.meteo, "wind_speed", date)
        model.state.meteo["wind_speed"][roi], _ = interpolate_param(
            "wind_speed",
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

        if model._has_wind_gusts:
            data, xs, ys, zs = _param_station_data(model.meteo, "wind_speed_gust", date)
            model.state.meteo["wind_speed_gust"][roi], _ = interpolate_param(
                "wind_speed",
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
    elif wind_config["method"] == "liston":
        data, xs, ys, zs = _param_station_data(model.meteo, ["wind_speed", "wind_dir"], date)
        wind_speeds = data[0, :]
        wind_dirs = data[1, :]
        wind_speed_corr, wind_dir_corr = _liston_wind_correction(
            model.date,
            wind_config,
            wind_speeds,
            wind_dirs,
            xs,
            ys,
            zs,
            target_xs,
            target_ys,
            target_zs,
            model.state.base.slope[roi],
            model.state.base.aspect[roi],
            model.state.base.scaled_curvature[roi],
        )
        model.state.meteo.wind_speed[roi] = np.clip(
            wind_speed_corr,
            *constants.ALLOWED_METEO_VAR_RANGES["wind_speed"],
        )
        model.state.meteo.wind_dir[roi] = wind_dir_corr

        if model._has_wind_gusts:
            data, xs, ys, zs = _param_station_data(
                model.meteo,
                ["wind_speed_gust", "wind_dir"],
                date,
            )
            wind_speed_gusts = data[0, :]
            wind_dirs = data[1, :]
            wind_speed_gusts_corr, _ = _liston_wind_correction(
                model.date,
                wind_config,
                wind_speed_gusts,
                wind_dirs,
                xs,
                ys,
                zs,
                target_xs,
                target_ys,
                target_zs,
                model.state.base.slope[roi],
                model.state.base.aspect[roi],
                model.state.base.scaled_curvature[roi],
            )
            model.state.meteo.wind_speed_gust[roi] = np.clip(
                wind_speed_gusts_corr,
                *constants.ALLOWED_METEO_VAR_RANGES["wind_speed_gust"],
            )
    else:
        raise NotImplementedError(f"Unsupported method: {wind_config.method}")


def interpolate_param(
    param: str,
    date: pd.Timestamp,
    param_config: dict,
    data: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    target_xs: np.ndarray,
    target_ys: np.ndarray,
    target_zs: np.ndarray,
    temps: np.ndarray | None = None,
    target_temps: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Interpolate a set of data points to a set of target points.

    Parameters
    ----------
    param : str
        Parameter to be interpolated. Can be one of 'temp', 'precip', 'rel_hum', 'wind_speed',
        'wind_vec'.

    date : datetime-like
        Date (required in the case of seasonally specified lapse rates).

    param_config : dict
        Interpolation configuration for the respective parameter from the model run configuration
        (e.g., config['meteo']['interpolation']['temperature']).

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
    if param not in ("temp", "precip", "rel_hum", "wind_speed", "wind_vec"):
        raise ValueError(f"Unsupported parameter: {param}")

    trend_method = param_config["trend_method"]
    lapse_rate = param_config["lapse_rate"][date.month - 1]

    interpolation_params = {"gradient_type": "none"}

    # If there are no points to be interpolated, return an all-zero array for precipitation, and an
    # all-nan array for all other parameters
    if data.size == 0:
        if param == "precip":
            fill_value = 0.0
        else:
            fill_value = np.nan

        return (np.full(target_xs.shape, fill_value), interpolation_params)

    # For relative humidity interpolate dew point temperature and convert back to humidity later
    if param == "rel_hum":
        if temps is None or target_temps is None:
            raise ValueError("Temperature must be provided for humidity interpolation")

        rel_hums = data
        dew_point_temps = meteo.dew_point_temperature(temps, rel_hums)
        data = dew_point_temps

    regression_params = param_config.get("regression_params", {})
    max_segments = regression_params.get("max_segments", 1)

    if trend_method in ("regression", "fixed", "fractional") and max_segments == 1:
        # Use simple linear regression
        if trend_method == "regression":
            # When using the regression method, the passed lapse rate is overwritten
            lapse_rate, _ = _linear_fit(zs, data)
        elif trend_method == "fixed":
            pass  # do nothing, i.e., use the passed lapse rate as is
        elif trend_method == "fractional":
            lapse_rate *= np.nanmean(data)

        data_detrended = _apply_linear_trend(data, zs, lapse_rate, "detrend")
        data_detrended_interpol = interpolation.idw(
            xs,
            ys,
            data_detrended,
            target_xs,
            target_ys,
        )
        data_interpol = _apply_linear_trend(
            data_detrended_interpol,
            target_zs,
            lapse_rate,
            "retrend",
        )
        interpolation_params.update(
            gradient_type="linear",
            gradient=lapse_rate,
        )
    elif trend_method == "regression" and max_segments > 1:
        # Use piecewise linear regression
        pw_lr = _piecewise_linear_fit(
            zs,
            data,
            regression_params["max_segments"],
            regression_params["min_points_per_segment"],
            regression_params["delta_aic_threshold"],
        )

        if pw_lr is None:
            return (np.full(target_xs.shape, np.nan), interpolation_params)

        pw_model = pw_lr["model"]
        data_detrended = _apply_piecewise_trend(data, zs, pw_model, "detrend")
        data_detrended_interpol = interpolation.idw(
            xs,
            ys,
            data_detrended,
            target_xs,
            target_ys,
        )
        data_interpol = _apply_piecewise_trend(
            data_detrended_interpol,
            target_zs,
            pw_model,
            "retrend",
        )

        interpolation_params["gradient_type"] = "piecewise_linear"
        if pw_lr is not None:
            interpolation_params.update(**pw_lr)
    elif trend_method == "adjustment_factor":
        data_interpol_notrend = interpolation.idw(xs, ys, data, target_xs, target_ys)
        zs_interpol = interpolation.idw(xs, ys, zs, target_xs, target_ys)
        z_diffs = target_zs - zs_interpol
        data_interpol = data_interpol_notrend * (  # eq. (33) from [1]
            (1 + lapse_rate * z_diffs) / (1 - lapse_rate * z_diffs)
        )
        interpolation_params.update(
            gradient_type="adjustment_factor",
            gradient=lapse_rate,
        )
    else:
        raise ValueError

    if param == "rel_hum":
        target_dew_point_temps = data_interpol
        vapor_press = meteo.saturation_vapor_pressure(target_dew_point_temps, "water")
        sat_vapor_press = meteo.saturation_vapor_pressure(target_temps, "water")
        data_interpol = 100 * vapor_press / sat_vapor_press

    # Restrict interpolated values to the range of the point values if extrapolation is disabled in
    # the parameter config
    if not param_config["extrapolate"] and len(data) > 0:
        min_range = np.nanmin(data)
        max_range = np.nanmax(data)

        if np.isfinite(min_range):
            data_interpol = np.clip(data_interpol, min_range, max_range)

    if param in constants.ALLOWED_METEO_VAR_RANGES:
        min_range, max_range = constants.ALLOWED_METEO_VAR_RANGES[param]
        data_interpol = np.clip(data_interpol, min_range, max_range)

    return data_interpol, interpolation_params


def _liston_wind_correction(
    date,
    wind_config,
    wind_speeds,
    wind_dirs,
    xs,
    ys,
    zs,
    target_xs,
    target_ys,
    target_zs,
    target_slopes,
    target_aspects,
    target_scaled_curvatures,
):
    wind_us, wind_vs = meteo.wind_to_uv(wind_speeds, wind_dirs)
    wind_u_roi, _ = interpolate_param(
        "wind_vec",
        date,
        wind_config,
        wind_us,
        xs,
        ys,
        zs,
        target_xs,
        target_ys,
        target_zs,
    )
    wind_v_roi, _ = interpolate_param(
        "wind_vec",
        date,
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
    wind_dir_minus_aspect = np.deg2rad(wind_dir_roi - target_aspects)
    wind_slope = (  # slope in the direction of the wind (eq. (15))
        np.deg2rad(target_slopes) * np.cos(wind_dir_minus_aspect)
    )
    wind_slope_scaled = util.normalize_array(wind_slope, -0.5, 0.5)
    wind_weighting_factor = (  # eq. (16)
        1
        + wind_config.slope_weight * wind_slope_scaled
        + wind_config.curvature_weight * target_scaled_curvatures
    )
    wind_dir_diverting_factor = (  # eq. (18)
        -0.5 * wind_slope_scaled * np.sin(-2 * wind_dir_minus_aspect)
    )
    wind_speed_corr = wind_speed_roi * wind_weighting_factor
    wind_dir_corr = (wind_dir_roi + wind_dir_diverting_factor) % 360

    return wind_speed_corr, wind_dir_corr
