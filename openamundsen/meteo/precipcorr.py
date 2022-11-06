import numpy as np
from openamundsen import constants, interpolation, meteo


# Transfer function coefficents for 10 m wind speeds from [2] (Table 3).
# Values in the tuples correspond to (a, b, c, max_U).
KOCHENDORFER_COEFFS = {
    'us_un': (0.045, 1.21, 0.66, 8),
    'all_sa': (0.03, 1.04, 0.57, 12),
    'nor_sa': (0.05, 0.66, 0.23, 12),
    'us_sa': (0.03, 1.06, 0.63, 12),
    'us_da': (0.021, 0.74, 0.66, 8),
    'us_bda': (0.01, 0.48, 0.51, 8),
    'us_sdfir': (0.004, 0.00, 0.00, 8),
}


def correct_station_precipitation(model):
    """
    Correct station precipitation recordings for wind-induced undercatch.

    Precipitation can be corrected using the following `method`s in the model
    configuration (config['meteo']['precipitation_correction']):

        - 'constant_scf': Snowfall is corrected using a constant snow correction
          factor (scaled accordingly for mixed precipitation).
        - 'wmo': Snowfall and mixed precipitation are corrected using the
          transfer functions derived for different gauges from [1].
        - 'kochendorfer': Precipitation (regardless of phase) is corrected using
          the transfer frunctions derived for different gauges from [2].

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    References
    ----------
    .. [1] Goodison, B. E., Louie, P., & Yang, D. (1998). WMO solid
       precipitation measurement intercomparison (World Meteorological
       Organization, p. 212). World Meteorological Organization.
       http://www.wmo.int/pages/prog/www/reports/WMOtd872.pdf

    .. [2] Kochendorfer, J., Rasmussen, R., Wolff, M., Baker, B., Hall, M. E.,
       Meyers, T., Landolt, S., Jachcik, A., Isaksen, K., Brækkan, R., & Leeper, R.
       (2017). The quantification and correction of wind-induced precipitation
       measurement errors. Hydrology and Earth System Sciences, 21(4), 1973–1989.
       https://doi.org/10.5194/hess-21-1973-2017
    """
    precip_corr_configs = model.config.meteo.precipitation_correction

    if len(precip_corr_configs) == 0:
        return

    m = model.meteo
    dates = m.indexes['time']
    xs = m['x'].values
    ys = m['y'].values
    zs = m['alt'].values
    temps = m['temp'].values.T.copy()
    rel_hums = m['rel_hum'].values.T.copy()
    wind_speeds = m['wind_speed'].values.T.copy()
    precips = m['precip'].values.T  # no copy, because this will be modified in place

    # Interpolate missing temperature, humidity and wind speed values
    # TODO this is only really necessary for nonzero precipitation values; could be optimized
    _interpolate_temp_hum_wind(
        dates,
        xs,
        ys,
        zs,
        temps,
        rel_hums,
        wind_speeds,
        model.config.meteo.interpolation.temperature,
        model.config.meteo.interpolation.humidity,
        model.config.meteo.interpolation.wind,
    )

    temps_c = temps - constants.T0

    # Calculate precipitation phase (if required, calculate wet-bulb temperature first)
    precip_phase_method = model.config.meteo.precipitation_phase.method
    if precip_phase_method == 'temp':
        pp_temp = temps
    elif precip_phase_method == 'wet_bulb_temp':
        atmos_presss = meteo.atmospheric_pressure(zs)
        vap_presss = meteo.vapor_pressure(temps, rel_hums)
        spec_hums = meteo.specific_humidity(atmos_presss, vap_presss)
        spec_heat_caps_moist_air = meteo.specific_heat_capacity_moist_air(spec_hums)
        lat_heat_vaps = meteo.latent_heat_of_vaporization(temps)
        psych_consts = meteo.psychrometric_constant(
            atmos_presss,
            spec_heat_caps_moist_air,
            lat_heat_vaps,
        )
        wet_bulb_temps = meteo.wet_bulb_temperature(
            temps,
            rel_hums,
            vap_presss,
            psych_consts,
        )
        pp_temp = wet_bulb_temps

    snowfall_frac = meteo.precipitation_phase(
        pp_temp,
        threshold_temp=model.config.meteo.precipitation_phase.threshold_temp,
        temp_range=model.config.meteo.precipitation_phase.temp_range,
    )

    pos_precip = precips > 0
    pos_snow = pos_precip & (snowfall_frac == 1)
    pos_mixed = pos_precip & (snowfall_frac > 0) & (snowfall_frac < 1)

    for config in precip_corr_configs:
        method = config['method']

        if method not in (
            'constant_scf',
            'wmo',
            'kochendorfer',
        ):
            continue

        model.logger.info(f'Correcting station precipitation with method: {method}')

        if method == 'constant_scf':
            precips *= snowfall_frac * config['scf']
        elif method == 'wmo':
            gauge = config['gauge']
            cr = np.full(precips.shape, 100.)

            # Calculate wind speed at gauge height
            wind_speeds_2m = meteo.log_wind_profile(
                wind_speeds,
                model.config.meteo.measurement_height.wind,
                model.config.meteo.measurement_height.temperature,
                model.config.snow.roughness_length,
            )

            # Correction functions have been derived for wind speeds < 7 m s-1, so we do not allow
            # higher values here
            wind_speeds_2m = wind_speeds_2m.clip(max=7)

            # Some correction functions from [1] use Tmin and Tmax values; these have been replaced
            # by standard air temperature in the following.
            if gauge == 'nipher':
                cr[pos_snow] = (
                    100
                    - 0.44 * wind_speeds_2m[pos_snow]**2
                    - 1.98 * wind_speeds_2m[pos_snow]
                )
                cr[pos_mixed] = (
                    97.29
                    - 3.18 * wind_speeds_2m[pos_mixed]
                    + (0.58 - 0.67) * temps_c[pos_mixed]
                )
            elif gauge == 'tretyakov':
                cr[pos_snow] = 103.11 - 8.67 * wind_speeds_2m[pos_snow] + 0.30 * temps_c[pos_snow]
                cr[pos_mixed] = (
                    96.99
                    - 4.46 * wind_speeds_2m[pos_mixed]
                    + (0.88 + 0.22) * temps_c[pos_mixed]
                )
            elif gauge == 'us_sh':
                cr[pos_snow] = np.exp(4.61 - 0.04 * wind_speeds_2m[pos_snow]**1.75)
                cr[pos_mixed] = 101.04 - 5.62 * wind_speeds_2m[pos_mixed]
            elif gauge == 'us_unsh':
                cr[pos_snow] = np.exp(4.61 - 0.16 * wind_speeds_2m[pos_snow]**1.28)
                cr[pos_mixed] = 100.77 - 8.34 * wind_speeds_2m[pos_mixed]
            if gauge == 'hellmann':
                cr[pos_snow] = (
                    100
                    + 1.13 * wind_speeds_2m[pos_snow]**2
                    - 19.45 * wind_speeds_2m[pos_snow]
                )
                cr[pos_mixed] = (
                    96.63
                    + 0.41 * wind_speeds_2m[pos_mixed]**2
                    - 9.84 * wind_speeds_2m[pos_mixed]
                    + 5.95 * temps_c[pos_mixed]
                )
            else:
                raise NotImplementedError(f'Unknown gauge: {gauge}')
            
            precips /= (cr / 100.)
        elif method == 'kochendorfer':
            gauge = config['gauge']
            cr = np.full(precips.shape, 1.)  # CR is here not in percent but a fraction

            try:
                coeff_a, coeff_b, coeff_c, max_wind_speed = KOCHENDORFER_COEFFS[gauge]
            except KeyError:
                raise NotImplementedError(f'Unknown gauge: {gauge}')

            # Calculate catch ratio following [2] (use eq. (1) from the corrigendum, not eq. (4)
            # from the original paper)
            cr[pos_precip] = np.exp(
                -coeff_a * wind_speeds[pos_precip].clip(max=max_wind_speed)
                * (1 - np.arctan(coeff_b * temps_c[pos_precip]) + coeff_c)
            )

            precips /= cr


def _interpolate_temp_hum_wind(
    dates,
    xs,
    ys,
    zs,
    temps,
    rel_hums,
    wind_speeds,
    temp_config,
    hum_config,
    wind_speed_config,
):
    """
    Interpolate air temperature, relative humidity and wind speed recordings
    from the meteorological stations to the locations of the stations missing
    these recordings.
    """
    num_stations = len(xs)

    for date_num, date in enumerate(dates):
        nan_pos = np.isnan(temps[date_num, :])
        num_nan = nan_pos.sum()
        if num_nan > 0 and num_nan < num_stations:
            temps[date_num, nan_pos] = meteo.interpolate_param(
                'temp',
                date,
                temp_config,
                temps[date_num, ~nan_pos],
                xs[~nan_pos],
                ys[~nan_pos],
                zs[~nan_pos],
                xs[nan_pos],
                ys[nan_pos],
                zs[nan_pos],
            )

        nan_pos = np.isnan(rel_hums[date_num, :])
        num_nan = nan_pos.sum()
        if num_nan > 0 and num_nan < num_stations:
            rel_hums[date_num, nan_pos] = meteo.interpolate_param(
                'rel_hum',
                date,
                hum_config,
                rel_hums[date_num, ~nan_pos],
                xs[~nan_pos],
                ys[~nan_pos],
                zs[~nan_pos],
                xs[nan_pos],
                ys[nan_pos],
                zs[nan_pos],
                temps=temps[date_num, ~nan_pos],
                target_temps=temps[date_num, nan_pos],
            )

        nan_pos = np.isnan(wind_speeds[date_num, :])
        num_nan = nan_pos.sum()
        if num_nan > 0 and num_nan < num_stations:
            wind_speeds[date_num, nan_pos] = meteo.interpolate_param(
                'wind_speed',
                date,
                temp_config,
                wind_speeds[date_num, ~nan_pos],
                xs[~nan_pos],
                ys[~nan_pos],
                zs[~nan_pos],
                xs[nan_pos],
                ys[nan_pos],
                zs[nan_pos],
            )
