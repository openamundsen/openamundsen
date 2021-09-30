import numpy as np
from .clearsky import _clear_sky_shortwave_irradiance
from openamundsen import (
    constants,
    interpolation,
    meteo,
    modules,
)


def clear_sky_shortwave_irradiance(model):
    roi = model.grid.roi

    # Extend ROI with the station positions (when cloudiness is calculated via 'clear_sky_fraction',
    # clear sky irradiance should be available for all radiation stations, not only the inside-ROI
    # ones)
    station_rows = model.meteo.row[model.meteo.within_grid_extent].values
    station_cols = model.meteo.col[model.meteo.within_grid_extent].values
    roi_plus_stations = roi.copy()
    roi_plus_stations[station_rows, station_cols] = True

    mean_surface_albedo = model.state.surface.albedo[roi].mean()
    if np.isnan(mean_surface_albedo):
        # E.g. in the first timestep no albedo has yet been calculated; assume a default value here
        mean_surface_albedo = model.config.soil.albedo

    if model.sun_params['sun_over_horizon']:
        model.logger.debug('Calculating shadows')
        shadows = modules.radiation.shadows(
            model.state.base.dem,
            model.grid.resolution,
            model.sun_params['sun_vector'],
            num_sweeps=model.config.meteo.radiation.num_shadow_sweeps,
        )

        model.logger.debug('Calculating clear-sky shortwave irradiance')
        dir_irr, diff_irr = _clear_sky_shortwave_irradiance(
            model.sun_params['day_angle'],
            model.sun_params['sun_vector'],
            shadows,
            model.state.base.dem,
            model.state.base.svf,
            model.state.base.normal_vec,
            model.state.meteo.atmos_press,
            model.state.meteo.precipitable_water,
            mean_surface_albedo,
            roi=roi_plus_stations,
            ozone_layer_thickness=model.config.meteo.radiation.ozone_layer_thickness,
            atmospheric_visibility=model.config.meteo.radiation.atmospheric_visibility,
            single_scattering_albedo=model.config.meteo.radiation.single_scattering_albedo,
            clear_sky_albedo=model.config.meteo.radiation.clear_sky_albedo,
        )
    else:
        dir_irr = np.zeros(model.grid.shape)
        diff_irr = np.zeros(model.grid.shape)

    pot_irr = dir_irr + diff_irr
    model.state.meteo.sw_in_clearsky[roi_plus_stations] = pot_irr[roi_plus_stations]
    model.state.meteo.dir_in_clearsky[roi_plus_stations] = dir_irr[roi_plus_stations]
    model.state.meteo.diff_in_clearsky[roi_plus_stations] = diff_irr[roi_plus_stations]


def shortwave_irradiance(model):
    model.logger.debug('Calculating actual shortwave irradiance')
    cloud_config = model.config.meteo.interpolation.cloudiness
    roi = model.grid.roi
    m = model.state.meteo

    # Select stations within the grid extent and with shortwave radiation measurements
    # for the current time step
    ds_rad = (
        model.meteo
        .isel(station=model.meteo.within_grid_extent)
        .sel(time=model.date)
        .dropna('station', subset=['sw_in'])
    )
    num_rad_stations = len(ds_rad.station)

    # Select stations with temperature and humidity measurements for the current time step
    ds_temp_hum = (
        model.meteo
        .sel(time=model.date)
        .dropna('station', how='any', subset=['temp', 'rel_hum'])
    )
    num_temp_hum_stations = len(ds_temp_hum.station)

    method = 'constant'
    if model.sun_params['sun_over_horizon']:
        if cloud_config['day_method'] == 'clear_sky_fraction':
            if num_rad_stations > 0:
                method = 'clear_sky_fraction'
            else:
                if cloud_config['allow_fallback'] and num_temp_hum_stations > 0:
                    model.logger.debug('No radiation measurements available, falling back '
                                      'to humidity-based cloudiness calculation')
                    method = 'humidity'
                else:
                    model.logger.debug('No radiation measurements available, using cloudiness '
                                      'from previous time step')
                    method = 'constant'
        else:
            method = cloud_config['day_method']
    else:
        if cloud_config['night_method'] == 'humidity':
            if num_temp_hum_stations > 0:
                method = 'humidity'
            else:
                model.logger.debug('No temperature and humidity measurements available, using '
                                  'cloudiness from previous time step')
                method = 'constant'

    if method == 'constant':  # use cloudiness from previous time step
        cloud_factor_roi_prev = m.cloud_factor[roi]

        # When there is no cloudiness from the previous time step (i.e. when the model run starts
        # during nighttime) set cloudiness to a constant value
        cloud_factor_roi_prev[np.isnan(cloud_factor_roi_prev)] = 0.75

        m.cloud_factor[roi] = cloud_factor_roi_prev
    elif method == 'humidity':
        lr_t = model.config.meteo.interpolation.temperature.lapse_rate[model.date.month - 1]
        lr_td = model.config.meteo.interpolation.humidity.lapse_rate[model.date.month - 1]
        # TODO use here also the same settings for regression/fixed gradients as for the interpolation
        cloud_fracs = meteo.cloud_fraction_from_humidity(
            ds_temp_hum.temp,
            ds_temp_hum.rel_hum,
            ds_temp_hum.alt,
            lr_t,
            lr_td,
        )
        cloud_factors = meteo.cloud_factor_from_cloud_fraction(cloud_fracs)
        cloud_factor_interpol = interpolation.idw(
            ds_temp_hum.x,
            ds_temp_hum.y,
            cloud_factors,
            model.grid.roi_points[:, 0],
            model.grid.roi_points[:, 1],
        )
        m.cloud_factor[roi] = cloud_factor_interpol.clip(0, 1)
    elif method == 'clear_sky_fraction':
        cloud_factors = ds_rad.sw_in.values / m.sw_in_clearsky[ds_rad.row, ds_rad.col]
        cloud_factor_interpol = interpolation.idw(
            ds_rad.x,
            ds_rad.y,
            cloud_factors,
            model.grid.roi_points[:, 0],
            model.grid.roi_points[:, 1],
        )
        m.cloud_factor[roi] = cloud_factor_interpol.clip(0, 1)

    m.cloud_fraction[roi] = meteo.cloud_fraction_from_cloud_factor(m.cloud_factor[roi])
    m.sw_in[roi] = m.sw_in_clearsky[roi] * m.cloud_factor[roi]


def longwave_irradiance(model):
    model.logger.debug('Calculating longwave irradiance')
    roi = model.grid.roi
    m = model.state.meteo
    clear_sky_emissivity = meteo.clear_sky_emissivity(m.precipitable_water[roi])

    # TODO these should be parameters
    cloud_emissivity = 0.976  # emissivity of totally overcast skies (Greuell et al., 1997)
    rock_emission_factor = 0.01  # (K W-1 m2) temperature of emitting rocks during daytime is assumed to be higher than the air temperature by this factor multiplied by the incoming shortwave radiation (Greuell et al., 1997)

    # Incoming longwave radiation from the clear sky
    lw_in_clearsky = (
        clear_sky_emissivity
        * constants.STEFAN_BOLTZMANN
        * m.temp[roi]**4
        * model.state.base.svf[roi]
        * (1 - m.cloud_fraction[roi]**2)
    )

    # Incoming longwave radiation from clouds
    lw_in_clouds = (
        cloud_emissivity
        * constants.STEFAN_BOLTZMANN
        * m.temp[roi]**4
        * model.state.base.svf[roi]
        * m.cloud_fraction[roi]**2
    )

    # Incoming longwave radiation from surrounding slopes
    snowfree_count = (model.state.snow.swe[roi] == 0).sum()
    rock_fraction = snowfree_count / model.grid.roi.sum()
    lw_in_slopes = (
        constants.STEFAN_BOLTZMANN
        * (1 - model.state.base.svf[roi]) * (
            rock_fraction * (m.temp[roi] + rock_emission_factor * m.sw_in[roi])**4
            + (1 - rock_fraction) * model.state.surface.temp[roi]**4
        )
    )

    # Total incoming longwave radiation
    m.lw_in[roi] = lw_in_clearsky + lw_in_clouds + lw_in_slopes
