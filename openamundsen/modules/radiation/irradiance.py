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

    if model.sun_params['sun_over_horizon']:
        mean_surface_albedo = model.state.surface.albedo[roi].mean()
        if np.isnan(mean_surface_albedo):
            # E.g. in the first timestep no albedo has yet been calculated; assume a default value here
            mean_surface_albedo = model.config.soil.albedo

        model.logger.debug('Calculating shadows')

        if model.grid.extended_grid.available:
            shadows_dem = model.grid.extended_grid.dem
        else:
            shadows_dem = model.state.base.dem

        shadows = modules.radiation.shadows(
            shadows_dem,
            model.grid.resolution,
            model.sun_params['sun_vector'],
            num_sweeps=model.config.meteo.radiation.num_shadow_sweeps,
        )

        if model.grid.extended_grid.available:
            shadows = shadows[
                model.grid.extended_grid.row_slice,
                model.grid.extended_grid.col_slice,
            ]

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
            roi=roi,
            ozone_layer_thickness=model.config.meteo.radiation.ozone_layer_thickness,
            atmospheric_visibility=model.config.meteo.radiation.atmospheric_visibility,
            single_scattering_albedo=model.config.meteo.radiation.single_scattering_albedo,
            clear_sky_albedo=model.config.meteo.radiation.clear_sky_albedo,
        )
    else:
        dir_irr = np.zeros(model.grid.shape)
        diff_irr = np.zeros(model.grid.shape)

    pot_irr = dir_irr + diff_irr
    model.state.meteo.sw_in_clearsky[roi] = pot_irr[roi]
    model.state.meteo.dir_in_clearsky[roi] = dir_irr[roi]
    model.state.meteo.diff_in_clearsky[roi] = diff_irr[roi]


def shortwave_irradiance(model):
    model.logger.debug('Calculating actual shortwave irradiance')
    cloud_config = model.config.meteo.interpolation.cloudiness
    roi = model.grid.roi
    m = model.state.meteo
    method = cloud_config['method']

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

    if method == 'clear_sky_fraction':
        if not model.sun_params['sun_over_horizon']:
            method = cloud_config['clear_sky_fraction_night_method']
        elif num_rad_stations == 0:
            if cloud_config['allow_fallback'] and num_temp_hum_stations > 0:
                model.logger.debug(
                    'No radiation measurements available, falling back to humidity-based '
                    'cloudiness calculation'
                )
                method = 'humidity'
            else:
                model.logger.debug(
                    'No radiation measurements available, using cloudiness from previous time '
                    'step'
                )
                method = 'constant'

    interpolate_cloud_factor = False

    if method == 'constant':  # use cloudiness from previous time step
        cloud_factor_roi_prev = m.cloud_factor[roi]

        # When there is no cloudiness from the previous time step (e.g. when the model run starts
        # during nighttime) set cloudiness to a constant value
        model.logger.debug(
            'No cloudiness from previous time step available, setting to constant value'
        )
        cloud_factor_roi_prev[np.isnan(cloud_factor_roi_prev)] = 0.75

        m.cloud_factor[roi] = cloud_factor_roi_prev
    elif method == 'humidity':
        lr_t = model.config.meteo.interpolation.temperature.lapse_rate[model.date.month - 1]
        lr_td = model.config.meteo.interpolation.humidity.lapse_rate[model.date.month - 1]
        # TODO use here also the same settings for regression/fixed gradients as for the
        # interpolation
        cloud_fracs = meteo.cloud_fraction_from_humidity(
            ds_temp_hum.temp,
            ds_temp_hum.rel_hum,
            ds_temp_hum.alt,
            lr_t,
            lr_td,
            pressure_level=cloud_config.pressure_level * 100,  # hPa -> Pa
            saturation_cloud_fraction=cloud_config.saturation_cloud_fraction,
            e_folding_humidity=cloud_config.e_folding_humidity,
        )
        cloud_factor_xs = ds_temp_hum.x
        cloud_factor_ys = ds_temp_hum.y
        cloud_factors = meteo.cloud_factor_from_cloud_fraction(cloud_fracs)
        interpolate_cloud_factor = True
    elif method == 'clear_sky_fraction':
        cloud_factor_xs = ds_rad.x
        cloud_factor_ys = ds_rad.y
        cloud_factors = ds_rad.sw_in.values / m.sw_in_clearsky[ds_rad.row, ds_rad.col]
        interpolate_cloud_factor = True

    if interpolate_cloud_factor:
        cloud_factor_interpol = interpolation.idw(
            cloud_factor_xs,
            cloud_factor_ys,
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
