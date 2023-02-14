import numpy as np
from .clearsky import _clear_sky_shortwave_irradiance
from openamundsen import (
    constants,
    interpolation,
    meteo,
    modules,
)
import xarray as xr


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

        # Downsample DEM before calculating shadows if requested
        shadows_downsampling_factor = model.config.meteo.radiation.shadows_downsampling_factor
        if shadows_downsampling_factor > 1:
            orig_shadows_shape = shadows_dem.shape
            shadows_dem = shadows_dem[::shadows_downsampling_factor, ::shadows_downsampling_factor]

        shadows = modules.radiation.shadows(
            shadows_dem,
            model.grid.resolution * shadows_downsampling_factor,
            model.sun_params['sun_vector'],
            num_sweeps=model.config.meteo.radiation.num_shadow_sweeps,
        )

        # Upsample shadows array if necessary (by simple repetition of array elements)
        if shadows_downsampling_factor > 1:
            shadows = (
                shadows
                .repeat(shadows_downsampling_factor, axis=0)
                .repeat(shadows_downsampling_factor, axis=1)
            )
            if shadows.shape[0] > orig_shadows_shape[0]:
                shadows = shadows[:orig_shadows_shape[0], :]
            if shadows.shape[1] > orig_shadows_shape[1]:
                shadows = shadows[:, :orig_shadows_shape[1]]

        if model.grid.extended_grid.available:
            model.grid.extended_grid.shadows = shadows
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


def extended_grid_stations_clear_sky_shortwave_irradiance(model):
    model.logger.debug('Calculating clear-sky shortwave irradiance for extended-grid stations')
    roi = model.grid.roi
    ext_grid = model.grid.extended_grid

    ext_row_offset = ext_grid.row_offset
    ext_col_offset = ext_grid.col_offset
    station_rows_ext = model.meteo.row + ext_row_offset
    station_cols_ext = model.meteo.col + ext_col_offset
    # TODO this is the same in every time step, maybe add a "within_extended_grid_extent"
    # variable to model.meteo
    stations_within_ext_grid = (
        (station_rows_ext >= 0)
        & (station_cols_ext >= 0)
        & (station_rows_ext < ext_grid.rows)
        & (station_cols_ext < ext_grid.cols)
    )

    meteo_ds = (
        model.meteo
        .sel(station=stations_within_ext_grid & ~model.meteo.within_grid_extent)
        .sel(time=model.date)
    )
    # Consider only stations with temperature and humidity measurements (required for calculating
    # precipitable water)
    meteo_ds = meteo_ds.dropna(dim='station', how='any', subset=['temp', 'rel_hum'])
    clear_sky_rad = xr.DataArray(
        np.full(meteo_ds.dims['station'], np.nan),
        coords=meteo_ds.coords,
        dims=meteo_ds.dims,
    )

    if meteo_ds.dims['station'] > 0:
        extgrid_rows = (meteo_ds.row + ext_grid.row_offset).values
        extgrid_cols = (meteo_ds.col + ext_grid.col_offset).values

        # This is also calculated twice (here and in clear_sky_shortwave_irradiance() :/)
        mean_surface_albedo = model.state.surface.albedo[roi].mean()
        if np.isnan(mean_surface_albedo):
            mean_surface_albedo = model.config.soil.albedo

        extgrid_atmos_press = meteo.atmospheric_pressure(
            ext_grid.dem[extgrid_rows, extgrid_cols],
        )
        extgrid_precipitable_water = meteo.precipitable_water(
            meteo_ds.temp,
            meteo.vapor_pressure(meteo_ds.temp, meteo_ds.rel_hum),
        ).values
        # (-> this works only for stations with temperature and humidity measurements)

        extgrid_dir_irr, extgrid_diff_irr = _clear_sky_shortwave_irradiance(
            model.sun_params['day_angle'],
            model.sun_params['sun_vector'],
            ext_grid.shadows[extgrid_rows, extgrid_cols],
            ext_grid.dem[extgrid_rows, extgrid_cols],
            ext_grid.svf[extgrid_rows, extgrid_cols],
            ext_grid.normal_vec[:, extgrid_rows, extgrid_cols],
            extgrid_atmos_press,
            extgrid_precipitable_water,
            mean_surface_albedo,
            ozone_layer_thickness=model.config.meteo.radiation.ozone_layer_thickness,
            atmospheric_visibility=model.config.meteo.radiation.atmospheric_visibility,
            single_scattering_albedo=model.config.meteo.radiation.single_scattering_albedo,
            clear_sky_albedo=model.config.meteo.radiation.clear_sky_albedo,
        )
        clear_sky_rad[:] = extgrid_dir_irr + extgrid_diff_irr

    return clear_sky_rad


def shortwave_irradiance(model):
    model.logger.debug('Calculating actual shortwave irradiance')
    cloud_config = model.config.meteo.interpolation.cloudiness
    roi = model.grid.roi
    m = model.state.meteo
    ext_grid = model.grid.extended_grid
    method = cloud_config['method']

    meteo_ds = model.meteo.sel(time=model.date)

    if method == 'clear_sky_fraction':
        # Select stations within the grid extent and with shortwave radiation measurements for the
        # current time step
        ds_rad = (
            meteo_ds
            .sel(station=model.meteo.within_grid_extent)
            .dropna('station', subset=['sw_in'])
        )
        num_rad_stations = len(ds_rad.station)

    # Select stations with temperature and humidity measurements for the current time step
    ds_temp_hum = meteo_ds.dropna('station', how='any', subset=['temp', 'rel_hum'])
    num_temp_hum_stations = len(ds_temp_hum.station)

    if method == 'clear_sky_fraction':
        if model.sun_params['sun_over_horizon']:
            if ext_grid.available:
                extgrid_sw_in_clearsky = extended_grid_stations_clear_sky_shortwave_irradiance(
                    model
                )
                rad_ds_extgrid = (
                    meteo_ds
                    .sel(station=extgrid_sw_in_clearsky.station)
                    .dropna('station', subset=['sw_in'])
                )
                num_rad_stations += rad_ds_extgrid.dims['station']
        else:
            method = cloud_config['clear_sky_fraction_night_method']

    if cloud_config['allow_fallback']:
        if method == 'clear_sky_fraction' and num_rad_stations == 0:
            model.logger.debug(
                'No radiation measurements available, trying humidity-based cloudiness calculation'
            )
            method = 'humidity'

        if method == 'humidity' and num_temp_hum_stations == 0:
            model.logger.debug(
                'No temperature/humidity measurements available, using cloudiness from previous '
                'time step'
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
        cloud_factors = (
            ds_rad.sw_in.values / m.sw_in_clearsky[ds_rad.row, ds_rad.col]
        )
        interpolate_cloud_factor = True

        if ext_grid.available:
            extgrid_cloud_factors = rad_ds_extgrid.sw_in / extgrid_sw_in_clearsky
            cloud_factor_xs = np.append(cloud_factor_xs, rad_ds_extgrid.x)
            cloud_factor_ys = np.append(cloud_factor_ys, rad_ds_extgrid.y)
            cloud_factors = np.append(cloud_factors, extgrid_cloud_factors)
    elif method == 'prescribed':
        cloudiness_ds = meteo_ds.dropna('station', subset=['cloud_cover'])
        cloud_factor_xs = cloudiness_ds.x
        cloud_factor_ys = cloudiness_ds.y
        cloud_factors = cloudiness_ds.cloud_cover / 100  # convert from % to 0-1
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
