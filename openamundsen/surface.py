from numba import njit, prange
import numpy as np
from openamundsen import constants, meteo


@njit(parallel=True, cache=True)
def surface_layer_properties(
    roi_idxs,
    surf_layer_temp,
    surf_thickness,
    surf_therm_cond,
    snow_temp,
    snow_thickness,
    snow_therm_cond,
    soil_temp,
    soil_thickness,
    soil_therm_cond,
):
    num_pixels = len(roi_idxs)
    for idx_num in prange(num_pixels):
        i, j = roi_idxs[idx_num]

        surf_thickness[i, j] = max(snow_thickness[0, i, j], soil_thickness[0, i, j])
        surf_layer_temp[i, j] = (
            soil_temp[0, i, j]
            + (snow_temp[0, i, j] - soil_temp[0, i, j]) * snow_thickness[0, i, j] / soil_thickness[0, i, j]
        )

        if snow_thickness[0, i, j] <= soil_thickness[0, i, j] / 2:
            surf_therm_cond[i, j] = (
                soil_thickness[0, i, j]
                / (
                    2 * snow_thickness[0, i, j] / snow_therm_cond[0, i, j]
                    + (soil_thickness[0, i, j] - 2 * snow_thickness[0, i, j]) / soil_therm_cond[0, i, j]
                )
            )
        else:
            surf_therm_cond[i, j] = snow_therm_cond[0, i, j]

            if snow_thickness[0, i, j] > soil_thickness[0, i, j]:
                surf_layer_temp[i, j] = snow_temp[0, i, j]


def surface_exchange(model):
    s = model.state
    roi = model.grid.roi

    # TODO implement measurement height adjustment
    adjusted_measurement_height = model.config.meteo.measurement_height.temperature

    # Neutral exchange coefficients
    heat_moisture_roughness_length = 0.1 * s.surface.roughness_length[roi]
    s.surface.heat_moisture_transfer_coeff[roi] = (
        constants.VON_KARMAN**2
        / (
            np.log(model.config.meteo.measurement_height.wind / s.surface.roughness_length[roi])
            * np.log(adjusted_measurement_height / heat_moisture_roughness_length)
        )
    )

    # TODO stability correction


def energy_balance(model):
    s = model.state
    roi = model.grid.roi

    surf_moisture_conductance = model.config.soil.saturated_soil_surface_conductance * (
        np.maximum(
            (
                s.soil.frac_unfrozen_moisture_content[0, roi]
                * s.soil.vol_moisture_content_sat[roi] / s.soil.vol_moisture_content_crit[roi]
            )**2,
            1,
        )
    )

    moisture_availability = surf_moisture_conductance / (
        surf_moisture_conductance
        + s.surface.heat_moisture_transfer_coeff[roi] * s.meteo.wind_speed[roi]
    )

    sat_vap_press_surf = meteo.saturation_vapor_pressure(s.surface.temp[roi])
    sat_spec_hum_surf = meteo.specific_humidity(  # saturation specific humidity at surface temperature
        s.meteo.atmos_press[roi],
        sat_vap_press_surf,
    )

    moisture_availability[sat_spec_hum_surf < s.meteo.spec_hum[roi]] = 1.
    moisture_availability[s.snow.ice_content[0, roi] > 0] = 1.

    latent_heat = np.where(
        s.surface.temp[roi] > constants.T0,
        constants.LATENT_HEAT_OF_VAPORIZATION,
        constants.LATENT_HEAT_OF_SUBLIMATION,
    )

    air_density = s.meteo.atmos_press[roi] / (constants.GAS_CONSTANT_DRY_AIR * s.meteo.temp[roi])
    # TODO rename rKH to meaningful name
    rKH = air_density * s.surface.heat_moisture_transfer_coeff[roi] * s.meteo.wind_speed[roi]

    # Surface energy balance without melt
    # TODO rename D to meaningful name
    D = latent_heat * sat_spec_hum_surf / (constants.SPEC_GAS_CONSTANT_WATER_VAPOR * s.surface.temp[roi]**2)
    surf_moisture_flux = moisture_availability * rKH * (sat_spec_hum_surf - s.meteo.spec_hum[roi])
    s.surface.heat_flux[roi] = (
        2 * s.surface.therm_cond[roi]
        * (s.surface.temp[roi] - s.surface.layer_temp[roi])
        / s.surface.thickness[roi]
    )
    sens_heat_flux = constants.SPEC_HEAT_CAP_DRY_AIR * rKH * (s.surface.temp[roi] - s.meteo.temp[roi])
    lat_heat_flux = latent_heat * surf_moisture_flux
    s.snow.melt[roi] = 0
    net_radiation = (
        (1 - s.surface.albedo[roi]) * s.meteo.sw_in[roi]
        + s.meteo.lw_in[roi]
        - constants.STEFAN_BOLTZMANN * s.surface.temp[roi]**4
    )
    surf_temp_change = (
        (net_radiation - sens_heat_flux - lat_heat_flux - s.surface.heat_flux[roi])
        / (
            (constants.SPEC_HEAT_CAP_DRY_AIR + latent_heat * moisture_availability * D) * rKH
            + 2 * s.surface.therm_cond[roi] / s.surface.thickness[roi]
            + 4 * constants.STEFAN_BOLTZMANN * s.surface.temp[roi]**3
        )
    )
    surf_moisture_flux_change = moisture_availability * rKH * D * surf_temp_change
    surf_heat_flux_change = 2 * s.surface.therm_cond[roi] * surf_temp_change / s.surface.thickness[roi]
    sens_heat_flux_change = constants.SPEC_HEAT_CAP_DRY_AIR * rKH * surf_temp_change

    # TODO surface melting

    # Update surface temperature and fluxes
    s.surface.temp[roi] += surf_temp_change
    surf_moisture_flux += surf_moisture_flux_change
    s.surface.heat_flux[roi] += surf_heat_flux_change
    sens_heat_flux += sens_heat_flux_change

    # TODO snow sublimation/soil evaporation
