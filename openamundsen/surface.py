from numba import njit, prange
import numpy as np
from openamundsen import constants, meteo


def surface_layer_properties(model):
    """
    Update surface layer properties following [1].

    Parameters
    ----------
    model : Model
        Model instance.

    References
    ----------
    .. [1] Cox, P. M., Betts, R. A., Bunton, C. B., Essery, R. L. H., Rowntree,
       P. R., & Smith, J. (1999). The impact of new land surface physics on the
       GCM simulation of climate and climate sensitivity. Climate Dynamics, 15(3),
       183–203. https://doi.org/10.1007/s003820050276
    """
    s = model.state
    roi = model.grid.roi

    s.surface.thickness[roi] = np.maximum(s.snow.thickness[0, roi], s.soil.thickness[0, roi])
    s.surface.layer_temp[roi] = (
        s.soil.temp[0, roi]
        + (s.snow.temp[0, roi] - s.soil.temp[0, roi])
        * s.snow.thickness[0, roi] / s.soil.thickness[0, roi]
    )

    with np.errstate(invalid='ignore'):  # ignore "invalid value encountered in less_equal" errors because here for simplicity outside-ROI values are also compared
        pos1 = roi & (s.snow.thickness[0, :, :] <= s.soil.thickness[0, :, :] / 2)
        pos2 = roi & (s.snow.thickness[0, :, :] > s.soil.thickness[0, :, :])

    # Effective surface thermal conductivity (eq. (79))
    s.surface.therm_cond[roi] = s.snow.therm_cond[0, roi]
    s.surface.therm_cond[pos1] = (
        s.soil.thickness[0, pos1]
        / (
            2 * s.snow.thickness[0, pos1] / s.snow.therm_cond[0, pos1]
            + (s.soil.thickness[0, pos1] - 2 * s.snow.thickness[0, pos1]) / s.soil.therm_cond[0, pos1]
        )
    )

    s.surface.layer_temp[pos2] = s.snow.temp[0, pos2]


def _stability_factor(
    temp,
    surface_temp,
    wind_speed,
    snow_cover_fraction,
    snow_roughness_length,
    snow_free_roughness_length,
    temp_measurement_height,
    wind_measurement_height,
    stability_adjustment_parameter,
):
    """
    Calculate the atmospheric stability factor from [1] (eq. (25)).

    Parameters
    ----------
    temp : ndarray
        Air temperature (K).

    surface_temp : ndarray
        Surface temperature (K).

    wind_speed : ndarray
        Wind speed (m s-1).

    snow_cover_fraction : ndarray
        Snow cover fraction.

    snow_roughness_length : float
        Roughness length of snow-covered ground (m).

    snow_free_roughness_length : float
        Roughness length of snow-free ground (m).

    temp_measurement_height : float
        Temperature measurement height (m).

    wind_measurement_height : float
        Wind measurement height (m).

    stability_adjustment_parameter : float
        Atmospheric stability adjustment parameter (b_h from [1]).

    Returns
    -------
    stability_factor : ndarray
        Atmospheric stability factor.

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    # Bulk Richardson number (eq. (24))
    richardson = (
        constants.GRAVITATIONAL_ACCELERATION
        * wind_measurement_height**2
        * (temp - surface_temp)
        / (temp_measurement_height * temp * wind_speed**2)
    )

    # Surface momentum roughness length (eq. (23))
    momentum_roughness_length = (
        snow_roughness_length**snow_cover_fraction
        * snow_free_roughness_length**(1 - snow_cover_fraction)
    )

    # eq. (26)
    c = (
        3 * stability_adjustment_parameter**2 * constants.VON_KARMAN**2
        * np.sqrt(wind_measurement_height / momentum_roughness_length)
        / (np.log(wind_measurement_height / momentum_roughness_length))**2
    )

    # eq. (25)
    pos = richardson >= 0
    stability_factor = np.empty(pos.shape)
    stability_factor[pos] = 1 / (1 + 3 * stability_adjustment_parameter * richardson[pos] * np.sqrt(
        1 + stability_adjustment_parameter * richardson[pos]))
    stability_factor[~pos] = 1 / (1 - 3 * stability_adjustment_parameter * richardson[~pos] * (
        1 + c[~pos] * np.sqrt(-richardson[~pos])))

    return stability_factor


def _heat_moisture_transfer_coefficient(
    surface_roughness_length,
    temp_measurement_height,
    wind_measurement_height,
):
    """
    Calculate the transfer coefficient C_H from [1] (eq. (22)), without the
    possible adjustment for atmospheric stability.

    Parameters
    ----------
    surface_roughness_length : ndarray
        Surface roughness length (m).

    temp_measurement_height : float
        Temperature/humidity measurement height.

    wind_measurement_height : float
        Wind measurement height.

    Returns
    -------
    coeff : ndarray
        Transfer coefficient.

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    heat_moisture_roughness_length = 0.1 * surface_roughness_length

    coeff = constants.VON_KARMAN**2 / (
        np.log(wind_measurement_height / surface_roughness_length)
        * np.log(temp_measurement_height / heat_moisture_roughness_length)
    )

    return coeff


def energy_balance(model):
    """
    Calculate the surface energy balance following [1].

    Parameters
    ----------
    model : Model
        Model instance.

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
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

    temp_measurement_height = model.config.meteo.measurement_height.temperature
    wind_measurement_height = model.config.meteo.measurement_height.wind

    if model.config.snow.measurement_height_adjustment:
        temp_measurement_height -= s.snow.depth[roi]
        temp_measurement_height = np.maximum(temp_measurement_height, 1.)  # as implemented in FSM

    heat_moisture_transfer_coeff = _heat_moisture_transfer_coefficient(
        s.surface.roughness_length[roi],
        temp_measurement_height,
        wind_measurement_height,
    )

    if model.config.meteo.stability_correction:
        snow_cover_frac = np.full(roi.sum(), 1.)  # TODO implement this
        stability_factor = _stability_factor(
            s.meteo.temp[roi],
            s.surface.temp[roi],
            s.meteo.wind_speed[roi],
            snow_cover_frac,
            model.config.snow.roughness_length,
            model.config.soil.roughness_length,
            temp_measurement_height,
            wind_measurement_height,
            model.config.meteo.stability_adjustment_parameter,
        )
        heat_moisture_transfer_coeff *= stability_factor

    moisture_availability = surf_moisture_conductance / (
        surf_moisture_conductance
        + heat_moisture_transfer_coeff * s.meteo.wind_speed[roi]
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
    rhoa_CH_Ua = air_density * heat_moisture_transfer_coeff * s.meteo.wind_speed[roi]

    # Surface energy balance without melt
    s.snow.melt[roi] = 0
    D = latent_heat * sat_spec_hum_surf / (  # D = dQsat/dTs (eq. (37))
        constants.SPEC_GAS_CONSTANT_WATER_VAPOR * s.surface.temp[roi]**2
    )
    surf_moisture_flux = moisture_availability * rhoa_CH_Ua * (sat_spec_hum_surf - s.meteo.spec_hum[roi])
    s.surface.heat_flux[roi] = (
        2 * s.surface.therm_cond[roi]
        * (s.surface.temp[roi] - s.surface.layer_temp[roi])
        / s.surface.thickness[roi]
    )
    sens_heat_flux = constants.SPEC_HEAT_CAP_DRY_AIR * rhoa_CH_Ua * (s.surface.temp[roi] - s.meteo.temp[roi])
    lat_heat_flux = latent_heat * surf_moisture_flux
    net_radiation = (
        (1 - s.surface.albedo[roi]) * s.meteo.sw_in[roi]
        + s.meteo.lw_in[roi]
        - constants.STEFAN_BOLTZMANN * s.surface.temp[roi]**4
    )
    surf_temp_change = (
        (net_radiation - sens_heat_flux - lat_heat_flux - s.surface.heat_flux[roi])
        / (
            (constants.SPEC_HEAT_CAP_DRY_AIR + latent_heat * moisture_availability * D) * rhoa_CH_Ua
            + 2 * s.surface.therm_cond[roi] / s.surface.thickness[roi]
            + 4 * constants.STEFAN_BOLTZMANN * s.surface.temp[roi]**3
        )
    )
    surf_moisture_flux_change = moisture_availability * rhoa_CH_Ua * D * surf_temp_change
    surf_heat_flux_change = 2 * s.surface.therm_cond[roi] * surf_temp_change / s.surface.thickness[roi]
    sens_heat_flux_change = constants.SPEC_HEAT_CAP_DRY_AIR * rhoa_CH_Ua * surf_temp_change

    # TODO surface melting

    # Update surface temperature and fluxes
    s.surface.temp[roi] += surf_temp_change
    surf_moisture_flux += surf_moisture_flux_change
    s.surface.heat_flux[roi] += surf_heat_flux_change
    sens_heat_flux += sens_heat_flux_change

    # TODO snow sublimation/soil evaporation