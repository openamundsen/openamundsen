from numba import njit, prange
import numpy as np
from openamundsen import constants, meteo
from openamundsen.modules.snow import CryoLayerID


def surface_properties(model):
    """
    Update surface properties following [1].

    Parameters
    ----------
    model : Model
        Model instance.

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015

    .. [2] Essery, R. L. H., Best, M. J., & Cox, P. M. (2001). MOSES 2.2
       Technical Documentation, Tech. rep., Hadley Centre, Met Office.
       http://jules.jchmr.org/sites/default/files/HCTN_30.pdf
    """
    s = model.state
    roi = model.grid.roi

    s.surface.albedo[roi] = np.where(
        s.snow.area_fraction[roi] > 0,
        s.snow.area_fraction[roi] * s.snow.albedo[roi] + (1 - s.snow.area_fraction[roi]) * model.config.soil.albedo,
        model.config.soil.albedo,
    )

    if model.config.snow.model == 'layers':
        s.surface.roughness_length[roi] = (
            model.config.snow.roughness_length**s.snow.area_fraction[roi]
            * model.config.soil.roughness_length**(1 - s.snow.area_fraction[roi])
        )
        calculate_heat_moisture_transfer_coefficient(model)

        # Calculate surface conductance (eq. (35) from [2])
        # (the constant 1/100 therein corresponds to
        # model.config.soil.saturated_soil_surface_conductance; the clipping of (theta_1/theta_c)**2 to
        # 1 is taken from FSM)
        s.surface.conductance[roi] = model.config.soil.saturated_soil_surface_conductance * (  # (m s-1)
            np.maximum(
                (
                    s.soil.frac_unfrozen_moisture_content[0, roi]
                    * s.soil.vol_moisture_content_sat[roi] / s.soil.vol_moisture_content_crit[roi]
                )**2,
                1,
            )
        )
    elif model.config.snow.model == 'cryolayers':
        s.surface.layer_type[roi] = CryoLayerID.SNOW_FREE

        for i in reversed(range(model.snow.num_layers)):
            pos = model.roi_mask_to_global(s.snow.ice_content[i, roi] > 0)
            s.surface.layer_type[pos] = i


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

    pos1 = model.roi_mask_to_global(s.snow.thickness[0, roi] <= s.soil.thickness[0, roi] / 2)
    pos2 = model.roi_mask_to_global(s.snow.thickness[0, roi] > s.soil.thickness[0, roi])

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


def calculate_heat_moisture_transfer_coefficient(model):
    """
    Calculate the transfer coefficient C_H from [1] (eq. (22)) including the
    possible adjustment for atmospheric stability.

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

    temp_measurement_height = model.config.meteo.measurement_height.temperature
    wind_measurement_height = model.config.meteo.measurement_height.wind

    if model.config.snow.measurement_height_adjustment:
        temp_measurement_height -= s.snow.depth[roi]
        temp_measurement_height = np.maximum(temp_measurement_height, 1.)  # as implemented in FSM

    heat_moisture_roughness_length = 0.1 * s.surface.roughness_length[roi]
    coeff = constants.VON_KARMAN**2 / (
        np.log(wind_measurement_height / s.surface.roughness_length[roi])
        * np.log(temp_measurement_height / heat_moisture_roughness_length)
    )

    if model.config.meteo.stability_correction:
        stability_factor = _stability_factor(
            s.meteo.temp[roi],
            s.surface.temp[roi],
            s.meteo.wind_speed[roi],
            s.snow.area_fraction[roi],
            model.config.snow.roughness_length,
            model.config.soil.roughness_length,
            temp_measurement_height,
            wind_measurement_height,
            model.config.meteo.stability_adjustment_parameter,
        )
        coeff *= stability_factor

    s.surface.heat_moisture_transfer_coeff[roi] = coeff


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

    .. [2] Essery, R. L. H., Best, M. J., & Cox, P. M. (2001). MOSES 2.2
       Technical Documentation, Tech. rep., Hadley Centre, Met Office.
       http://jules.jchmr.org/sites/default/files/HCTN_30.pdf
    """
    s = model.state
    roi = model.grid.roi

    sat_vap_press_surf = meteo.saturation_vapor_pressure(s.surface.temp[roi])
    sat_spec_hum_surf = meteo.specific_humidity(  # saturation specific humidity at surface temperature
        s.meteo.atmos_press[roi],
        sat_vap_press_surf,
    )

    moisture_availability = s.surface.conductance[roi] / (  # part of eq. (38) from [2]
        s.surface.conductance[roi]
        + s.surface.heat_moisture_transfer_coeff[roi] * s.meteo.wind_speed[roi]
    )
    moisture_availability[sat_spec_hum_surf < s.meteo.spec_hum[roi]] = 1.
    moisture_availability[s.snow.ice_content[0, roi] > 0] = 1.

    latent_heat = np.where(  # (J kg-1)
        s.surface.temp[roi] > constants.T0,
        constants.LATENT_HEAT_OF_VAPORIZATION,
        constants.LATENT_HEAT_OF_SUBLIMATION,
    )

    rhoa_CH_Ua = s.meteo.dry_air_density[roi] * s.surface.heat_moisture_transfer_coeff[roi] * s.meteo.wind_speed[roi]  # (kg m-2 s-1)

    # Calculate surface energy balance without melt
    s.snow.melt[roi] = 0
    dQsat_by_dTs = latent_heat * sat_spec_hum_surf / (  # eq. (37) (K-1)
        constants.SPEC_GAS_CONSTANT_WATER_VAPOR * s.surface.temp[roi]**2
    )
    surf_moisture_flux = moisture_availability * rhoa_CH_Ua * (sat_spec_hum_surf - s.meteo.spec_hum[roi])  # (kg m-2 s-1)
    s.surface.heat_flux[roi] = (
        2 * s.surface.therm_cond[roi]
        * (s.surface.temp[roi] - s.surface.layer_temp[roi])
        / s.surface.thickness[roi]
    )
    s.surface.sens_heat_flux[roi] = constants.SPEC_HEAT_CAP_DRY_AIR * rhoa_CH_Ua * (s.surface.temp[roi] - s.meteo.temp[roi])
    s.surface.lat_heat_flux[roi] = latent_heat * surf_moisture_flux
    radiation_balance(model)
    surf_temp_change = (  # eq. (38) (K)
        (s.meteo.net_radiation[roi] - s.surface.sens_heat_flux[roi] - s.surface.lat_heat_flux[roi] - s.surface.heat_flux[roi])
        / (
            (constants.SPEC_HEAT_CAP_DRY_AIR + latent_heat * moisture_availability * dQsat_by_dTs) * rhoa_CH_Ua
            + 2 * s.surface.therm_cond[roi] / s.surface.thickness[roi]
            + 4 * constants.STEFAN_BOLTZMANN * s.surface.temp[roi]**3
        )
    )
    surf_moisture_flux_change = moisture_availability * rhoa_CH_Ua * dQsat_by_dTs * surf_temp_change  # eq. (33) (kg m-2 s-1)
    surf_heat_flux_change = 2 * s.surface.therm_cond[roi] * surf_temp_change / s.surface.thickness[roi]  # eq. (34) (W m-2)
    sens_heat_flux_change = constants.SPEC_HEAT_CAP_DRY_AIR * rhoa_CH_Ua * surf_temp_change  # eq. (35) (W m-2)

    # Calculate melt
    melties_roi = (
        ((s.surface.temp[roi] + surf_temp_change) > constants.T0)
        & (s.snow.ice_content[0, roi] > 0)
    )
    if melties_roi.any():
        melties = model.roi_mask_to_global(melties_roi)
        s.snow.melt[melties] = s.snow.ice_content[:, melties].sum(axis=0) / model.timestep
        surf_temp_change[melties_roi] = (
            (
                s.meteo.net_radiation[melties]
                - s.surface.sens_heat_flux[melties]
                - s.surface.lat_heat_flux[melties]
                - s.surface.heat_flux[melties]
                - constants.LATENT_HEAT_OF_FUSION * s.snow.melt[melties]
            ) / (
                (
                    constants.SPEC_HEAT_CAP_DRY_AIR
                    + latent_heat[melties_roi] * moisture_availability[melties_roi] * dQsat_by_dTs[melties_roi]
                ) * rhoa_CH_Ua[melties_roi]
                + 2 * s.surface.therm_cond[melties] / s.surface.thickness[melties]
                + 4 * constants.STEFAN_BOLTZMANN * s.surface.temp[melties]**3
            )
        )

        surf_moisture_flux_change[melties_roi] = (
            rhoa_CH_Ua[melties_roi]
            * dQsat_by_dTs[melties_roi]
            * surf_temp_change[melties_roi]
        )

        surf_heat_flux_change[melties_roi] = (
            2 * s.surface.therm_cond[melties]
            * surf_temp_change[melties_roi]
            / s.surface.thickness[melties]
        )

        sens_heat_flux_change[melties_roi] = (
            constants.SPEC_HEAT_CAP_DRY_AIR
            * rhoa_CH_Ua[melties_roi]
            * surf_temp_change[melties_roi]
        )

        melties2_roi = melties_roi & (s.surface.temp[roi] + surf_temp_change < constants.T0)
        if melties2_roi.any():
            melties2 = model.roi_mask_to_global(melties2_roi)

            sat_vap_press_surf[melties2_roi] = meteo.saturation_vapor_pressure(constants.T0)
            sat_spec_hum_surf[melties2_roi] = meteo.specific_humidity(
                s.meteo.atmos_press[melties2],
                sat_vap_press_surf[melties2_roi],
            )

            surf_moisture_flux[melties2_roi] = (
                moisture_availability[melties2_roi]
                * rhoa_CH_Ua[melties2_roi]
                * (sat_spec_hum_surf[melties2_roi] - s.meteo.spec_hum[melties2])
            )

            s.surface.heat_flux[melties2] = (
                2 * s.surface.therm_cond[melties2]
                * (constants.T0 - s.surface.layer_temp[melties2])
                / s.surface.thickness[melties2]
            )

            s.surface.sens_heat_flux[melties2] = constants.SPEC_HEAT_CAP_DRY_AIR * rhoa_CH_Ua[melties2_roi] * (constants.T0 - s.meteo.temp[melties2])
            s.surface.lat_heat_flux[melties2] = constants.LATENT_HEAT_OF_SUBLIMATION * surf_moisture_flux[melties2_roi]

            s.surface.temp[melties2] = constants.T0
            radiation_balance(model, melties2)  # update net radiation

            s.snow.melt[melties2] = np.maximum(
                (
                    (
                        s.meteo.net_radiation[melties2]
                        - s.surface.sens_heat_flux[melties2]
                        - s.surface.lat_heat_flux[melties2]
                        - s.surface.heat_flux[melties2]
                    ) / constants.LATENT_HEAT_OF_FUSION
                ),
                0,
            )

            surf_moisture_flux_change[melties2_roi] = 0.
            surf_heat_flux_change[melties2_roi] = 0.
            sens_heat_flux_change[melties2_roi] = 0.
            surf_temp_change[melties2_roi] = 0.  # surface temperature has already been set to 0 °C above

    # Update surface temperature and fluxes
    s.surface.temp[roi] += surf_temp_change
    surf_moisture_flux += surf_moisture_flux_change
    s.surface.heat_flux[roi] += surf_heat_flux_change
    s.surface.sens_heat_flux[roi] += sens_heat_flux_change

    # Snow sublimation
    s.snow.sublimation[roi] = 0.
    pos = (s.snow.ice_content[0, roi] > 0) | (s.surface.temp[roi] < constants.T0)
    s.snow.sublimation[model.roi_mask_to_global(pos)] = surf_moisture_flux[pos]
    latent_heat = np.where(
        pos,
        constants.LATENT_HEAT_OF_SUBLIMATION,
        constants.LATENT_HEAT_OF_VAPORIZATION,
    )
    s.surface.lat_heat_flux[roi] = latent_heat * surf_moisture_flux
    # soil_evaporation[model.roi_mask_to_global(~pos)] = surf_moisture_flux[~pos]


def radiation_balance(model, pos=None):
    """
    Calculate outgoing shortwave and longwave radiation and net radiation.

    Parameters
    ----------
    model : Model
        Model instance.

    pos : ndarray(bool), default None
        Pixels for which the radiation fluxes should be calculated. If None,
        calculation is performed for the entire ROI.
    """
    s = model.state

    if pos is None:
        pos = model.grid.roi

    # TODO this is a parameter
    snow_emissivity = 0.99

    s.meteo.sw_out[pos] = s.surface.albedo[pos] * s.meteo.sw_in[pos]
    s.meteo.lw_out[pos] = snow_emissivity * constants.STEFAN_BOLTZMANN * s.surface.temp[pos]**4
    s.meteo.net_radiation[pos] = (
        s.meteo.sw_in[pos]
        - s.meteo.sw_out[pos]
        + s.meteo.lw_in[pos]
        - s.meteo.lw_out[pos]
    )
