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

    s.surface.roughness_length[roi] = (
        model.config.snow.roughness_length**s.snow.area_fraction[roi]
        * model.config.soil.roughness_length**(1 - s.snow.area_fraction[roi])
    )
    calc_turbulent_exchange_coefficient(model)

    if model.config.snow.model == 'layers':
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
            pos = model.roi_mask_to_global(s.snow.thickness[i, roi] > 0)
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

    calc_saturation_specific_humidity(model, roi)
    calc_moisture_availability(model, roi)
    calc_latent_heat(model, roi)

    # Calculate surface energy balance without melt
    s.snow.melt[roi] = 0
    calc_fluxes(model, roi)
    calc_radiation_balance(model)

    (
        surf_temp_change,
        surf_moisture_flux_change,
        surf_heat_flux_change,
        sens_heat_flux_change,
    ) = solve_energy_balance(model, roi)

    # Calculate melt
    melties_roi = (
        ((s.surface.temp[roi] + surf_temp_change) > constants.T0)
        & (s.snow.ice_content[0, roi] > 0)
    )
    if melties_roi.any():
        melties = model.roi_mask_to_global(melties_roi)
        s.snow.melt[melties] = s.snow.ice_content[:, melties].sum(axis=0) / model.timestep

        (
            surf_temp_change[melties_roi],
            surf_moisture_flux_change[melties_roi],
            surf_heat_flux_change[melties_roi],
            sens_heat_flux_change[melties_roi],
        ) = solve_energy_balance(model, melties)

        melties2_roi = melties_roi & (s.surface.temp[roi] + surf_temp_change < constants.T0)
        if melties2_roi.any():
            melties2 = model.roi_mask_to_global(melties2_roi)

            s.surface.temp[melties2] = constants.T0

            calc_latent_heat(model, melties2)
            calc_saturation_specific_humidity(model, melties2)
            calc_fluxes(model, melties2)
            calc_radiation_balance(model, melties2)  # update net radiation

            s.snow.melt[melties2] = (
                (
                    s.meteo.net_radiation[melties2]
                    - s.surface.sens_heat_flux[melties2]
                    - s.surface.lat_heat_flux[melties2]
                    - s.surface.heat_flux[melties2]
                ) / constants.LATENT_HEAT_OF_FUSION
            ).clip(min=0)

            surf_temp_change[melties2_roi] = 0.
            surf_moisture_flux_change[melties2_roi] = 0.
            surf_heat_flux_change[melties2_roi] = 0.
            sens_heat_flux_change[melties2_roi] = 0.

    # Update surface temperature and fluxes
    s.surface.temp[roi] += surf_temp_change
    s.surface.moisture_flux[roi] += surf_moisture_flux_change
    s.surface.heat_flux[roi] += surf_heat_flux_change
    s.surface.sens_heat_flux[roi] += sens_heat_flux_change
    calc_latent_heat(model, roi)
    calc_fluxes(model, roi, surface=False, moisture=False, sensible=False, latent=True)

    # Snow sublimation
    s.snow.sublimation[roi] = 0.
    pos_roi = (s.snow.ice_content[0, roi] > 0) | (s.surface.temp[roi] < constants.T0)
    pos = model.roi_mask_to_global(pos_roi)
    s.snow.sublimation[pos] = s.surface.moisture_flux[pos]
    # soil_evaporation[model.roi_mask_to_global(~pos)] = surf_moisture_flux[~pos]


def cryo_layer_energy_balance(model):
    s = model.state
    roi = model.grid.roi

    s.surface.temp[roi] = s.meteo.temp[roi]

    snowies_roi = s.snow.swe[roi] > 0.
    melties_roi = snowies_roi & (s.meteo.temp[roi] >= constants.T0)
    frosties_roi = snowies_roi & (s.meteo.temp[roi] < constants.T0)
    snowies = model.roi_mask_to_global(snowies_roi)
    melties = model.roi_mask_to_global(melties_roi)
    frosties = model.roi_mask_to_global(frosties_roi)

    calc_saturation_specific_humidity(model, snowies)
    calc_moisture_availability(model, snowies)
    calc_latent_heat(model, snowies)
    calc_radiation_balance(model, roi)

    s.snow.melt[roi] = 0

    # Where air temperature >= 0 °C -> potential melt, no iteration
    calc_fluxes(model, melties, surface=False, moisture=True, sensible=True, latent=True)
    available_melt_time = np.zeros(roi.shape)
    en_bal = np.zeros(roi.shape)
    available_melt_time[melties] = model.timestep  # contains the time (in seconds) available for melt in this time step for each pixel
    s.surface.temp[melties] = constants.T0

    for layer_num in range(model.snow.num_layers):
        possible_melties = model.roi_mask_to_global(
            melties_roi
            & (s.snow.thickness[layer_num, roi] > 0)
            & (available_melt_time[roi] > 0)
        )

        # TODO update radiation balance with possibly updated albedo?
        # radiation_balance(model, possible_melties)

        advect_heat_flux = 0.  # XXX
        surf_heat_flux = -2.  # XXX

        en_bal[possible_melties] = (
            s.meteo.net_radiation[possible_melties]
            - s.surface.sens_heat_flux[possible_melties]
            - s.surface.lat_heat_flux[possible_melties]
            - advect_heat_flux
            - surf_heat_flux
        )

        layer_melties = model.roi_mask_to_global(possible_melties[roi] & (en_bal[roi] > 0.))

        layer_we = (
            s.snow.ice_content[layer_num, :]
            # + s.snow.liquid_water_content[layer_num, :]
            + s.snow.cold_content[layer_num, :]
        )

        layer_melt_time = np.zeros(roi.shape)
        layer_melt_time[layer_melties] = (  # time needed to melt the entire layer
            constants.LATENT_HEAT_OF_FUSION
            * layer_we[layer_melties]
            / en_bal[layer_melties]
        )
        total_melties = model.roi_mask_to_global(
            layer_melties[roi]
            & (layer_melt_time[roi] < available_melt_time[roi])
        )
        partial_melties = model.roi_mask_to_global(
            layer_melties[roi]
            & (layer_melt_time[roi] >= available_melt_time[roi])
        )

        # Process pixels where the entire layer melts (and energy is left to melt the lower layers)
        s.snow.melt[total_melties] += s.snow.ice_content[layer_num, total_melties]
        s.snow.cold_content[layer_num, total_melties] = 0.

        # Process pixels where the available energy cannot melt the entire layer
        layer_melt = en_bal[partial_melties] * available_melt_time[partial_melties] / constants.LATENT_HEAT_OF_FUSION  # actual melt and cold content reduction
        s.snow.cold_content[layer_num, partial_melties] -= layer_melt
        actual_layer_melt = -1 * np.minimum(s.snow.cold_content[layer_num, partial_melties], 0.)  # actual melt not used for reducing the cold content
        s.snow.melt[partial_melties] += actual_layer_melt
        s.snow.cold_content[layer_num, partial_melties] = s.snow.cold_content[layer_num, partial_melties].clip(min=0)

        available_melt_time[melties] -= layer_melt_time[melties]

    # Iteration for calculating snow surface temperature
    if frosties.any():
        iteraties = frosties
        max_temp = constants.T0
        min_temp = s.meteo.temp[frosties].min() - 3.  # TODO this might be not realistic
        temp_inc = -0.25
        for surf_temp_iter in np.arange(max_temp, min_temp - 1e-6, temp_inc):
            s.surface.temp[iteraties] = surf_temp_iter

            advect_heat_flux = 0.  # XXX
            surf_heat_flux = -2.  # XXX

            calc_radiation_balance(model, iteraties)
            calc_fluxes(model, iteraties, surface=False, moisture=True, sensible=True, latent=True)

            en_bal[iteraties] = (
                s.meteo.net_radiation[iteraties]
                - s.surface.sens_heat_flux[iteraties]
                - s.surface.lat_heat_flux[iteraties]
                - advect_heat_flux
                - surf_heat_flux
            )

            iteraties = model.roi_mask_to_global(frosties[roi] & (en_bal[roi] < 0.))


def stability_factor(
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


def calc_turbulent_exchange_coefficient(model):
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
        sf = stability_factor(
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
        coeff *= sf

    s.surface.turbulent_exchange_coeff[roi] = coeff


def calc_radiation_balance(model, pos=None):
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


def calc_saturation_specific_humidity(model, pos):
    s = model.state
    sat_vap_press_surf = meteo.saturation_vapor_pressure(s.surface.temp[pos])
    s.surface.sat_spec_hum[pos] = meteo.specific_humidity(
        s.meteo.atmos_press[pos],
        sat_vap_press_surf,
    )


def calc_moisture_availability(model, pos):
    s = model.state
    moisture_availability = s.surface.conductance[pos] / (  # eq. (38) from [2]
        s.surface.conductance[pos]
        + s.surface.turbulent_exchange_coeff[pos] * s.meteo.wind_speed[pos]
    )
    moisture_availability[s.surface.sat_spec_hum[pos] < s.meteo.spec_hum[pos]] = 1.
    moisture_availability[s.snow.swe[pos] > 0] = 1.
    s.surface.moisture_availability[pos] = moisture_availability


def calc_latent_heat(model, pos):
    s = model.state
    s.surface.lat_heat[pos] = np.where(
        (s.snow.ice_content[0, pos] > 0) | (s.surface.temp[pos] < constants.T0),
        constants.LATENT_HEAT_OF_SUBLIMATION,
        constants.LATENT_HEAT_OF_VAPORIZATION,
    )


def calc_fluxes(model, pos, surface=True, moisture=True, sensible=True, latent=True):
    s = model.state

    rhoa_CH_Ua = (  # (kg m-2 s-1)
        s.meteo.dry_air_density[pos]
        * s.surface.turbulent_exchange_coeff[pos]
        * s.meteo.wind_speed[pos]
    )

    if surface:
        s.surface.heat_flux[pos] = (
            2 * s.surface.therm_cond[pos]
            * (s.surface.temp[pos] - s.surface.layer_temp[pos])
            / s.surface.thickness[pos]
        )

    if moisture:
        s.surface.moisture_flux[pos] = (
            s.surface.moisture_availability[pos]
            * rhoa_CH_Ua
            * (s.surface.sat_spec_hum[pos] - s.meteo.spec_hum[pos])
        )

    if sensible:
        s.surface.sens_heat_flux[pos] = (
            constants.SPEC_HEAT_CAP_DRY_AIR
            * rhoa_CH_Ua
            * (s.surface.temp[pos] - s.meteo.temp[pos])
        )

    if latent:
        s.surface.lat_heat_flux[pos] = s.surface.lat_heat[pos] * s.surface.moisture_flux[pos]


def solve_energy_balance(model, pos):
    s = model.state

    rhoa_CH_Ua = (  # (kg m-2 s-1)
        s.meteo.dry_air_density[pos]
        * s.surface.turbulent_exchange_coeff[pos]
        * s.meteo.wind_speed[pos]
    )

    dQsat_by_dTs = s.surface.lat_heat[pos] * s.surface.sat_spec_hum[pos] / (  # eq. (37) (K-1)
        constants.SPEC_GAS_CONSTANT_WATER_VAPOR * s.surface.temp[pos]**2
    )

    surf_temp_change = (  # eq. (38) (K)
        (
            s.meteo.net_radiation[pos]
            - s.surface.sens_heat_flux[pos]
            - s.surface.lat_heat_flux[pos]
            - s.surface.heat_flux[pos]
            - constants.LATENT_HEAT_OF_FUSION * s.snow.melt[pos]
        ) / (
            (
                constants.SPEC_HEAT_CAP_DRY_AIR
                + s.surface.lat_heat[pos]
                * s.surface.moisture_availability[pos]
                * dQsat_by_dTs
            ) * rhoa_CH_Ua
            + 2 * s.surface.therm_cond[pos] / s.surface.thickness[pos]
            + 4 * constants.STEFAN_BOLTZMANN * s.surface.temp[pos]**3
        )
    )

    surf_moisture_flux_change = (  # eq. (33) (kg m-2 s-1)
        s.surface.moisture_availability[pos]
        * rhoa_CH_Ua
        * dQsat_by_dTs
        * surf_temp_change
    )

    surf_heat_flux_change = (  # eq. (34) (W m-2)
        2 * s.surface.therm_cond[pos]
        * surf_temp_change / s.surface.thickness[pos]
    )

    sens_heat_flux_change = (  # eq. (35) (W m-2)
        constants.SPEC_HEAT_CAP_DRY_AIR
        * rhoa_CH_Ua
        * surf_temp_change
    )

    return (
        surf_temp_change,
        surf_moisture_flux_change,
        surf_heat_flux_change,
        sens_heat_flux_change,
    )
