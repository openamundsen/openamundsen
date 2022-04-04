from numba import njit, prange
import numpy as np
from openamundsen import constants, meteo
from openamundsen.modules.snow import CryoLayerID


def surface_properties(model):
    """
    Update surface properties following [1].

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

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

    s.surface.albedo[:] = np.where(
        s.snow.area_fraction > 0,
        (
            s.snow.area_fraction * s.snow.albedo
            + (1 - s.snow.area_fraction) * model.config.soil.albedo
        ),
        model.config.soil.albedo,
    )

    s.surface.roughness_length[:] = (
        model.config.snow.roughness_length**s.snow.area_fraction
        * model.config.soil.roughness_length**(1 - s.snow.area_fraction)
    )
    calc_turbulent_exchange_coefficient(model)

    if model.config.snow.model == 'multilayer':
        # Calculate surface conductance (eq. (35) from [2])
        # (the constant 1/100 therein corresponds to
        # model.config.soil.saturated_soil_surface_conductance; the clipping of (theta_1/theta_c)**2
        # to 1 is taken from FSM)
        s.surface.conductance[:] = model.config.soil.saturated_soil_surface_conductance * (
            np.maximum(
                (
                    s.soil.frac_unfrozen_moisture_content[0, :]
                    * s.soil.vol_moisture_content_sat / s.soil.vol_moisture_content_crit
                )**2,
                1,
            )
        )

        surface_layer_properties(model)


def surface_layer_properties(model):
    """
    Update surface layer properties following [1].

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    References
    ----------
    .. [1] Cox, P. M., Betts, R. A., Bunton, C. B., Essery, R. L. H., Rowntree,
       P. R., & Smith, J. (1999). The impact of new land surface physics on the
       GCM simulation of climate and climate sensitivity. Climate Dynamics, 15(3),
       183–203. https://doi.org/10.1007/s003820050276
    """
    s = model.state

    s.surface.thickness[:] = np.maximum(s.snow.thickness[0, :], s.soil.thickness[0, :])
    s.surface.layer_temp[:] = (
        s.soil.temp[0, :]
        + (s.snow.temp[0, :] - s.soil.temp[0, :])
        * s.snow.thickness[0, :] / s.soil.thickness[0, :]
    )

    pos1 = s.snow.thickness[0, :] <= s.soil.thickness[0, :] / 2
    pos2 = s.snow.thickness[0, :] > s.soil.thickness[0, :]

    # Effective surface thermal conductivity (eq. (79))
    s.surface.therm_cond[:] = s.snow.therm_cond[0, :]
    s.surface.therm_cond[pos1] = (
        s.soil.thickness[0, pos1]
        / (
            2 * s.snow.thickness[0, pos1] / s.snow.therm_cond[0, pos1]
            + (
                s.soil.thickness[0, pos1] - 2 * s.snow.thickness[0, pos1]
            ) / s.soil.therm_cond[0, pos1]
        )
    )

    s.surface.layer_temp[pos2] = s.snow.temp[0, pos2]


def energy_balance(model):
    snow_model = model.config.snow.model

    if snow_model == 'multilayer':
        multilayer_energy_balance(model)
    elif snow_model == 'cryolayers':
        cryo_layer_energy_balance(model)


def multilayer_energy_balance(model):
    """
    Calculate the surface energy balance for the multilayer model
    following [1].

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

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
    calc_radiation_balance(model, roi)
    calc_surface_flux(model, roi)
    calc_turbulent_fluxes(model, roi)
    calc_advective_heat(model, roi)

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
        s.snow.melt[melties] = s.snow.ice_content[:, melties].sum(axis=0)

        (
            surf_temp_change[melties_roi],
            surf_moisture_flux_change[melties_roi],
            surf_heat_flux_change[melties_roi],
            sens_heat_flux_change[melties_roi],
        ) = solve_energy_balance(model, melties)

        partial_melties_roi = melties_roi & (s.surface.temp[roi] + surf_temp_change < constants.T0)
        if partial_melties_roi.any():
            partial_melties = model.roi_mask_to_global(partial_melties_roi)

            s.surface.temp[partial_melties] = constants.T0

            calc_latent_heat(model, partial_melties)
            calc_saturation_specific_humidity(model, partial_melties)
            calc_radiation_balance(model, partial_melties)  # update net radiation
            calc_surface_flux(model, partial_melties)
            calc_turbulent_fluxes(model, partial_melties)
            calc_advective_heat(model, partial_melties)

            s.snow.melt[partial_melties] = (
                (
                    s.meteo.net_radiation[partial_melties]
                    + s.surface.sens_heat_flux[partial_melties]
                    + s.surface.lat_heat_flux[partial_melties]
                    + s.surface.advective_heat_flux[partial_melties]
                    - s.surface.heat_flux[partial_melties]
                ) / constants.LATENT_HEAT_OF_FUSION * model.timestep
            ).clip(min=0)

            surf_temp_change[partial_melties_roi] = 0.
            surf_moisture_flux_change[partial_melties_roi] = 0.
            surf_heat_flux_change[partial_melties_roi] = 0.
            sens_heat_flux_change[partial_melties_roi] = 0.

    # Update surface temperature and fluxes
    s.surface.temp[roi] += surf_temp_change
    s.surface.moisture_flux[roi] += surf_moisture_flux_change
    s.surface.heat_flux[roi] += surf_heat_flux_change
    s.surface.sens_heat_flux[roi] += sens_heat_flux_change
    calc_radiation_balance(model, roi)
    calc_latent_heat(model, roi)
    calc_turbulent_fluxes(model, roi, sensible=False)


def cryo_layer_energy_balance(model):
    """
    Calculate the surface energy balance for the cryo layer model following [1-2].

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    References
    ----------
    .. [1] Strasser, U. (2008). Die Modellierung der Gebirgsschneedecke im
       Nationalpark Berchtesgaden. Modelling of the mountain snow cover in the
       Berchtesgaden National Park, Berchtesgaden National Park research report,
       No. 55, Berchtesgaden.

    .. [2] Hanzer, F., Helfricht, K., Marke, T., & Strasser, U. (2016).
       Multilevel spatiotemporal validation of snow/ice mass balance and runoff
       modeling in glacierized catchments. The Cryosphere, 10(4), 1859–1881.
       https://doi.org/10.5194/tc-10-1859-2016
    """
    s = model.state
    roi = model.grid.roi

    s.surface.temp[roi] = s.meteo.temp[roi]

    snowies_roi = s.snow.swe[roi] > 0.
    melties_roi = snowies_roi & (s.meteo.temp[roi] >= constants.T0)
    frosties_roi = snowies_roi & (s.meteo.temp[roi] < constants.T0)
    snowies = model.roi_mask_to_global(snowies_roi)
    melties = model.roi_mask_to_global(melties_roi)
    frosties = model.roi_mask_to_global(frosties_roi)
    snow_freeies = model.roi_mask_to_global(~snowies_roi)

    calc_saturation_specific_humidity(model, roi)
    calc_moisture_availability(model, roi)
    calc_latent_heat(model, roi)

    s.snow.melt[roi] = 0.
    s.snow.refreezing[roi] = 0.

    s.surface.heat_flux[snowies] = model.config.snow.cryolayers.surface_heat_flux
    s.surface.heat_flux[snow_freeies] = np.nan

    # Where air temperature >= 0 °C -> potential melt, no iteration
    s.surface.temp[melties] = constants.T0
    available_melt_time = np.zeros(roi.shape)
    en_bal = np.full(roi.shape, np.nan)
    available_melt_time[melties] = model.timestep  # contains the time (in seconds) available for melt in this time step for each pixel

    for layer_num in range(model.snow.num_cryo_layers):
        possible_melties = model.roi_mask_to_global(
            melties_roi
            & (s.snow.thickness[layer_num, roi] > 0)
            & (available_melt_time[roi] > 0)
        )

        en_bal[possible_melties] = energy_balance_remainder(
            model,
            possible_melties,
            s.surface.temp[possible_melties],
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
        layer_melt = (  # actual melt and cold content reduction
            en_bal[partial_melties]
            * available_melt_time[partial_melties]
            / constants.LATENT_HEAT_OF_FUSION
        )
        s.snow.cold_content[layer_num, partial_melties] -= layer_melt
        actual_layer_melt = -1 * np.minimum(s.snow.cold_content[layer_num, partial_melties], 0.)  # actual melt not used for reducing the cold content
        s.snow.melt[partial_melties] += actual_layer_melt
        s.snow.cold_content[layer_num, partial_melties] = (
            s.snow.cold_content[layer_num, partial_melties].clip(min=0)
        )

        available_melt_time[melties] -= layer_melt_time[melties]

    # Refreeze liquid water and build up cold content
    en_bal[frosties] = energy_balance_remainder(model, frosties, constants.T0)
    refreezing_factor = model.config.snow.cryolayers.refreezing_factor
    available_cc = (
        -1
        * en_bal * model.timestep / constants.LATENT_HEAT_OF_FUSION
        * refreezing_factor
    ).clip(min=0)  # kg m-2
    cold_holding_capacity = model.config.snow.cryolayers.cold_holding_capacity
    for layer_num in range(model.snow.num_cryo_layers):
        # No cold content for firn and ice
        if layer_num not in (CryoLayerID.NEW_SNOW, CryoLayerID.OLD_SNOW):
            continue

        cold_conties = model.global_mask(
            frosties_roi
            & (s.snow.thickness[layer_num, roi] > 0)
            & (available_cc[roi] > 0)
        )

        refreeze_amount = np.minimum(
            s.snow.liquid_water_content[layer_num, cold_conties],
            available_cc[cold_conties],
        )
        s.snow.liquid_water_content[layer_num, cold_conties] -= refreeze_amount
        s.snow.refreezing[cold_conties] += refreeze_amount
        available_cc[cold_conties] -= refreeze_amount

        max_layer_cc = cold_holding_capacity * (
            s.snow.ice_content[layer_num, cold_conties]
            + s.snow.liquid_water_content[layer_num, cold_conties]
        )
        cc_buildup = np.minimum(
            available_cc[cold_conties],
            max_layer_cc - s.snow.cold_content[layer_num, cold_conties],
        )
        s.snow.cold_content[layer_num, cold_conties] += cc_buildup
        available_cc[cold_conties] -= cc_buildup

    # Iteration for calculating snow surface temperature
    iterate_surface_temperature(model, frosties)

    calc_radiation_balance(model, roi)
    calc_turbulent_fluxes(model, roi)
    calc_advective_heat(model, roi)


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
    # Replace zero wind speeds (would lead to a divide by zero) with a plausible value
    wind_speed = wind_speed.copy()
    wind_speed[wind_speed == 0.] = 0.5

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

    pos = richardson >= 0

    # eq. (26)
    c = (
        3 * stability_adjustment_parameter**2 * constants.VON_KARMAN**2
        * np.sqrt(wind_measurement_height / momentum_roughness_length[~pos])
        / (np.log(wind_measurement_height / momentum_roughness_length[~pos]))**2
    )

    # eq. (25)
    stability_factor = np.empty(pos.shape)
    stability_factor[pos] = 1 / (1 + 3 * stability_adjustment_parameter * richardson[pos] * np.sqrt(
        1 + stability_adjustment_parameter * richardson[pos]))
    stability_factor[~pos] = 1 - 3 * stability_adjustment_parameter * richardson[~pos] / (
        1 + c * np.sqrt(-richardson[~pos]))

    stability_factor[np.isnan(stability_factor)] = 1.

    return stability_factor


def calc_turbulent_exchange_coefficient(model):
    """
    Calculate the transfer coefficient C_H from [1] (eq. (22)) including the
    possible adjustment for atmospheric stability.

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

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


def calc_radiation_balance(model, pos):
    """
    Calculate outgoing shortwave and longwave radiation and net radiation.

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    pos : ndarray(bool)
        Pixels for which the radiation fluxes should be calculated.
    """
    s = model.state

    # TODO this should be a parameter
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
    """
    Calculate saturation specific humidity at surface temperature.

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    pos : ndarray(bool)
        Pixels to be considered.
    """
    s = model.state
    sat_vap_press_surf = meteo.saturation_vapor_pressure(s.surface.temp[pos])
    s.surface.sat_spec_hum[pos] = meteo.specific_humidity(
        s.meteo.atmos_press[pos],
        sat_vap_press_surf,
    )


def calc_moisture_availability(model, pos):
    """
    Calculate moisture availability following [1-2].

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    pos : ndarray(bool)
        Pixels to be considered.

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
    moisture_availability = s.surface.conductance[pos] / (  # eq. (38) from [2]
        s.surface.conductance[pos]
        + s.surface.turbulent_exchange_coeff[pos] * s.meteo.wind_speed[pos]
    )
    moisture_availability[s.surface.sat_spec_hum[pos] < s.meteo.spec_hum[pos]] = 1.
    moisture_availability[s.snow.swe[pos] > 0] = 1.
    s.surface.moisture_availability[pos] = moisture_availability


def calc_latent_heat(model, pos):
    """
    Calculate latent heat.

    Latent heat is set to the latent heat of sublimation for snow covered
    surfaces, and to the latent heat of vaporization elsewhere.

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    pos : ndarray(bool)
        Pixels to be considered.
    """
    s = model.state
    lat_heat_vap = meteo.latent_heat_of_vaporization(s.meteo.temp[pos])
    s.surface.lat_heat[pos] = np.where(
        (s.snow.ice_content[0, pos] > 0) | (s.surface.temp[pos] < constants.T0),
        lat_heat_vap + constants.LATENT_HEAT_OF_FUSION,
        lat_heat_vap,
    )


def calc_surface_flux(model, pos):
    """
    Calculate the surface flux following [1].

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    pos : ndarray(bool)
        Pixels to be considered.

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    s = model.state
    s.surface.heat_flux[pos] = (
        2 * s.surface.therm_cond[pos]
        * (s.surface.temp[pos] - s.surface.layer_temp[pos])
        / s.surface.thickness[pos]
    )


def calc_turbulent_fluxes(model, pos, sensible=True, latent=True):
    """
    Calculate turbulent fluxes following [1] (opposed to [1] however, the fluxes
    are here oriented towards the surface).

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    pos : ndarray(bool)
        Pixels to be considered.

    sensible : bool, default True
        Calculate the sensible heat flux.

    latent : bool, default True
        Calculate the latent heat flux.

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    s = model.state

    rhoa_CH_Ua = (  # (kg m-2 s-1)
        s.meteo.dry_air_density[pos]
        * s.surface.turbulent_exchange_coeff[pos]
        * s.meteo.wind_speed[pos]
    )

    if sensible:
        s.surface.sens_heat_flux[pos] = (
            constants.SPEC_HEAT_CAP_DRY_AIR
            * rhoa_CH_Ua
            * (s.meteo.temp[pos] - s.surface.temp[pos])
        )

    if latent:
        s.surface.moisture_flux[pos] = (
            s.surface.moisture_availability[pos]
            * rhoa_CH_Ua
            * (s.meteo.spec_hum[pos] - s.surface.sat_spec_hum[pos])
        )
        s.surface.lat_heat_flux[pos] = s.surface.lat_heat[pos] * s.surface.moisture_flux[pos]


def calc_advective_heat(model, pos):
    """
    Calculate heat advected by precipitation following [1].

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    pos : ndarray(bool)
        Pixels to be considered.

    References
    ----------
    .. [1] Strasser, U. (2008). Die Modellierung der Gebirgsschneedecke im
       Nationalpark Berchtesgaden. Modelling of the mountain snow cover in the
       Berchtesgaden National Park, Berchtesgaden National Park research report,
       No. 55, Berchtesgaden.
    """
    s = model.state
    s.surface.advective_heat_flux[pos] = (
        (  # rainfall on snow
            constants.SPEC_HEAT_CAP_WATER
            * (s.meteo.temp[pos] - s.surface.temp[pos])
            * s.meteo.rainfall[pos]
        ) + (  # snowfall on snow
            constants.SPEC_HEAT_CAP_ICE
            * (s.meteo.wet_bulb_temp[pos] - s.surface.temp[pos])
            * s.meteo.snowfall[pos]
        )
    ) / model.timestep


def solve_energy_balance(model, pos):
    """
    Calculate surface temperature and flux changes following [1] (eqs.
    (32)-(35)).

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    pos : ndarray(bool)
        Pixels to be considered.

    Returns
    -------
    length-4 ndarray tuple
        Tuple containing the surface temperature change, moisture flux change,
        surface heat flux change and sensible heat flux change.

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    s = model.state

    rhoa_CH_Ua = (  # (kg m-2 s-1)
        s.meteo.dry_air_density[pos]
        * s.surface.turbulent_exchange_coeff[pos]
        * s.meteo.wind_speed[pos]
    )

    dQsat_by_dTs = s.surface.lat_heat[pos] * s.surface.sat_spec_hum[pos] / (  # eq. (37) (K-1)
        constants.SPEC_GAS_CONSTANT_WATER_VAPOR * s.surface.temp[pos]**2
    )

    surf_temp_change = (  # eq. (38) plus heat advected by precipitation
        (
            s.meteo.net_radiation[pos]
            + s.surface.sens_heat_flux[pos]
            + s.surface.lat_heat_flux[pos]
            + s.surface.advective_heat_flux[pos]
            - s.surface.heat_flux[pos]
            - constants.LATENT_HEAT_OF_FUSION * (s.snow.melt[pos] / model.timestep)
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

    surf_temp_change[np.isnan(surf_temp_change)] = 0.
    surf_moisture_flux_change[np.isnan(surf_moisture_flux_change)] = 0.
    surf_heat_flux_change[np.isnan(surf_heat_flux_change)] = 0.
    sens_heat_flux_change[np.isnan(sens_heat_flux_change)] = 0.

    return (
        surf_temp_change,
        surf_moisture_flux_change,
        surf_heat_flux_change,
        sens_heat_flux_change,
    )


def energy_balance_remainder(model, pos, surf_temp):
    """
    Update the radiation balance and turbulent fluxes and calculate the
    remainder of the surface energy balance assuming no melt for a given surface
    temperature following [1-2].

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    pos : ndarray(bool)
        Pixels to be considered.

    surf_temp : ndarray
        Surface temperature (K).

    Returns
    -------
    en_bal : ndarray
        Surface energy balance remainder.

    References
    ----------
    .. [1] Strasser, U. (2008). Die Modellierung der Gebirgsschneedecke im
       Nationalpark Berchtesgaden. Modelling of the mountain snow cover in the
       Berchtesgaden National Park, Berchtesgaden National Park research report,
       No. 55, Berchtesgaden.

    .. [2] Hanzer, F., Helfricht, K., Marke, T., & Strasser, U. (2016).
       Multilevel spatiotemporal validation of snow/ice mass balance and runoff
       modeling in glacierized catchments. The Cryosphere, 10(4), 1859–1881.
       https://doi.org/10.5194/tc-10-1859-2016
    """
    s = model.state
    s.surface.temp[pos] = surf_temp

    calc_radiation_balance(model, pos)
    calc_saturation_specific_humidity(model, pos)
    calc_turbulent_fluxes(model, pos)
    calc_advective_heat(model, pos)

    en_bal = (
        s.meteo.net_radiation[pos]
        + s.surface.sens_heat_flux[pos]
        + s.surface.lat_heat_flux[pos]
        + s.surface.advective_heat_flux[pos]
        - s.surface.heat_flux[pos]
    )

    return en_bal


def iterate_surface_temperature(model, frosties):
    """
    Iterate the snow surface temperature to find the zeros of the energy balance
    equation assuming no melt.

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    frosties : ndarray(bool)
        Pixels to be considered.
    """
    if not frosties.any():
        return

    s = model.state
    method = model.config.snow.cryolayers.surface_temperature_iteration_method

    if method == 'legacy':
        roi = model.grid.roi
        en_bal = np.zeros(roi.shape)
        iteraties = frosties
        max_temp = constants.T0
        min_temp = s.meteo.temp[frosties].min() - 3.
        temp_inc = -0.25
        for surf_temp_iter in np.arange(max_temp, min_temp - 1e-6, temp_inc):
            en_bal[iteraties] = energy_balance_remainder(model, iteraties, surf_temp_iter)
            iteraties = model.roi_mask_to_global(frosties[roi] & (en_bal[roi] < 0.))
    elif method == 'secant':
        tol = 1e-2

        iteraties = frosties.copy()
        iteraties_idxs = np.where(iteraties.flat)[0]
        x0 = np.full(len(iteraties_idxs), constants.T0 - 10.)
        x1 = np.full(len(iteraties_idxs), constants.T0)
        y0 = energy_balance_remainder(model, iteraties, x0)
        y1 = energy_balance_remainder(model, iteraties, x1)

        while True:
            d = (x1 - x0) / (y1 - y0) * y1  # secant method
            iter_pos = np.abs(d) > tol

            if iter_pos.sum() == 0:
                break

            d = d[iter_pos]
            x0 = x0[iter_pos]
            x1 = x1[iter_pos]
            y0 = y0[iter_pos]
            y1 = y1[iter_pos]

            x0 = x1.copy()
            y0 = y1.copy()
            x1 -= d
            iteraties.flat[iteraties_idxs[~iter_pos]] = False
            y1 = energy_balance_remainder(model, iteraties, x1)
            iteraties_idxs = iteraties_idxs[iter_pos]

        # TODO calculate melt when surface temperature is positive after iteration?
        s.surface.temp[frosties] = s.surface.temp[frosties].clip(max=constants.T0)
