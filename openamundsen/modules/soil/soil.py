from numba import njit, prange
import numpy as np
from openamundsen import (
    constants as c,
    heatconduction,
)


def soil_properties(model):
    """
    Wrapper function for _soil_properties().
    """
    s = model.state

    _soil_properties(
        model.grid.roi_idxs,
        s.soil.thickness,
        s.soil.temp,
        s.soil.heat_cap,
        s.soil.vol_heat_cap_dry,
        s.soil.therm_cond,
        s.soil.therm_cond_dry,
        s.soil.vol_moisture_content,
        s.soil.vol_moisture_content_sat,
        s.soil.frac_frozen_moisture_content,
        s.soil.frac_unfrozen_moisture_content,
        s.soil.sat_water_pressure,
        s.soil.clapp_hornberger,
    )


@njit(parallel=True, cache=True)
def _soil_properties(
    roi_idxs,
    thickness,
    temp,
    heat_cap,
    vol_heat_cap_dry,
    therm_cond,
    therm_cond_dry,
    vol_moisture_content,
    vol_moisture_content_sat,
    frac_frozen_moisture_content,
    frac_unfrozen_moisture_content,
    sat_water_pressure,
    clapp_hornberger,
):
    """
    Calculate soil properties following [1] (equation numbers in the comments
    refer to this article).

    Parameters
    ----------
    roi_idxs : ndarray(int, ndim=2)
        (N, 2)-array specifying the (row, col) indices within the data arrays
        that should be considered.

    thickness : ndarray(float, ndim=3)
        Soil thickness (m).

    temp : ndarray(float, ndim=3)
        Soil temperature (K).

    heat_cap : ndarray(float, ndim=3)
        Soil areal heat capacity (J K-1 m-2).

    vol_heat_cap_dry : ndarray(float, ndim=2)
        Volumetric heat capacity of dry soil (J K-1 m-3).

    therm_cond : ndarray(float, ndim=3)
        Soil thermal conductivity (W m-1 K-1).

    therm_cond_dry : ndarray(float, ndim=2)
        Dry soil thermal conductivity (W m-1 K-1).

    vol_moisture_content : ndarray(float, ndim=3)
        Volumetric soil moisture content (m3 m-3).

    vol_moisture_content_sat : ndarray(float, ndim=2)
        Volumetric soil moisture content at saturation (m3 m-3).

    frac_frozen_moisture_content : ndarray(float, ndim=3)
        Mass of frozen water as a fraction of that of liquid water at
        saturation.

    frac_unfrozen_moisture_content : ndarray(float, ndim=3)
        Mass of unfrozen water as a fraction of that of liquid water at
        saturation.

    sat_water_pressure : ndarray(float, ndim=2)
        Saturated soil water pressure (m).

    clapp_hornberger : ndarray(float, ndim=2)
        Clapp-Hornberger exponent.

    References
    ----------
    .. [1] Cox, P. M., Betts, R. A., Bunton, C. B., Essery, R. L. H., Rowntree,
       P. R., & Smith, J. (1999). The impact of new land surface physics on the
       GCM simulation of climate and climate sensitivity. Climate Dynamics, 15(3),
       183–203. https://doi.org/10.1007/s003820050276
    """
    num_pixels = len(roi_idxs)
    num_soil_layers = thickness.shape[0]

    # Calculate kappa (eq. (40))
    k = 1.  # dimensionless constant depending on the nature of the soil
    kappa = k * (
        (c.ICE_DENSITY / c.WATER_DENSITY)
        * c.LATENT_HEAT_OF_FUSION / (c.T0 * c.GRAVITATIONAL_ACCELERATION)
    )

    for idx_num in prange(num_pixels):
        i, j = roi_idxs[idx_num]

        for k in range(num_soil_layers):
            if vol_moisture_content[k, i, j] > 0:
                # Temperature above which all soil moisture is unfrozen (eq. (43))
                temp_max_frozen = (
                    c.T0
                    - sat_water_pressure[i, j] / kappa
                    * (
                        vol_moisture_content_sat[i, j] / vol_moisture_content[k, i, j]
                    )**clapp_hornberger[i, j]
                )

                if temp[k, i, j] < temp_max_frozen:
                    # Maximum unfrozen water that can exist at current soil temperature (eq. (39))
                    max_vol_unfrozen_moisture_content = (
                        vol_moisture_content_sat[i, j]
                        * (
                            -kappa * (temp[k, i, j] - c.T0) / sat_water_pressure[i, j]
                        )**(-1 / clapp_hornberger[i, j])
                    )

                    # Actual amount of unfrozen water (limited by the total water content of the soil)
                    # (eqs. (41), (44))
                    vol_unfrozen_moisture_content = min(
                        max_vol_unfrozen_moisture_content,
                        vol_moisture_content[k, i, j],
                    )

                    # Change of unfrozen water concentration by temperature (eq. (45))
                    dTheta_u_by_dT = (
                        kappa * vol_moisture_content_sat[i, j]
                        / (clapp_hornberger[i, j] * sat_water_pressure[i, j])
                        * (
                            -kappa * (temp[k, i, j] - c.T0)
                            / sat_water_pressure[i, j]
                        )**(-1 / clapp_hornberger[i, j] - 1)
                    )
                else:
                    vol_unfrozen_moisture_content = vol_moisture_content[k, i, j]
                    dTheta_u_by_dT = 0.

                vol_frozen_moisture_content = (
                    (vol_moisture_content[k, i, j] - vol_unfrozen_moisture_content)
                    * c.WATER_DENSITY / c.ICE_DENSITY
                )

                # Frozen/unfrozen moisture content (eqs. (48)-(49))
                frac_unfrozen_moisture_content[k, i, j] = vol_unfrozen_moisture_content / vol_moisture_content_sat[i, j]
                frac_frozen_moisture_content[k, i, j] = (
                    (c.ICE_DENSITY / c.WATER_DENSITY)
                    * vol_frozen_moisture_content / vol_moisture_content_sat[i, j]
                )

                # Volumetric heat capacity (eq. (37))
                vol_heat_cap = (
                    vol_heat_cap_dry[i, j]
                    + c.WATER_DENSITY * c.SPEC_HEAT_CAP_WATER * vol_unfrozen_moisture_content
                    + c.ICE_DENSITY * c.SPEC_HEAT_CAP_ICE * vol_frozen_moisture_content
                    + c.WATER_DENSITY * (
                        (c.SPEC_HEAT_CAP_WATER - c.SPEC_HEAT_CAP_ICE)
                        * temp[k, i, j]
                        + c.LATENT_HEAT_OF_FUSION
                    ) * dTheta_u_by_dT
                )

                # Areal heat capacity
                heat_cap[k, i, j] = vol_heat_cap * thickness[k, i, j]

                # Saturation concentration of liquid water for the current liquid water to
                # ice mass ratio (eq. (74))
                liquid_water_sat_conc = vol_moisture_content_sat[i, j] * (
                    frac_unfrozen_moisture_content[k, i, j]
                    / (frac_unfrozen_moisture_content[k, i, j] + frac_frozen_moisture_content[k, i, j])
                )

                # Thermal conductivity of saturated soil (eq. (75))
                therm_cond_sat = (
                    therm_cond_dry[i, j]
                    * c.THERM_COND_WATER**liquid_water_sat_conc
                    * c.THERM_COND_ICE**(vol_moisture_content_sat[i, j] - liquid_water_sat_conc)
                    / c.THERM_COND_AIR**vol_moisture_content_sat[i, j]
                )

                # Effective thermal conductivity (eq. (71))
                therm_cond[k, i, j] = (
                    (therm_cond_sat - therm_cond_dry[i, j])
                    * vol_moisture_content[k, i, j] / vol_moisture_content_sat[i, j]
                    + therm_cond_dry[i, j]
                )
            else:  # vol_moisture_content[k, i, j] == 0
                heat_cap[k, i, j] = vol_heat_cap_dry[i, j] * thickness[k, i, j]
                therm_cond[k, i, j] = therm_cond_dry[i, j]


def soil_temperature(model):
    """
    Update soil layer temperatures.

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
    roi = model.grid.roi
    thickness = model.state.soil.thickness[:, roi]
    temp = model.state.soil.temp[:, roi]
    therm_cond = model.state.soil.therm_cond[:, roi]

    dT = heatconduction.temp_change_array(
        thickness,
        model.timestep,
        temp,
        therm_cond,
        temp[-1, :],
        thickness[-1, :],
        therm_cond[-1, :],
        model.state.soil.heat_flux[roi],
        model.state.soil.heat_cap[:, roi],
    )
    model.state.soil.temp[:, roi] += dT


def soil_heat_flux(model):
    """
    Wrapper function for _soil_heat_flux().
    """
    s = model.state

    _soil_heat_flux(
        model.grid.roi_idxs,
        s.surface.heat_flux,
        s.soil.heat_flux,
        s.soil.thickness,
        s.soil.temp,
        s.soil.therm_cond,
        s.snow.num_layers,
        s.snow.thickness,
        s.snow.temp,
        s.snow.therm_cond,
    )


@njit(parallel=True, cache=True)
def _soil_heat_flux(
    roi_idxs,
    surf_heat_flux,
    soil_heat_flux,
    soil_thickness,
    soil_temp,
    therm_cond_soil,
    num_snow_layers,
    snow_thickness,
    snow_temp,
    therm_cond_snow,
):
    """
    Calculate the soil heat flux, i.e., the flux from the surface to the top
    soil layer, following [1].

    Parameters
    ----------
    roi_idxs : ndarray(int, ndim=2)
        (N, 2)-array specifying the (row, col) indices within the data arrays
        that should be considered.

    surf_heat_flux : ndarray(float, ndim=2)
        Surface heat flux (W m-2).

    soil_heat_flux : ndarray(float, ndim=2)
        Soil heat flux (W m-2).

    soil_thickness : ndarray(float, ndim=3)
        Soil thickness (m).

    soil_temp : ndarray(float, ndim=3)
        Soil temperature (K).

    therm_cond_soil : ndarray(float, ndim=3)
        Soil thermal conductivity (W m-1 K-1).

    num_snow_layers : ndarray(float, ndim=2)
        Number of snow layers.

    snow_thickness : ndarray(float, ndim=3)
        Snow thickness (m).

    snow_temp : ndarray(float, ndim=3)
        Snow temperature (K).

    therm_cond_snow : ndarray(float, ndim=3)
        Snow thermal conductivity (W m-1 K-1).

    References
    ----------
    .. [1] Cox, P. M., Betts, R. A., Bunton, C. B., Essery, R. L. H., Rowntree,
       P. R., & Smith, J. (1999). The impact of new land surface physics on the
       GCM simulation of climate and climate sensitivity. Climate Dynamics, 15(3),
       183–203. https://doi.org/10.1007/s003820050276
    """
    num_pixels = len(roi_idxs)
    for idx_num in prange(num_pixels):
        i, j = roi_idxs[idx_num]

        ns = num_snow_layers[i, j]

        if ns == 0:
            soil_heat_flux[i, j] = surf_heat_flux[i, j]
        else:
            surf_moisture_conductance_bottom = 2. / (
                snow_thickness[ns - 1, i, j] / therm_cond_snow[ns - 1, i, j]
                + soil_thickness[0, i, j] / therm_cond_soil[0, i, j]
            )
            soil_heat_flux[i, j] = (
                surf_moisture_conductance_bottom
                * (snow_temp[ns - 1, i, j] - soil_temp[0, i, j])
            )
