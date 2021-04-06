from openamundsen import constants


def initialize(model):
    """
    Initialize state variables for the soil model.

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    References
    ----------
    .. [1] Cox, P. M., Betts, R. A., Bunton, C. B., Essery, R. L. H., Rowntree,
       P. R., & Smith, J. (1999). The impact of new land surface physics on the
       GCM simulation of climate and climate sensitivity. Climate Dynamics, 15(3),
       183–203.  https://doi.org/10.1007/s003820050276

    .. [2] Best, M. J., Pryor, M., Clark, D. B., Rooney, G. G., Essery, R. L. H.,
       Ménard, C. B., Edwards, J. M., Hendry, M. A., Porson, A., Gedney, N.,
       Mercado, L. M., Sitch, S., Blyth, E., Boucher, O., Cox, P. M., Grimmond, C.
       S. B., & Harding, R. J. (2011). The Joint UK Land Environment Simulator
       (JULES), model description – Part 1: Energy and water fluxes. Geoscientific
       Model Development, 4(3), 677–699. https://doi.org/10.5194/gmd-4-677-2011

    .. [3] Chadburn, S., Burke, E., Essery, R., Boike, J., Langer, M., Heikenfeld, M.,
       Cox, P., & Friedlingstein, P. (2015). An improved representation of
       physical permafrost dynamics in the JULES land-surface model. Geoscientific
       Model Development, 8(5), 1493–1508. https://doi.org/10.5194/gmd-8-1493-2015
    """
    roi = model.grid.roi
    cfg = model.config.soil
    s = model.state.soil

    # Temperature
    s.temp[:, roi] = cfg.init_temp

    # Thickness
    for i, thickness in enumerate(cfg.thickness):
        s.thickness[i, roi] = thickness

    # Volumetric heat capacity of dry soil
    s.vol_heat_cap_dry[roi] = (
        (
            constants.VOL_HEAT_CAP_SAND * cfg.sand_fraction
            + constants.VOL_HEAT_CAP_CLAY * cfg.clay_fraction
        ) / (cfg.sand_fraction + cfg.clay_fraction)
    )

    # Clapp-Hornberger exponent b
    s.clapp_hornberger[roi] = 3.1 + 15.7 * cfg.clay_fraction - 0.3 * cfg.sand_fraction

    # Saturated soil water pressure
    s.sat_water_pressure[roi] = 10**(0.17 - 0.63 * cfg.clay_fraction - 1.58 * cfg.sand_fraction)

    # Volumetric soil moisture content at saturation (values from doi:10.1029/wr020i006p00682, table 4)
    s.vol_moisture_content_sat[roi] = 0.505 - 0.037 * cfg.clay_fraction - 0.142 * cfg.sand_fraction

    # Volumetric soil moisture content at critical point
    s.vol_moisture_content_crit[roi] = (
        s.vol_moisture_content_sat[roi]
        * (
            s.sat_water_pressure[roi]
            / 3.364  # soil suction of 3.364 m
        )**(1. / s.clapp_hornberger[roi])
    )

    # Thermal conductivity of soil minerals
    s.therm_cond_minerals[roi] = (
        (constants.THERM_COND_CLAY**cfg.clay_fraction)
        * (constants.THERM_COND_SAND**(1 - cfg.clay_fraction))  # TODO really 1 - clay_fraction and not just sand_fraction?
    )

    # Thermal conductivity of dry soil
    s.therm_cond_dry[roi] = (
        (constants.THERM_COND_AIR**s.vol_moisture_content_sat[roi])
        * s.therm_cond_minerals[roi]**(1 - s.vol_moisture_content_sat[roi])
    )

    # Volumetric soil moisture content
    s.vol_moisture_content[:, roi] = cfg.init_moisture_content * s.vol_moisture_content_sat[roi]
