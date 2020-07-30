from openamundsen import constants


def initialize(model):
    """
    Initialize state variables for the snow layer model.

    Parameters
    ----------
    model : Model
        Model instance.

    References
    ----------

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    roi = model.grid.roi
    s = model.state.snow

    s.swe[roi] = 0
    s.depth[roi] = 0
    s.area_fraction[roi] = 0
    s.num_layers[roi] = 0
    s.sublimation[roi] = 0
    s.therm_cond[:, roi] = model.config.snow.thermal_conductivity
    s.thickness[:, roi] = 0
    s.ice_content[:, roi] = 0
    s.liquid_water_content[:, roi] = 0
    s.temp[:, roi] = constants.T0
