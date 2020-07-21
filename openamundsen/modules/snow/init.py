from openamundsen import constants


def initialize(model):
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
