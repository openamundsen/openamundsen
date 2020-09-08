from openamundsen import constants
from openamundsen.snowmodel import SnowModel
from . import snow


class LayerSnowModel(SnowModel):
    def initialize(self):
        roi = self.model.grid.roi
        s = self.model.state.snow

        s.swe[roi] = 0
        s.depth[roi] = 0
        s.area_fraction[roi] = 0
        s.num_layers[roi] = 0
        s.sublimation[roi] = 0
        s.therm_cond[:, roi] = self.model.config.snow.thermal_conductivity
        s.thickness[:, roi] = 0
        s.ice_content[:, roi] = 0
        s.liquid_water_content[:, roi] = 0
        s.temp[:, roi] = constants.T0

    def albedo_aging(self):
        snow.albedo(self.model)

    def compaction(self):
        snow.compaction(self.model)

    def accumulation(self):
        snow.accumulation(self.model)

    def heat_conduction(self):
        snow.heat_conduction(self.model)

    def melt(self):
        snow.melt(self.model)

    def sublimation(self):
        snow.sublimation(self.model)

    def runoff(self):
        snow.runoff(self.model)

    def update_layers(self):
        snow.update_layers(self.model)

    def update_properties(self):
        snow.snow_properties(self.model)
