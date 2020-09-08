from openamundsen import constants
from openamundsen.snowmodel import SnowModel
from . import snow


class LayerSnowModel(SnowModel):
    def __init__(self, model):
        self.model = model

        s = model.state.snow
        num_snow_layers = len(model.config.snow.min_thickness)

        s.add_variable('num_layers', '1', 'Number of snow layers', dtype=int)
        s.add_variable('thickness', 'm', 'Snow thickness', dim3=num_snow_layers)
        s.add_variable('density', 'kg m-3', 'Snow density', 'snow_density', dim3=num_snow_layers)
        s.add_variable('ice_content', 'kg m-2', 'Ice content of snow', dim3=num_snow_layers)
        s.add_variable('liquid_water_content', 'kg m-2', 'Liquid water content of snow', dim3=num_snow_layers)
        s.add_variable('temp', 'K', 'Snow temperature', dim3=num_snow_layers)
        s.add_variable('therm_cond', 'W m-1 K-1', 'Thermal conductivity of snow', dim3=num_snow_layers)
        s.add_variable('areal_heat_cap', 'J K-1 m-2', 'Areal heat capacity of snow', dim3=num_snow_layers)

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
