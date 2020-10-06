import numpy as np
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
        s.add_variable('heat_cap', 'J K-1 m-2', 'Areal heat capacity of snow', dim3=num_snow_layers)

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
        model = self.model
        s = model.state
        roi = model.grid.roi

        self.add_snow(
            roi,
            s.meteo.snowfall[roi],
            density=snow._fresh_snow_density(s.meteo.wetbulb_temp[roi]),
        )

    def heat_conduction(self):
        snow.heat_conduction(self.model)

    def melt(self):
        snow.melt(self.model)

    def sublimation(self):
        model = self.model
        s = model.state
        roi = model.grid.roi

        # First resublimation
        frost = -np.minimum(s.snow.sublimation[roi], 0)
        self.add_snow(
            roi,
            frost,
            density=snow._fresh_snow_density(s.meteo.wetbulb_temp[roi]),
        )

        # Then sublimation
        snow.sublimation(self.model)

    def runoff(self):
        snow.runoff(self.model)

    def update_layers(self):
        snow.update_layers(self.model)

    def update_properties(self):
        snow.snow_properties(self.model)

    def add_snow(
            self,
            pos,
            ice_content,
            liquid_water_content=0,
            density=None,
            albedo=None,
    ):
        """
        Add snow to the top of the snowpack.
        """
        model = self.model
        s = model.state

        ice_content = np.nan_to_num(ice_content, nan=0., copy=True)

        pos_init = (s.snow.num_layers[pos] == 0) & (ice_content > 0)
        pos_init_global = model.global_mask(pos_init, pos)

        # If albedo is None, set it to the maximum albedo for currently snow-free pixels and keep
        # the current albedo for the other pixels
        if albedo is None:
            albedo = s.snow.albedo[pos]
            albedo[pos_init] = model.config.snow.albedo.max
            s.snow.albedo[pos] = albedo

        # Initialize first snow layer where necessary
        s.snow.num_layers[pos_init_global] = 1
        s.snow.temp[0, pos_init_global] = np.minimum(s.meteo.temp[pos_init_global], constants.T0)

        # Add snow to first layer
        s.snow.ice_content[0, pos] += ice_content
        s.snow.liquid_water_content[0, pos] += liquid_water_content
        s.snow.thickness[0, pos] += ice_content / density
