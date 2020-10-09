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
        pos = s.meteo.snowfall > 0
        self.add_snow(
            pos,
            s.meteo.snowfall[pos],
            density=snow.fresh_snow_density(s.meteo.wetbulb_temp[pos]),
        )

    def heat_conduction(self):
        model = self.model
        s = model.state
        snow._heat_conduction(
            model.grid.roi_idxs,
            s.snow.num_layers,
            s.snow.thickness,
            s.soil.thickness,
            model.timestep,
            s.snow.temp,
            s.snow.therm_cond,
            s.soil.therm_cond,
            s.surface.heat_flux,
            s.snow.heat_cap,
        )

    def melt(self):
        model = self.model
        s = model.state
        snow._melt(
            model.grid.roi_idxs,
            model.timestep,
            s.snow.num_layers,
            s.snow.melt,
            s.snow.thickness,
            s.snow.temp,
            s.snow.ice_content,
            s.snow.liquid_water_content,
            s.snow.heat_cap,
        )

    def sublimation(self):
        model = self.model
        s = model.state

        # First resublimation
        frost = -np.minimum(s.snow.sublimation, 0)
        pos = frost > 0
        self.add_snow(
            pos,
            frost[pos],
            density=snow.fresh_snow_density(s.meteo.wetbulb_temp[pos]),
        )

        # Then sublimation
        snow._sublimation(
            model.grid.roi_idxs,
            model.timestep,
            s.snow.num_layers,
            s.snow.ice_content,
            s.snow.thickness,
            s.snow.sublimation,
        )

    def runoff(self):
        model = self.model
        s = model.state
        snow._runoff(
            model.grid.roi_idxs,
            snow.max_liquid_water_content(model),
            s.meteo.rainfall,
            s.snow.num_layers,
            s.snow.thickness,
            s.snow.temp,
            s.snow.ice_content,
            s.snow.liquid_water_content,
            s.snow.runoff,
            s.snow.heat_cap,
        )

    def update_layers(self):
        model = self.model
        s = model.state
        snow._update_layers(
            model.grid.roi_idxs,
            s.snow.num_layers,
            np.array(model.config.snow.min_thickness),
            s.snow.thickness,
            s.snow.ice_content,
            s.snow.liquid_water_content,
            s.snow.heat_cap,
            s.snow.temp,
            s.snow.depth,
        )

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
