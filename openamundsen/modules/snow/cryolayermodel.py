import numpy as np
from openamundsen import constants
from openamundsen.snowmodel import SnowModel
from .snow import (
    _compaction_anderson,
    fresh_snow_density,
    albedo,
    max_liquid_water_content,
    snow_properties,
)


class CryoLayerID:
    SNOW_FREE = -1
    NEW_SNOW = 0
    OLD_SNOW = 1
    FIRN = 2
    ICE = 3


class CryoLayerSnowModel(SnowModel):
    def __init__(self, model):
        s = model.state.snow
        num_layers = 4

        s.add_variable('num_layers', '1', 'Number of snow layers', dtype=int)
        s.add_variable('thickness', 'm', 'Snow thickness', dim3=num_layers)
        s.add_variable('density', 'kg m-3', 'Snow density', 'snow_density', dim3=num_layers)
        s.add_variable('ice_content', 'kg m-2', 'Ice content of snow', dim3=num_layers)
        s.add_variable('liquid_water_content', 'kg m-2', 'Liquid water content of snow', dim3=num_layers)
        s.add_variable('cold_content', 'kg m-2', 'Cold content of snow', dim3=num_layers)
        s.add_variable('temp', 'K', 'Snow temperature', dim3=num_layers)  # XXX only added because this is in the default point outputs
        s.add_variable('layer_albedo', '1', 'Snow layer albedo', dim3=num_layers)
        s.add_variable('heat_cap', 'J K-1 m-2', 'Areal heat capacity of snow', dim3=num_layers)  # XXX only added because of snow.snow_properties()

        self.model = model
        self.num_layers = num_layers

    def initialize(self):
        roi = self.model.grid.roi
        s = self.model.state.snow

        s.swe[roi] = 0
        s.depth[roi] = 0
        s.area_fraction[roi] = 0
        s.num_layers[roi] = 0
        s.sublimation[roi] = 0
        s.thickness[:, roi] = 0
        s.ice_content[:, roi] = 0
        s.liquid_water_content[:, roi] = 0
        s.cold_content[:, roi] = 0
        s.density[:, roi] = np.nan
        s.layer_albedo[:, roi] = np.nan
        # s.temp[:, roi] = np.nan

    def albedo_aging(self):
        model = self.model
        s = model.state

        self.update_surface_layer_type()

        for i in range(model.snow.num_layers):
            pos = s.surface.layer_type == i

            if i in (CryoLayerID.NEW_SNOW, CryoLayerID.OLD_SNOW):
                s.snow.albedo[pos] = s.snow.layer_albedo[i, pos]
                albedo(model, pos)
                s.snow.layer_albedo[i, pos] = s.snow.albedo[pos]
            elif i == CryoLayerID.FIRN:
                s.snow.layer_albedo[i, pos] = model.config.snow.albedo.firn
            elif i == CryoLayerID.ICE:
                s.snow.layer_albedo[i, pos] = model.config.snow.albedo.ice

    def compaction(self):
        model = self.model
        snow = model.state.snow

        # Consider only old snow and new snow, but always set the number of layers to 2 regardless
        # if both layers exist, due to the loop over the layers in _compaction_anderson (and the
        # fact that an old snow layer could exist without a new snow layer (so using 1 as the number
        # of layers would not work here)
        num_layers = np.full(model.grid.roi.shape, 2)

        if model.config.snow.compaction.method == 'anderson':
            _compaction_anderson(
                model.grid.roi_idxs,
                model.timestep,
                num_layers,
                snow.thickness,
                snow.ice_content,
                snow.liquid_water_content,
                snow.density,
                model.state.meteo.temp,
            )
        else:
            raise NotImplementedError

    def accumulation(self):
        model = self.model
        s = model.state

        density = fresh_snow_density(s.meteo.wetbulb_temp)

        pos_accum = s.meteo.snowfall > 0
        pos_init_layer = (s.snow.ice_content[0, :] == 0) & (s.meteo.snowfall > 0)

        # Initialize new snow layer where required
        s.snow.layer_albedo[CryoLayerID.NEW_SNOW, pos_init_layer] = model.config.snow.albedo.max
        s.snow.albedo[pos_init_layer] = s.snow.layer_albedo[CryoLayerID.NEW_SNOW, pos_init_layer]
        s.snow.density[CryoLayerID.NEW_SNOW, pos_init_layer] = density[pos_init_layer]

        # Add snow to new snow layer
        s.snow.ice_content[CryoLayerID.NEW_SNOW, :] += s.meteo.snowfall
        s.snow.thickness[CryoLayerID.NEW_SNOW, :] += s.meteo.snowfall / density
        s.snow.density[CryoLayerID.NEW_SNOW, pos_accum] = (
            (
                s.snow.ice_content[CryoLayerID.NEW_SNOW, pos_accum]
                + s.snow.liquid_water_content[CryoLayerID.NEW_SNOW, pos_accum]
            ) / s.snow.thickness[CryoLayerID.NEW_SNOW, pos_accum]
        )

    def heat_conduction(self):
        pass

    def melt(self):
        model = self.model
        s = model.state

        ice_content_change = s.snow.melt.copy()

        for i in range(model.snow.num_layers):
            pos = (ice_content_change > 0) & (s.snow.ice_content[i, :] > 0)

            cur_ice_content_change = np.minimum(
                ice_content_change[pos],
                s.snow.ice_content[i, pos],
            )
            s.snow.thickness[i, pos] *= (1 - cur_ice_content_change / s.snow.ice_content[i, pos])
            s.snow.ice_content[i, pos] -= cur_ice_content_change
            s.snow.liquid_water_content[i, pos] += cur_ice_content_change
            ice_content_change[pos] -= cur_ice_content_change

    def sublimation(self):
        model = self.model
        s = model.state

        # Ice content change is positive for sublimation (mass loss) and negative for mass gain
        # (resublimation)
        ice_content_change = s.snow.sublimation.copy()

        for i in range(model.snow.num_layers):
            pos = (ice_content_change > 0) & (s.snow.ice_content[i, :] > 0)

            cur_ice_content_change = np.minimum(
                ice_content_change[pos],
                s.snow.ice_content[i, pos],
            )
            s.snow.thickness[i, pos] *= (1 - cur_ice_content_change / s.snow.ice_content[i, pos])
            s.snow.ice_content[i, pos] -= cur_ice_content_change
            ice_content_change[pos] -= cur_ice_content_change

    def runoff(self):
        model = self.model
        s = model.state

        max_lwc = max_liquid_water_content(model)
        max_lwc[2:, :] = 0.  # no LWC for firn and ice

        runoff = model.state.meteo.rainfall.copy()
        runoff[np.isnan(runoff)] = 0.

        for i in range(model.snow.num_layers):
            pos = (s.snow.ice_content[i, :] + s.snow.liquid_water_content[i, :]) > 0.

            s.snow.liquid_water_content[i, pos] += runoff[pos]
            runoff_cur = (s.snow.liquid_water_content[i, pos] - max_lwc[i, pos]).clip(min=0)
            runoff[pos] = runoff_cur
            s.snow.liquid_water_content[i, pos] -= runoff_cur

    def update_layers(self):
        model = self.model
        s = model.state.snow
        transition_params = model.config.snow.cryolayers.transition

        total_we = s.ice_content + s.liquid_water_content

        # Reset empty layers
        for i in range(self.num_layers):
            self.reset_layer(i, total_we[i, :] <= 0.)

        # Update thickness
        pos = total_we > 0.
        s.thickness[pos] = total_we[pos] / s.density[pos]

        # Transition new snow -> old snow
        self.layer_transition(
            CryoLayerID.NEW_SNOW,
            CryoLayerID.OLD_SNOW,
            s.density[CryoLayerID.NEW_SNOW, :] >= transition_params.old_snow
        )

        # Transition old snow -> firn at the first timestep of the "transition month"
        if (
            model.date.month == transition_params.firn
            and model.date_idx > 0
            and model.dates[model.date_idx - 1].month != transition_params.firn
        ):
            self.layer_transition(CryoLayerID.OLD_SNOW, CryoLayerID.FIRN)

        # Transition firn -> ice
        self.layer_transition(
            CryoLayerID.FIRN,
            CryoLayerID.ICE,
            s.density[CryoLayerID.FIRN, :] >= transition_params.ice,
        )

    def update_properties(self):
        model = self.model
        s = model.state

        snow_properties(self.model)

        s.snow.albedo[:] = np.nan

        for i in reversed(range(self.num_layers)):
            pos = s.snow.thickness[i, :] > 0
            s.snow.albedo[pos] = s.snow.layer_albedo[i, pos]

    def reset_layer(self, layer, pos=None):
        s = self.model.state.snow

        if pos is None:
            pos = slice(None)

        s.thickness[layer, pos] = 0
        s.ice_content[layer, pos] = 0
        s.liquid_water_content[layer, pos] = 0
        s.cold_content[layer, pos] = 0
        s.density[layer, pos] = np.nan
        s.layer_albedo[layer, pos] = np.nan

    def layer_transition(self, src_layer, dst_layer, pos=None):
        s = self.model.state.snow

        if pos is None:
            pos = self.model.grid.roi

        pos_merge = pos & (s.thickness[dst_layer, :] > 0.)
        pos_init = pos & (s.thickness[dst_layer, :] == 0.)

        # Initialize the destination layer for pixels where SWE = 0
        s.thickness[dst_layer, pos_init] = s.thickness[src_layer, pos_init]
        s.ice_content[dst_layer, pos_init] = s.ice_content[src_layer, pos_init]
        s.liquid_water_content[dst_layer, pos_init] = s.liquid_water_content[src_layer, pos_init]
        s.cold_content[dst_layer, pos_init] = s.cold_content[src_layer, pos_init]
        s.density[dst_layer, pos_init] = s.density[src_layer, pos_init]
        s.layer_albedo[dst_layer, pos_init] = s.layer_albedo[src_layer, pos_init]

        # Take weighted mean for pixels where destination layer already exists
        w1 = s.ice_content[src_layer, pos_merge] + s.liquid_water_content[src_layer, pos_merge]
        w2 = s.ice_content[dst_layer, pos_merge] + s.liquid_water_content[dst_layer, pos_merge]

        # Normalize weights
        sum_weights = w1 + w2
        w1 /= sum_weights
        w2 /= sum_weights

        s.ice_content[dst_layer, pos_merge] += s.ice_content[src_layer, pos_merge]
        s.liquid_water_content[dst_layer, pos_merge] += s.liquid_water_content[src_layer, pos_merge]
        s.cold_content[dst_layer, pos_merge] += s.cold_content[src_layer, pos_merge]
        s.density[dst_layer, pos_merge] = (
            w1 * s.density[src_layer, pos_merge]
            + w2 * s.density[dst_layer, pos_merge]
        )
        s.layer_albedo[dst_layer, pos_merge] = (
            w1 * s.layer_albedo[src_layer, pos_merge]
            + w2 * s.layer_albedo[dst_layer, pos_merge]
        )
        s.thickness[dst_layer, pos_merge] = (
            (s.ice_content[dst_layer, pos_merge] + s.liquid_water_content[dst_layer, pos_merge])
            / s.density[dst_layer, pos_merge]
        )

        self.reset_layer(src_layer, pos)

    def update_surface_layer_type(self):
        model = self.model
        roi = model.grid.roi
        s = model.state

        s.surface.layer_type[roi] = CryoLayerID.SNOW_FREE

        for i in reversed(range(model.snow.num_layers)):
            s.surface.layer_type[s.snow.thickness[i, :] > 0] = i
