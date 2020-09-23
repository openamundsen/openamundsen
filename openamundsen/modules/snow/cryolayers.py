import numpy as np
from openamundsen import constants
from openamundsen.snowmodel import SnowModel
from .snow import _compaction_anderson, _fresh_snow_density, albedo, snow_properties


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
        s.add_variable('areal_heat_cap', 'J K-1 m-2', 'Areal heat capacity of snow', dim3=num_layers)  # XXX only added because of snow.snow_properties()

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
        roi = model.grid.roi
        s = model.state

        for i in range(model.snow.num_layers):
            pos = model.roi_mask_to_global(s.surface.layer_type[roi] == i)

            if i in (CryoLayerID.NEW_SNOW, CryoLayerID.OLD_SNOW):
                albedo(model, pos)
                s.snow.layer_albedo[i, pos] = s.snow.albedo[pos]
            elif i == CryoLayerID.FIRN:
                s.snow.layer_albedo[i, pos] = model.config.snow.albedo.firn
            elif i == CryoLayerID.ICE:
                s.snow.layer_albedo[i, pos] = model.config.snow.albedo.ice

    def compaction(self):
        model = self.model
        snow = model.state.snow

        num_layers = (snow.thickness[:2] > 0).sum(axis=0)  # consider only new snow and old snow

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
        roi = model.grid.roi
        s = model.state

        density = _fresh_snow_density(s.meteo.wetbulb_temp[roi])
        ice_content_change = s.meteo.snowfall[roi] * model.timestep

        pos_accum_roi = ice_content_change > 0
        pos_init_layer_roi = (s.snow.ice_content[0, roi] == 0) & (ice_content_change > 0)
        pos_accum = model.roi_mask_to_global(pos_accum_roi)
        pos_init_layer = model.roi_mask_to_global(pos_init_layer_roi)

        # Initialize new snow layer where required
        s.snow.layer_albedo[CryoLayerID.NEW_SNOW, pos_init_layer] = model.config.snow.albedo.max
        s.snow.density[CryoLayerID.NEW_SNOW, pos_init_layer] = density[pos_init_layer_roi]

        # Add snow to new snow layer
        s.snow.ice_content[CryoLayerID.NEW_SNOW, pos_accum] += ice_content_change[pos_accum_roi]
        s.snow.thickness[CryoLayerID.NEW_SNOW, pos_accum] += ice_content_change[pos_accum_roi] / density[pos_accum_roi]
        s.snow.density[CryoLayerID.NEW_SNOW, pos_accum] = (
            (s.snow.ice_content[CryoLayerID.NEW_SNOW, pos_accum] + s.snow.liquid_water_content[CryoLayerID.NEW_SNOW, pos_accum])
            / s.snow.thickness[CryoLayerID.NEW_SNOW, pos_accum]
        )

    def heat_conduction(self):
        pass

    def melt(self):
        model = self.model
        roi = model.grid.roi
        s = model.state

        ice_content_change = s.snow.melt[roi].copy()

        for i in range(model.snow.num_layers):
            pos_roi = (ice_content_change > 0) & (s.snow.ice_content[i, roi] > 0)
            pos = model.roi_mask_to_global(pos_roi)

            cur_ice_content_change = np.minimum(ice_content_change[pos_roi], s.snow.ice_content[i, pos])
            s.snow.thickness[i, pos] *= (1 - cur_ice_content_change / s.snow.ice_content[i, pos])
            s.snow.ice_content[i, pos] -= cur_ice_content_change
            s.snow.liquid_water_content[i, pos] += cur_ice_content_change
            ice_content_change[pos_roi] -= cur_ice_content_change

    def sublimation(self):
        pass

    def runoff(self):
        model = self.model
        roi = model.grid.roi
        s = model.state

        max_lwc_frac = 0.05  # XXX

        runoff = model.state.meteo.rainfall_amount[roi].copy()

        for i in range(model.snow.num_layers):
            pos_roi = s.snow.thickness[i, roi] > 0
            pos = model.roi_mask_to_global(pos_roi)

            max_lwc = max_lwc_frac * s.snow.ice_content[i, pos]
            s.snow.liquid_water_content[i, pos] += runoff[pos_roi]
            runoff_cur = (s.snow.liquid_water_content[i, pos] - max_lwc).clip(min=0)
            runoff[pos_roi] = runoff_cur
            s.snow.liquid_water_content[i, pos] -= runoff_cur

    def update_layers(self):
        model = self.model
        roi = model.grid.roi
        s = model.state.snow
        transition_params = model.config.snow.cryolayers.transition

        # Transition new snow -> old snow
        self.layer_transition(
            CryoLayerID.NEW_SNOW,
            CryoLayerID.OLD_SNOW,
            model.roi_mask_to_global(s.density[CryoLayerID.NEW_SNOW, roi] >= transition_params.old_snow),
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
            model.roi_mask_to_global(s.density[CryoLayerID.FIRN, roi] >= transition_params.ice),
        )

    def update_properties(self):
        model = self.model
        roi = model.grid.roi
        s = model.state

        snow_properties(self.model)

        s.snow.albedo[roi] = np.nan

        for i in reversed(range(self.num_layers)):
            pos = model.roi_mask_to_global(s.snow.thickness[i, roi] > 0)
            s.snow.albedo[pos] = s.snow.layer_albedo[i, pos]

    def reset_layer(self, layer, pos=None):
        s = self.model.state.snow

        if pos is None:
            pos = self.model.grid.roi

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

        pos_existing = pos & (s.thickness[dst_layer, :] > 0.)
        pos_new = pos & (s.thickness[dst_layer, :] == 0.)

        # Initialize the destination layer for pixels where SWE = 0
        s.thickness[dst_layer, pos_new] = s.thickness[src_layer, pos_new]
        s.ice_content[dst_layer, pos_new] = s.ice_content[src_layer, pos_new]
        s.liquid_water_content[dst_layer, pos_new] = s.liquid_water_content[src_layer, pos_new]
        s.cold_content[dst_layer, pos_new] = s.cold_content[src_layer, pos_new]
        s.density[dst_layer, pos_new] = s.density[src_layer, pos_new]
        s.layer_albedo[dst_layer, pos_new] = s.layer_albedo[src_layer, pos_new]

        # Take weighted mean for pixels where destination layer already exists
        w1 = s.ice_content[src_layer, pos_existing] + s.liquid_water_content[src_layer, pos_existing]
        w2 = s.ice_content[dst_layer, pos_existing] + s.liquid_water_content[dst_layer, pos_existing]

        # Normalize weights
        sum_weights = w1 + w2
        w1 /= sum_weights
        w2 /= sum_weights

        s.ice_content[dst_layer, pos_existing] += s.ice_content[src_layer, pos_existing]
        s.liquid_water_content[dst_layer, pos_existing] += s.liquid_water_content[src_layer, pos_existing]
        s.cold_content[dst_layer, pos_existing] += s.cold_content[src_layer, pos_existing]
        s.density[dst_layer, pos_existing] = w1 * s.density[src_layer, pos_existing] + w2 * s.density[dst_layer, pos_existing]
        s.layer_albedo[dst_layer, pos_existing] = w1 * s.layer_albedo[src_layer, pos_existing] + w2 * s.layer_albedo[dst_layer, pos_existing]
        s.thickness[dst_layer, pos_existing] = (
            (s.ice_content[dst_layer, pos_existing] + s.liquid_water_content[dst_layer, pos_existing])
            / s.density[dst_layer, pos_existing]
        )

        self.reset_layer(src_layer, pos)
