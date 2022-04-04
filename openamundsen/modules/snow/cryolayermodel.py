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
        num_cryo_layers = 4

        s.add_variable('num_layers', '1', 'Number of snow layers', dtype=int, retain=True)
        s.add_variable('thickness', 'm', 'Snow thickness', dim3=num_cryo_layers, retain=True)
        s.add_variable('density', 'kg m-3', 'Snow density', 'snow_density', dim3=num_cryo_layers, retain=True)
        s.add_variable('ice_content', 'kg m-2', 'Ice content of snow', dim3=num_cryo_layers, retain=True)
        s.add_variable('liquid_water_content', 'kg m-2', 'Liquid water content of snow', 'liquid_water_content_of_snow_layer', dim3=num_cryo_layers, retain=True)
        s.add_variable('cold_content', 'kg m-2', 'Cold content of snow', dim3=num_cryo_layers, retain=True)
        s.add_variable('temp', 'K', 'Snow temperature', dim3=num_cryo_layers, retain=True)  # TODO remove this (only added because this is in the default point outputs)
        s.add_variable('layer_albedo', '1', 'Snow layer albedo', dim3=num_cryo_layers, retain=True)
        s.add_variable('heat_cap', 'J K-1 m-2', 'Areal heat capacity of snow', dim3=num_cryo_layers)  # TODO remove this (only added because of snow.snow_properties())

        self.model = model
        self.num_cryo_layers = num_cryo_layers

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

        pos_old = s.snow.thickness[CryoLayerID.OLD_SNOW, :] > 0.
        pos_new = s.snow.thickness[CryoLayerID.NEW_SNOW, :] > 0.
        s.snow.albedo[pos_old] = s.snow.layer_albedo[CryoLayerID.OLD_SNOW, pos_old]
        s.snow.albedo[pos_new] = s.snow.layer_albedo[CryoLayerID.NEW_SNOW, pos_new]
        albedo(model, pos_old | pos_new)

        if model.config.snow.cryolayers.use_single_snow_albedo:
            s.snow.layer_albedo[CryoLayerID.OLD_SNOW, pos_old] = s.snow.albedo[pos_old]
            s.snow.layer_albedo[CryoLayerID.NEW_SNOW, pos_new] = s.snow.albedo[pos_new]
        else:
            pos_old_surf = s.surface.layer_type == CryoLayerID.OLD_SNOW
            pos_new_surf = s.surface.layer_type == CryoLayerID.NEW_SNOW
            s.snow.layer_albedo[CryoLayerID.OLD_SNOW, pos_old_surf] = s.snow.albedo[pos_old_surf]
            s.snow.layer_albedo[CryoLayerID.NEW_SNOW, pos_new_surf] = s.snow.albedo[pos_new_surf]

        # Firn and ice albedo stay constant
        if self.num_cryo_layers > 2:
            for i in (CryoLayerID.FIRN, CryoLayerID.ICE):
                pos = s.surface.layer_type == i

                if i == CryoLayerID.FIRN:
                    s.snow.layer_albedo[i, pos] = model.config.snow.albedo.firn
                elif i == CryoLayerID.ICE:
                    s.snow.layer_albedo[i, pos] = model.config.snow.albedo.ice

    def compaction(self):
        model = self.model
        s = model.state

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
                s.snow.thickness,
                s.snow.ice_content,
                s.snow.liquid_water_content,
                s.snow.density,
                model.state.meteo.temp,
            )
        else:
            raise NotImplementedError

        # Now handle firn and ice
        if self.num_cryo_layers > 2:
            ice_density = model.config.snow.cryolayers.transition.ice

            # Firn: linear transition to ice in ~10 yrs
            firnies = s.snow.thickness[CryoLayerID.FIRN, :] > 0.
            densification_rate = (
                (ice_density - 500.)  # old snow->firn transition density is ~500 kg m-3
                / (constants.HOURS_PER_DAY * constants.DAYS_PER_YEAR * 10)  # 10 years
            )
            s.snow.density[CryoLayerID.FIRN, firnies] += densification_rate * (model.timestep / constants.SECONDS_PER_HOUR)

            # Ice: density stays constant
            icies = s.snow.thickness[CryoLayerID.ICE, :] > 0.
            s.snow.density[CryoLayerID.ICE, icies] = ice_density

            total_we = s.snow.ice_content + s.snow.liquid_water_content
            s.snow.thickness[CryoLayerID.FIRN, firnies] = (
                total_we[CryoLayerID.FIRN, firnies] / s.snow.density[CryoLayerID.FIRN, firnies]
            )
            s.snow.thickness[CryoLayerID.ICE, icies] = (
                total_we[CryoLayerID.ICE, icies] / s.snow.density[CryoLayerID.ICE, icies]
            )

    def accumulation(self):
        model = self.model
        s = model.state
        pos = s.meteo.snowfall > 0
        self.add_snow(
            pos,
            s.meteo.snowfall[pos],
            density=fresh_snow_density(s.meteo.wet_bulb_temp[pos]),
        )

    def heat_conduction(self):
        pass

    def temperature_index(self):
        """
        Calculate snowmelt based on a temperature index approach.
        """
        model = self.model
        melt_config = model.config.snow.melt
        s = model.state

        timestep_d = model.timestep / (constants.SECONDS_PER_HOUR * constants.HOURS_PER_DAY)  # (d)

        snowies = s.snow.thickness.sum(axis=0) > 0.
        degree_days = (s.meteo.temp[snowies] - melt_config.threshold_temp) * timestep_d

        if melt_config.method == 'temperature_index':
            melt = melt_config.degree_day_factor * degree_days.clip(min=0)
        elif melt_config.method == 'enhanced_temperature_index':
            melt = (
                melt_config.degree_day_factor * degree_days
                + melt_config.albedo_factor
                * (1. - s.surface.albedo[snowies])
                * s.meteo.sw_in[snowies]
                * timestep_d
            )
            melt[degree_days <= 0.] = 0.
        else:
            raise NotImplementedError

        s.snow.melt[model.grid.roi] = 0.
        s.snow.melt[snowies] = np.minimum(
            melt.clip(min=0),
            s.snow.ice_content[:, snowies].sum(axis=0),
        )

    def melt(self):
        model = self.model
        s = model.state

        ice_content_change = s.snow.melt.copy()

        for i in range(self.num_cryo_layers):
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
        roi = model.grid.roi
        s = model.state
        snowies_roi = s.snow.swe[roi] > 0.
        snowies = model.roi_mask_to_global(snowies_roi)

        s.snow.sublimation[roi] = 0.
        pot_sublim = -1 * s.surface.moisture_flux[snowies] * model.timestep
        pot_sublim[np.isnan(pot_sublim)] = 0.
        s.snow.sublimation[snowies] = np.minimum(pot_sublim, s.snow.ice_content[:, snowies].sum(axis=0))

        # Ice content change is positive for sublimation (mass loss) and negative for mass gain
        # (resublimation)
        ice_content_change = s.snow.sublimation.copy()

        # Sublimation: remove snow from the top layer
        for i in range(self.num_cryo_layers):
            pos = (ice_content_change > 0) & (s.snow.ice_content[i, :] > 0)

            cur_ice_content_change = np.minimum(
                ice_content_change[pos],
                s.snow.ice_content[i, pos],
            )
            s.snow.thickness[i, pos] *= (1 - cur_ice_content_change / s.snow.ice_content[i, pos])
            s.snow.ice_content[i, pos] -= cur_ice_content_change
            ice_content_change[pos] -= cur_ice_content_change

        # Resublimation: add snow to the top layer
        pos = (ice_content_change < 0)
        self.add_snow(
            pos,
            -ice_content_change[pos],
            density=fresh_snow_density(s.meteo.wet_bulb_temp[pos]),
        )

    def runoff(self):
        model = self.model
        s = model.state

        max_lwc = max_liquid_water_content(model)
        max_lwc[2:, :] = 0.  # no LWC for firn and ice

        runoff = model.state.meteo.rainfall.copy()
        runoff[np.isnan(runoff)] = 0.

        for i in range(self.num_cryo_layers):
            pos = (s.snow.ice_content[i, :] + s.snow.liquid_water_content[i, :]) > 0.

            s.snow.liquid_water_content[i, pos] += runoff[pos]
            runoff_cur = (s.snow.liquid_water_content[i, pos] - max_lwc[i, pos]).clip(min=0)
            runoff[pos] = runoff_cur
            s.snow.liquid_water_content[i, pos] -= runoff_cur

        s.snow.runoff[:] = runoff

    def update_layers(self):
        model = self.model
        s = model.state.snow
        transition_params = model.config.snow.cryolayers.transition

        total_we = s.ice_content + s.liquid_water_content

        # Reset empty layers
        for i in range(self.num_cryo_layers):
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

        # Update snow depth (might have changed due to density changes) and number of layers
        s.depth[:] = s.thickness.sum(axis=0)
        s.num_layers[:] = (s.thickness > 0).sum(axis=0)

    def update_properties(self):
        model = self.model
        s = model.state

        snow_properties(self.model)

        s.snow.albedo[:] = np.nan

        for i in reversed(range(self.num_cryo_layers)):
            pos = s.snow.thickness[i, :] > 0
            s.snow.albedo[pos] = s.snow.layer_albedo[i, pos]
            s.snow.density[i, pos] = (
                s.snow.ice_content[i, pos] + s.snow.liquid_water_content[i, pos]
            ) / s.snow.thickness[i, pos]

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

        pos_src_snow = pos & (s.thickness[src_layer, :] > 0.)
        pos_merge = pos_src_snow & (s.thickness[dst_layer, :] > 0.)
        pos_init = pos_src_snow & (s.thickness[dst_layer, :] == 0.)

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

        for i in reversed(range(self.num_cryo_layers)):
            s.surface.layer_type[s.snow.thickness[i, :] > 0] = i

    def add_snow(
            self,
            pos,
            ice_content,
            liquid_water_content=0,
            density=None,
            albedo=None,
    ):
        model = self.model
        s = model.state

        pos_init = (s.snow.ice_content[CryoLayerID.NEW_SNOW, pos] == 0) & (ice_content > 0)
        pos_init_global = model.global_mask(pos_init, pos)

        # Initialize new snow layer where required
        if np.isscalar(density):
            density = np.full(ice_content.shape, density)
        s.snow.density[CryoLayerID.NEW_SNOW, pos_init_global] = density[pos_init]

        if albedo is None:
            s.snow.layer_albedo[CryoLayerID.NEW_SNOW, pos_init_global] = model.config.snow.albedo.max

            if model.config.snow.cryolayers.use_single_snow_albedo:
                pos_albedo = pos_init_global & (s.snow.thickness[CryoLayerID.OLD_SNOW, :] > 0.)
                s.snow.layer_albedo[CryoLayerID.NEW_SNOW, pos_albedo] = (
                    s.snow.layer_albedo[CryoLayerID.OLD_SNOW, pos_albedo]
                )

            s.snow.albedo[pos_init_global] = s.snow.layer_albedo[CryoLayerID.NEW_SNOW, pos_init_global]
        else:
            s.snow.layer_albedo[CryoLayerID.NEW_SNOW, pos] = albedo
            s.snow.albedo[pos] = albedo

        # Add snow to new snow layer
        s.snow.ice_content[CryoLayerID.NEW_SNOW, pos] += ice_content
        s.snow.liquid_water_content[CryoLayerID.NEW_SNOW, pos] += liquid_water_content
        s.snow.thickness[CryoLayerID.NEW_SNOW, pos] += ice_content / density
        s.snow.density[CryoLayerID.NEW_SNOW, pos] = (
            (
                s.snow.ice_content[CryoLayerID.NEW_SNOW, pos]
                + s.snow.liquid_water_content[CryoLayerID.NEW_SNOW, pos]
            ) / s.snow.thickness[CryoLayerID.NEW_SNOW, pos]
        )
