from openamundsen import constants as c
import numpy as np


class CanopyModel:
    def __init__(self, model):
        self.model = model
        num_timesteps_per_day = int(c.HOURS_PER_DAY * c.SECONDS_PER_HOUR / model.timestep)
        model.state.meteo.add_variable(
            'last_24h_temps',
            long_name='Air temperatures of the previous 24 hours',
            dim3=max(num_timesteps_per_day, 1),
        )
        self._temp_idx = 0  # current index within the last_24h_temps array

    def initialize(self):
        model = self.model
        self.forest_pos = np.logical_or.reduce(
            [model.land_cover.class_pixels[lcc] for lcc in model.land_cover.forest_classes]
        )

    def meteo_modification(self):
        model = self.model
        s = model.state
        pos = self.forest_pos
        lai_eff = s.land_cover.lai_eff[pos]

        canopy_frac = (0.55 + 0.29 * np.log(lai_eff)).clip(0, 1)  # eq. (2)

        s.meteo.sw_in[pos] *= np.exp(-model.config.canopy.extinction_coefficient * lai_eff)  # eq. (1)

        s.meteo.last_24h_temps[self._temp_idx, :, :] = s.meteo.temp
        self._temp_idx += 1
        if self._temp_idx == s.meteo.last_24h_temps.shape[0]:
            self._temp_idx = 0

        mean_24h_temp = np.nanmean(s.meteo.last_24h_temps, axis=0)[pos]
        delta_t = ((mean_24h_temp - c.T0) / 3).clip(-2, 2)  # eq. (5)
        s.meteo.temp[pos] -= (  # eq. (4)
            canopy_frac
            * (
                s.meteo.temp[pos]
                - (
                    model.config.canopy.temperature_scaling_coefficient
                    * (s.meteo.temp[pos] - mean_24h_temp)
                    + mean_24h_temp
                    - delta_t
                )
            )
        )

        s.meteo.lw_in[pos] = (  # eq. (3)
            (1 - canopy_frac) * s.meteo.lw_in[pos]
            + canopy_frac * c.STEFAN_BOLTZMANN * s.meteo.temp[pos]**4
        )

        s.meteo.rel_hum[pos] *= (1 + 0.1 * canopy_frac)  # eq. (6)

        # Wind speed is modified assuming a reference level of z = 0.6h, i.e., the term (1 - z/h)
        # from Liston & Elder (2006, eq. (16)) becomes 0.4
        canopy_flow_index = model.config.canopy.canopy_flow_index_coefficient * lai_eff  # eq. (8)
        s.meteo.wind_speed[pos] *= np.exp(-0.4 * canopy_flow_index)  # eq. (7)
