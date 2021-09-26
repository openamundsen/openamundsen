from openamundsen import constants as c, meteo
import numpy as np
import warnings


class CanopyModel:
    """
    Canopy model for adjusting meteorology for inside-canopy conditions and
    calculating snow-canopy interactions following [1].

    References
    ----------
    .. [1] Strasser, U., Warscher, M., & Liston, G. E. (2011). Modeling
       Snow-Canopy Processes on an Idealized Mountain. Journal of Hydrometeorology,
       12(4), 663–677.  https://doi.org/10.1175/2011JHM1344.1
    """

    def __init__(self, model):
        self.model = model
        num_timesteps_per_day = int(c.HOURS_PER_DAY * c.SECONDS_PER_HOUR / model.timestep)
        model.state.meteo.add_variable(
            'last_24h_temps',
            'K',
            'Air temperatures of the previous 24 hours',
            dim3=max(num_timesteps_per_day, 1),
            retain=True,
        )
        self._temp_idx = 0  # current index within the last_24h_temps array

        model.state.snow.add_variable(
            'canopy_intercepted_load',
            'kg m-2',
            'Canopy snow interception storage',
            retain=True,
        )
        model.state.snow.add_variable(
            'canopy_intercepted_snowfall',
            'kg m-2',
            'Canopy-intercepted snowfall',
        )
        model.state.snow.add_variable(
            'canopy_sublimation',
            'kg m-2',
            'Canopy snow sublimation',
        )
        model.state.snow.add_variable(
            'canopy_melt',
            'kg m-2',
            'Melt of canopy-intercepted snow',
        )

    def initialize(self):
        model = self.model

        if len(model.land_cover.forest_classes) > 0:
            self.forest_pos = np.logical_or.reduce(
                [model.land_cover.class_pixels[lcc] for lcc in model.land_cover.forest_classes]
            )
        else:
            self.forest_pos = np.full(model.grid.shape, False)

        model.state.snow.canopy_intercepted_load[self.forest_pos] = 0.

    def meteorology(self):
        """
        Modify meteorology for inside-canopy conditions.
        """
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

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
            mean_24h_temp = np.nanmean(s.meteo.last_24h_temps, axis=0)

        pos_finite_temp = pos & np.isfinite(mean_24h_temp)
        delta_t = ((mean_24h_temp[pos_finite_temp] - c.T0) / 3).clip(-2, 2)  # eq. (5)
        s.meteo.temp[pos_finite_temp] -= (  # eq. (4)
            canopy_frac
            * (
                s.meteo.temp[pos_finite_temp]
                - (
                    model.config.canopy.temperature_scaling_coefficient
                    * (s.meteo.temp[pos_finite_temp] - mean_24h_temp[pos_finite_temp])
                    + mean_24h_temp[pos_finite_temp]
                    - delta_t
                )
            )
        )

        s.meteo.lw_in[pos] = (  # eq. (3)
            (1 - canopy_frac) * s.meteo.lw_in[pos]
            + canopy_frac * c.STEFAN_BOLTZMANN * s.meteo.temp[pos]**4
        )

        s.meteo.rel_hum[pos] = (  # eq. (6)
            s.meteo.rel_hum[pos] * (1 + 0.1 * canopy_frac)
        ).clip(0, 100)

        # Wind speed is modified assuming a reference level of z = 0.6h, i.e., the term (1 - z/h)
        # from Liston & Elder (2006, eq. (16)) becomes 0.4
        canopy_flow_index = model.config.canopy.canopy_flow_index_coefficient * lai_eff  # eq. (8)
        s.meteo.wind_speed[pos] *= np.exp(-0.4 * canopy_flow_index)  # eq. (7)

    def snow(self):
        """
        Calculate canopy snow processes (interception, sublimation and melt unload).
        """
        model = self.model
        s = model.state
        pos = self.forest_pos

        # Absorbed solar radiation by a snow particle in the canopy (W) (eq. (11))
        absorbed_rad = (
            np.pi
            * model.config.canopy.spherical_ice_particle_radius**2
            * (1 - s.surface.albedo[pos])
            * s.meteo.top_canopy_sw_in[pos]
        )

        reynolds = (  # Reynolds number (eq. (12))
            2 * model.config.canopy.spherical_ice_particle_radius
            * s.meteo.wind_speed[pos]
            / model.config.canopy.kinematic_air_viscosity
        ).clip(0.7, 10)
        nusselt = 1.79 + 0.606 * np.sqrt(reynolds)  # Nusselt number (eq. (13))
        sherwood = nusselt  # Sherwood number

        # Saturation density of water vapor (kg m-3) (eq. (15))
        wat_vap_sat_dens = 0.622 * meteo.absolute_humidity(
            s.meteo.temp[pos],
            s.meteo.sat_vap_press[pos],
        )

        # Diffusivity of water vapor in the atmosphere (m2 s-1) (eq. (16))
        diff_wat_vap = 2.06e-5 * (s.meteo.temp[pos] / 273.)**1.75

        omega = (  # eq. (18)
            1. / (c.THERM_COND_AIR * s.meteo.temp[pos] * nusselt)
            * (
                c.LATENT_HEAT_OF_SUBLIMATION
                * c.MOLAR_MASS_WATER
                / (c.UNIVERSAL_GAS_CONSTANT * s.meteo.temp[pos])
                - 1.
            )
        )

        # Mass loss rate from an ice sphere (kg s-1) (eq. (17))
        mass_loss_rate = (
            2 * np.pi * model.config.canopy.spherical_ice_particle_radius
            * (s.meteo.rel_hum[pos] / 100. - 1.)
            - absorbed_rad * omega
        ) / (
            c.LATENT_HEAT_OF_SUBLIMATION * omega
            + 1. / (diff_wat_vap * wat_vap_sat_dens * sherwood)
        )
        mass_loss_rate[np.isnan(mass_loss_rate)] = 0.

        # Particle mass (kg) (eq. (20))
        particle_mass = (
            4./3. * np.pi
            * c.ICE_DENSITY
            * model.config.canopy.spherical_ice_particle_radius**3
        )

        # Sublimation loss rate coefficient from an ice sphere (s-1) (eq. (19))
        sublim_loss_rate_coeff = mass_loss_rate / particle_mass

        # Update canopy-intercepted load (kg m-2) (eq. (20))
        max_interception_storage = (
            model.config.canopy.max_interception_storage_coefficient
            * s.land_cover.lai_eff[pos]
        )
        new_canopy_intercepted_load = (
            s.snow.canopy_intercepted_load[pos]
            + 0.7
            * (max_interception_storage - s.snow.canopy_intercepted_load[pos])
            * (1 - np.exp(-s.meteo.snowfall[pos] / max_interception_storage))
        )
        s.snow.canopy_intercepted_snowfall[pos] = (
            new_canopy_intercepted_load
            - s.snow.canopy_intercepted_load[pos]
        ).clip(min=0)
        s.snow.canopy_intercepted_load[pos] = new_canopy_intercepted_load

        # Canopy exposure coefficient (eq. (23))
        exposure_coeff = np.zeros(new_canopy_intercepted_load.shape)
        pos_int = new_canopy_intercepted_load > 0
        exposure_coeff[pos_int] = (
            model.config.canopy.exposure_coefficient_coefficient
            * (new_canopy_intercepted_load[pos_int] / max_interception_storage[pos_int])**(-0.4)
        )

        # Canopy sublimation (eq. (22))
        s.snow.canopy_sublimation[pos] = np.minimum(
            (
                -exposure_coeff
                * new_canopy_intercepted_load
                * sublim_loss_rate_coeff
                * model.timestep
            ),
            new_canopy_intercepted_load,
        ).clip(min=0)
        s.snow.canopy_intercepted_load[pos] -= s.snow.canopy_sublimation[pos]

        # Calculate melt unload using a temperature index method
        # Instead of applying an additional scaling factor to the default degree day factors for
        # snow on the ground as in [1], here a distinct degree day factor for canopy-intercepted
        # snow is used directly.
        s.snow.canopy_melt[pos] = np.minimum(
            (
                model.config.canopy.degree_day_factor
                * (s.meteo.temp[pos] - c.T0).clip(min=0)
                * model.timestep / (c.SECONDS_PER_HOUR * c.HOURS_PER_DAY)
            ),
            s.snow.canopy_intercepted_load[pos],
        )
        s.snow.canopy_intercepted_load[pos] -= s.snow.canopy_melt[pos]

        # Precipitation reaching the ground: reduce snowfall by canopy interception and increase by
        # melt unload
        s.meteo.snowfall[pos] -= s.snow.canopy_intercepted_snowfall[pos]
        s.meteo.snowfall[pos] += s.snow.canopy_melt[pos]
        s.meteo.precip[pos] = s.meteo.snowfall[pos] + s.meteo.rainfall[pos]


def above_canopy_meteorology(model):
    """
    Save above-canopy meteorological variables before they (potentially)
    are modified by the canopy model.
    """
    s_m = model.state.meteo
    roi = model.grid.roi

    s_m.top_canopy_temp[roi] = s_m.temp[roi]
    s_m.top_canopy_rel_hum[roi] = s_m.rel_hum[roi]
    s_m.top_canopy_wind_speed[roi] = s_m.wind_speed[roi]
    s_m.top_canopy_sw_in[roi] = s_m.sw_in[roi]
    s_m.top_canopy_lw_in[roi] = s_m.lw_in[roi]
