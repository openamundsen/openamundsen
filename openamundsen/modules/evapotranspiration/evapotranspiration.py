from openamundsen import constants as c, meteo
import numpy as np


class EvapotranspirationModel:
    def __init__(self, model):
        self.model = model

        s = model.state.add_category('evapotranspiration')
        s.add_variable('et_ref', 'kg m-2', 'Reference evapotranspiration')
        s.add_variable('soil_heat_flux', 'W m-2', 'Soil heat flux beneath the grass reference surface')

    def evapotranspiration(self):
        self.reference_evapotranspiration()

    def reference_evapotranspiration(self):
        """
        Calculate reference evapotranspiration (ETo) following [1].

        References
        ----------
        .. [1] Allen, R.G., Pereira, L.S., Raes, D., et al. (1998). Crop
           Evapotranspiration-Guidelines for Computing Crop Water Requirements-FAO
           Irrigation and Drainage Paper 56. FAO, Rome, 300(9): D05109.
           http://www.fao.org/3/x0490e/x0490e00.htm
        """
        model = self.model
        roi = model.grid.roi
        s = model.state
        s_et = s.evapotranspiration

        s_et.soil_heat_flux[roi] = 0.1 * s.meteo.net_radiation[roi]  # eq. (45)
        # XXX use 0.5 for nighttime periods (46)

        Wm2_to_MJm2h = 1e-6 * c.SECONDS_PER_HOUR  # conversion factor from W m-2 (= J m-2 s-1) to MJ m-2 h-1

        Rn = s.meteo.net_radiation[roi] * Wm2_to_MJm2h  # net radiation at the grass surface (MJ m-2 h-1)
        G = s_et.soil_heat_flux[roi] * Wm2_to_MJm2h  # soil heat flux density (MJ m-2 h-1)
        T = s.meteo.temp[roi] - c.T0  # air temperature (°C)
        D = 4098 * (0.6108 * np.exp(17.27 * T / (T + 273.3))) / (T + 273.3)**2  # slope of the relationship between saturation vapor pressure and temperature (kPa °C-1) (eq. (13))
        g = s.meteo.psych_const[roi] * 1e-3  # psychrometric constant (kPa °C-1)
        e = s.meteo.sat_vap_press[roi] * 1e-3  # saturation vapor pressure (kPa)
        ea = s.meteo.vap_press[roi] * 1e-3  # actual vapor pressure (kPa)

        grass_roughness_length = 0.03  # (m)
        u2 = meteo.log_wind_profile(  # 2 m wind speed (m s-1)
            s.meteo.wind_speed[roi],
            model.config.meteo.measurement_height.wind,
            2,
            grass_roughness_length,
        )

        ET0 = (  # reference evapotranspiration (kg m-2 h-1) (eq. (53))
            (0.408 * D * (Rn - G) + g * 37 / (T + 273) * u2 * (e - ea))
            / (D + g * (1 + 0.34 * u2))
        )
        ET0 = ET0.clip(min=0)  # do not allow negative values
        s_et.et_ref[roi] = ET0 * model.timestep / c.SECONDS_PER_HOUR  # (kg m-2)
