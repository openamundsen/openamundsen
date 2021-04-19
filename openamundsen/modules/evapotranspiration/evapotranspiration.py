from openamundsen import constants as c, meteo
from .landcover import LandCoverID
import numpy as np


DEFAULT_CROP_COEFFICIENT_TYPES = {
    LandCoverID.CONIFEROUS_FOREST: 'dual',
    LandCoverID.DECIDUOUS_FOREST: 'dual',
}

DEFAULT_CROP_COEFFICIENTS = {  # (ini, mid, end)
    LandCoverID.CONIFEROUS_FOREST: (0.95, 0.95, 0.95),
    LandCoverID.DECIDUOUS_FOREST: (0.4, 1.05, 0.6),
}

DEFAULT_GROWTH_STAGE_LENGTHS = {  # (plant date (DOY), ini, dev, mid, late)
    LandCoverID.CONIFEROUS_FOREST: (1, 366, 0, 0),
    LandCoverID.DECIDUOUS_FOREST: (60, 20, 70, 120, 60),
}

DEFAULT_MAX_PLANT_HEIGHTS = {  # maximum plant heights (m), see Table 12 in Allen et al. (1998)
    LandCoverID.CONIFEROUS_FOREST: 26.,  # derived from data for Berchtesgaden National Park, default value from FAO is 20 m
    LandCoverID.DECIDUOUS_FOREST: 24.8,  # derived from data for Berchtesgaden National Park, default value from FAO is 14 m
}


class EvapotranspirationModel:
    """
    Evapotranspiration model following [1].

    References
    ----------
    .. [1] Allen, R.G., Pereira, L.S., Raes, D., et al. (1998). Crop
       Evapotranspiration-Guidelines for Computing Crop Water Requirements-FAO
       Irrigation and Drainage Paper 56. FAO, Rome, 300(9): D05109.
       http://www.fao.org/3/x0490e/x0490e00.htm
    """

    def __init__(self, model):
        self.model = model

        s = model.state.add_category('evapotranspiration')
        s.add_variable('et_ref', 'kg m-2', 'Reference evapotranspiration')
        s.add_variable('soil_heat_flux', 'W m-2', 'Soil heat flux beneath the grass reference surface')
        s.add_variable('basal_crop_coeff', '1', 'Basal crop coefficient')
        s.add_variable('evaporation_coeff', '1', 'Evaporation coefficient')
        s.add_variable('clim_corr', '1', 'Climate correction term')

    def initialize(self):
        model = self.model
        roi = model.grid.roi
        s = self.model.state

        # Prepare unique land cover classes occurring in the model domain and their associated pixel
        # locations
        lccs = np.unique(s.base.land_cover[roi])
        lccs = lccs[lccs > 0]
        self.land_cover_class_pixels = {}
        for lcc in lccs:
            self.land_cover_class_pixels[lcc] = s.base.land_cover == lcc

        self._climate_correction()

    def _climate_correction(self):
        """
        Calculate climate correction term for the crop coefficients (eq. (70) from [1]).
        """
        model = self.model
        s = self.model.state
        s_et = s.evapotranspiration

        # Clip WS/RH values to allowed ranges according to [1]
        mean_wind_speed = np.clip(model.config.evapotranspiration.mean_wind_speed, 1, 6)
        mean_min_rel_hum = np.clip(model.config.evapotranspiration.mean_min_humidity, 20, 80)

        for lcc, pos in self.land_cover_class_pixels.items():
            h = DEFAULT_MAX_PLANT_HEIGHTS.get(lcc, np.nan)
            s_et.clim_corr[pos] = (  # eq. (70)
                (0.04 * (mean_wind_speed - 2) - 0.004 * (mean_min_rel_hum - 45))
                * (h / 3)**0.3
            )

    def evapotranspiration(self):
        self.reference_evapotranspiration()
        self.crop_coefficient()

    def reference_evapotranspiration(self):
        """
        Calculate reference evapotranspiration (ETo).
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

    def crop_coefficient(self):
        """
        Calculate the basal crop coefficient K_cb depending on the day of the growing period.
        """
        model = self.model
        doy = model.date.dayofyear
        s = model.state
        s_et = s.evapotranspiration

        for lcc, pos in self.land_cover_class_pixels.items():
            crop_coefficient_type = DEFAULT_CROP_COEFFICIENT_TYPES[lcc]
            plant_date, length_ini, length_dev, length_mid, length_late = DEFAULT_GROWTH_STAGE_LENGTHS[lcc]
            crop_coeff_ini, crop_coeff_mid, crop_coeff_end = DEFAULT_CROP_COEFFICIENTS[lcc]
            growing_period_day = doy - plant_date + 1  # (1-based)

            # Apply climate correction for Kcb_mid and Kcb_end values >= 0.45 (eq. (70))
            # (convert crop_coeff_mid and crop_coeff_end into fields to allow for possibly
            # non-uniform climate correction values)
            if crop_coeff_mid >= 0.45:
                crop_coeff_mid = np.full(pos.sum(), crop_coeff_mid) + s_et.clim_corr[pos]
            if crop_coeff_end >= 0.45:
                crop_coeff_end = np.full(pos.sum(), crop_coeff_end) + s_et.clim_corr[pos]

            if crop_coefficient_type == 'single':
                raise NotImplementedError
            elif crop_coefficient_type == 'dual':
                s_et.basal_crop_coeff[pos] = basal_crop_coefficient(
                    growing_period_day,
                    length_ini,
                    length_dev,
                    length_mid,
                    length_late,
                    crop_coeff_ini,
                    crop_coeff_mid,
                    crop_coeff_end,
                )
            else:
                raise NotImplementedError


def basal_crop_coefficient(
    growing_period_day,
    length_ini,
    length_dev,
    length_mid,
    length_late,
    crop_coeff_ini,
    crop_coeff_mid,
    crop_coeff_end,
):
    """
    Calculate the daily basal crop coefficient K_cb following [1].

    Parameters
    ----------
    growing_period_day : int
        Day within the growing period (1 = first day of the period).

    length_ini : int
        Length of the initial growth stage (days).

    length_dev : int
        Length of the crop development stage (days).

    length_mid : int
        Length of the mid-season stage (days).

    length_late : int
        Length of the late season stage (days).

    crop_coeff_ini : float
        Crop coefficient for the initial stage.

    crop_coeff_mid : float or ndarray(float)
        Crop coefficient for the mid-season stage.

    crop_coeff_end : float or ndarray(float)
        Crop coefficient for the end of the late season stage.

    Returns
    -------
    basal_crop_coefficient : float or ndarray(float)
        Basal crop coefficient for the given day.
        Depending on the data types of crop_coeff_mid and crop_coeff_end, this
        is either a scalar or an array.

    References
    ----------
    .. [1] Allen, R.G., Pereira, L.S., Raes, D., et al. (1998). Crop
       Evapotranspiration-Guidelines for Computing Crop Water Requirements-FAO
       Irrigation and Drainage Paper 56. FAO, Rome, 300(9): D05109.
       http://www.fao.org/3/x0490e/x0490e00.htm
    """
    total_length = length_ini + length_dev + length_mid + length_late

    if growing_period_day < 1 or growing_period_day > total_length:  # outside of growing period
        bcc = 0.
    elif growing_period_day < length_ini:  # initial
        bcc = crop_coeff_ini
    elif growing_period_day < (length_ini + length_dev):  # crop development
        bcc = (  # eq. (66)
            crop_coeff_ini
            + (growing_period_day - length_ini) / length_dev
            * (crop_coeff_mid - crop_coeff_ini)
        )
    elif growing_period_day < (length_ini + length_dev + length_mid):  # mid season
        bcc = crop_coeff_mid
    else:  # late season
        bcc = (  # eq. (66)
            crop_coeff_mid
            + (growing_period_day - (length_ini + length_dev + length_mid)) / length_late
            * (crop_coeff_end - crop_coeff_mid)
        )

    return bcc
