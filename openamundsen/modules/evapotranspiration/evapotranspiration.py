from openamundsen import constants as c, meteo
from .landcover import LandCoverClass
import numpy as np
from .soiltexture import SoilTextureClass


DEFAULT_CROP_COEFFICIENT_TYPES = {
    LandCoverClass.WATER: 'single',
    LandCoverClass.PASTURE: 'dual',
    LandCoverClass.CONIFEROUS_FOREST: 'dual',
    LandCoverClass.DECIDUOUS_FOREST: 'dual',
}
DEFAULT_CROP_COEFFICIENTS = {  # (ini, mid, end)
    LandCoverClass.WATER: (1.25, 0.65, 1.25),
    LandCoverClass.PASTURE: (0.30, 0.90, 0.80),
    LandCoverClass.CONIFEROUS_FOREST: (0.95, 0.95, 0.95),
    LandCoverClass.DECIDUOUS_FOREST: (0.4, 1.05, 0.6),
}
DEFAULT_GROWTH_STAGE_LENGTHS = {  # (plant date (DOY), ini, dev, mid, late)
    LandCoverClass.WATER: (1, 0, 183, 0, 183),  # TODO should be adjusted
    LandCoverClass.PASTURE: (60, 10, 20, 210, 30),  # TODO should be adjusted
    LandCoverClass.CONIFEROUS_FOREST: (1, 366, 0, 0, 0),
    LandCoverClass.DECIDUOUS_FOREST: (60, 20, 70, 120, 60),
}
DEFAULT_MAX_PLANT_HEIGHTS = {  # maximum plant heights (m), see Table 12 in Allen et al. (1998)
    LandCoverClass.WATER: 0.,
    LandCoverClass.PASTURE: (0.15 + 0.30) / 2.,
    LandCoverClass.CONIFEROUS_FOREST: 26.,  # derived from data for Berchtesgaden National Park, default value from FAO is 20 m
    LandCoverClass.DECIDUOUS_FOREST: 24.8,  # derived from data for Berchtesgaden National Park, default value from FAO is 14 m
}
DEFAULT_ROOTING_DEPTHS = {  # maximum effective rooting depths (m), from Table 22 in Allen et al. (1998)
    LandCoverClass.WATER: np.nan,
    LandCoverClass.PASTURE: 0.5,
    LandCoverClass.CONIFEROUS_FOREST: (1.0 + 1.5) / 2.,
    LandCoverClass.DECIDUOUS_FOREST: (1.7 + 2.4) / 2.,
}
DEFAULT_DEPLETION_FRACTIONS = {  # soil water depletion fractions for no stress, from Table 22 in Allen et al. (1998)
    LandCoverClass.WATER: np.nan,
    LandCoverClass.PASTURE: 0.6,
    LandCoverClass.CONIFEROUS_FOREST: 0.7,
    LandCoverClass.DECIDUOUS_FOREST: 0.5,
}

# Default soil water characteristics for different soil types (from Table 19 in Allen et al. (1998))
DEFAULT_SOIL_WATER_CONTENTS_AT_FIELD_CAPACITY = {  # m3 m-3
    SoilTextureClass.SAND: (0.07 + 0.17) / 2,
    SoilTextureClass.LOAMY_SAND: (0.11 + 0.19) / 2,
    SoilTextureClass.SANDY_LOAM: (0.18 + 0.28) / 2,
    SoilTextureClass.LOAM: (0.20 + 0.30) / 2,
    SoilTextureClass.SILT_LOAM: (0.22 + 0.36) / 2,
    SoilTextureClass.SILT: (0.28 + 0.36) / 2,
    SoilTextureClass.SILT_CLAY_LOAM: (0.30 + 0.37) / 2,
    SoilTextureClass.SILTY_CLAY: (0.30 + 0.42) / 2,
    SoilTextureClass.CLAY: (0.32 + 0.40) / 2,
}
DEFAULT_SOIL_WATER_CONTENTS_AT_WILTING_POINT = {  # m3 m-3
    SoilTextureClass.SAND: (0.02 + 0.07) / 2,
    SoilTextureClass.LOAMY_SAND: (0.03 + 0.10) / 2,
    SoilTextureClass.SANDY_LOAM: (0.06 + 0.16) / 2,
    SoilTextureClass.LOAM: (0.07 + 0.17) / 2,
    SoilTextureClass.SILT_LOAM: (0.09 + 0.21) / 2,
    SoilTextureClass.SILT: (0.12 + 0.22) / 2,
    SoilTextureClass.SILT_CLAY_LOAM: (0.17 + 0.24) / 2,
    SoilTextureClass.SILTY_CLAY: (0.17 + 0.29) / 2,
    SoilTextureClass.CLAY: (0.20 + 0.24) / 2,
}
DEFAULT_READILY_EVAPORABLE_WATER = {  # kg m-2
    SoilTextureClass.SAND: (2. + 7.) / 2,
    SoilTextureClass.LOAMY_SAND: (4. + 8.) / 2,
    SoilTextureClass.SANDY_LOAM: (6. + 10.) / 2,
    SoilTextureClass.LOAM: (8. + 10.) / 2,
    SoilTextureClass.SILT_LOAM: (8. + 11.) / 2,
    SoilTextureClass.SILT: (8. + 11.) / 2,
    SoilTextureClass.SILT_CLAY_LOAM: (8. + 11.) / 2,
    SoilTextureClass.SILTY_CLAY: (8. + 12.) / 2,
    SoilTextureClass.CLAY: (8. + 12.) / 2,
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
        s.add_variable('soil_texture', long_name='Soil texture class', dtype=int)  # TODO move to base or soil group eventually
        s.add_variable('evaporation', 'kg m-2', 'Evaporation')
        s.add_variable('transpiration', 'kg m-2', 'Transpiration')
        s.add_variable('evapotranspiration', 'kg m-2', 'Evapotranspiration')
        s.add_variable('et_ref', 'kg m-2', 'Reference evapotranspiration')
        s.add_variable('soil_heat_flux', 'W m-2', 'Soil heat flux beneath the grass reference surface')
        s.add_variable('crop_coeff', '1', 'Crop coefficient')
        s.add_variable('basal_crop_coeff', '1', 'Basal crop coefficient')
        s.add_variable('evaporation_coeff', '1', 'Evaporation coefficient')
        s.add_variable('water_stress_coeff', '1', 'Water stress coefficient')
        s.add_variable('clim_corr', '1', 'Climate correction term')
        s.add_variable('cum_soil_surface_depletion', 'kg m-2', 'Cumulative evaporation from the soil surface layer')
        s.add_variable('cum_root_zone_depletion', 'kg m-2', 'Cumulative evapotranspiration from the root zone')
        s.add_variable('total_evaporable_water', 'kg m-2', 'Total evaporable water')
        s.add_variable('total_available_water', 'kg m-2', 'Total available water')
        s.add_variable('readily_evaporable_water', 'kg m-2', 'Readily evaporable water')
        s.add_variable('readily_available_water', 'kg m-2', 'Readily available water')
        s.add_variable('deep_percolation', 'kg m-2', 'Deep percolation')
        s.add_variable('deep_percolation_evaporation_layer', 'kg m-2', 'Deep percolation from the evaporation layer')

    def initialize(self):
        model = self.model
        roi = model.grid.roi
        s = self.model.state
        s_et = s.evapotranspiration

        # Prepare unique land cover classes occurring in the model domain and their associated pixel
        # locations
        lccs = np.unique(s.base.land_cover[roi])
        lccs = lccs[lccs > 0]
        self.land_cover_class_pixels = {}
        for lcc in lccs:
            self.land_cover_class_pixels[lcc] = (s.base.land_cover == lcc) & roi

        self._climate_correction()

        # Calculate total evaporable water (eq. (73)), initialize readily evaporable water, and
        # begin calculation of total available water (eq. (82)) and initial root zone depletion (eq.
        # (87))
        stcs = np.unique(s_et.soil_texture[roi])
        stcs = stcs[stcs > 0]
        for stc in stcs:
            pos = s_et.soil_texture == stc
            swc_field_cap = DEFAULT_SOIL_WATER_CONTENTS_AT_FIELD_CAPACITY[stc]
            swc_wilting_point = DEFAULT_SOIL_WATER_CONTENTS_AT_WILTING_POINT[stc]

            s_et.total_evaporable_water[pos] = (
                1000
                * (swc_field_cap - 0.5 * swc_wilting_point)
                * model.config.evapotranspiration.surface_soil_layer_evaporation_depth
            )
            s_et.readily_evaporable_water[pos] = DEFAULT_READILY_EVAPORABLE_WATER[stc]

            # In the calculation of TAW the multiplication by the rooting depth is missing here (as
            # this is a land cover specific parameter and not soil specific) - follows below
            s_et.total_available_water[pos] = (
                1000
                * (swc_field_cap - swc_wilting_point)
                # * rooting_depth
            )

            # Same for the calculation of initial root zone depletion
            swc = swc_field_cap  # assume root zone is near field capacity following heavy rain or irrigation
            s_et.cum_root_zone_depletion[pos] = (  # eq. (87)
                1000
                * (swc_field_cap - swc)
                # * rooting_depth
            )

        # Finish calculation of TAW (eq. (82)) and root zone depletion (eq. (87)) and calculate
        # readily available water (eq. (83))
        # (depletion fractions are assumed constant; adjustment using ETc as suggested in [1] (p.
        # 162) is not performed here)
        for lcc, pos in self.land_cover_class_pixels.items():
            rooting_depth = DEFAULT_ROOTING_DEPTHS[lcc]
            depletion_fraction = DEFAULT_DEPLETION_FRACTIONS[lcc]
            s_et.total_available_water[pos] *= rooting_depth
            s_et.cum_root_zone_depletion[pos] *= rooting_depth
            s_et.readily_available_water[pos] = depletion_fraction * s_et.total_available_water[pos]
            # TODO modify depletion fractions depending on soil type (reduce by 5-10% for fine
            # textured soils and increase by 5-10% for coarse textured soils) ([1], p. 167)

        # Set D_e to 0 at the start of the model run, i.e., assume the topsoil is near field
        # capacity following a heavy rain or irrigation
        s_et.cum_soil_surface_depletion[roi] = 0.

        s_et.deep_percolation_evaporation_layer[roi] = 0.

    def _climate_correction(self):
        """
        Calculate climate correction term for the crop coefficients.
        """
        model = self.model
        s = self.model.state
        s_et = s.evapotranspiration

        # Clip WS/RH values to allowed ranges according to [1]
        mean_wind_speed = np.clip(model.config.evapotranspiration.mean_wind_speed, 1, 6)
        mean_min_rel_hum = np.clip(model.config.evapotranspiration.mean_min_humidity, 20, 80)

        for lcc, pos in self.land_cover_class_pixels.items():
            plant_height = DEFAULT_MAX_PLANT_HEIGHTS.get(lcc, np.nan)
            s_et.clim_corr[pos] = climate_correction(
                mean_wind_speed,
                mean_min_rel_hum,
                plant_height,
            )

    def evapotranspiration(self):
        model = self.model
        roi = model.grid.roi
        doy = model.date.dayofyear
        s = model.state
        s_et = s.evapotranspiration
        snowies_roi = s.snow.swe[roi] > 0.

        s_et.crop_coeff[roi] = np.nan
        s_et.basal_crop_coeff[roi] = np.nan
        s_et.evaporation_coeff[roi] = np.nan
        s_et.water_stress_coeff[roi] = np.nan

        self._reference_evapotranspiration()

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

            crop_coeff = crop_coefficient(
                growing_period_day,
                length_ini,
                length_dev,
                length_mid,
                length_late,
                crop_coeff_ini,
                crop_coeff_mid,
                crop_coeff_end,
                model.config.evapotranspiration.min_crop_coefficient,
            )
            
            # Derive global masks for pixels with the current land cover class which are
            # snow-covered and snow-free
            pos_snow = model.global_mask(pos[roi] & snowies_roi)
            pos_snowfree = model.global_mask(pos[roi] & (~snowies_roi))

            # Calculate crop ET under standard conditions
            if crop_coefficient_type == 'single':
                s_et.crop_coeff[pos] = crop_coeff
                self._single_coeff_crop_et(pos_snowfree)
                s_et.evapotranspiration[pos_snow] = 0.
            elif crop_coefficient_type == 'dual':
                s_et.basal_crop_coeff[pos] = crop_coeff
                self._dual_coeff_crop_et(pos_snowfree, lcc)
                s_et.evaporation[pos_snow] = 0.
                s_et.transpiration[pos_snow] = 0.
                s_et.evapotranspiration[pos_snow] = 0.
            else:
                raise NotImplementedError

            # Adjust ET for soil water stress conditions
            if lcc != LandCoverClass.WATER:
                self._water_stress_coefficient(pos_snowfree)
                if crop_coefficient_type == 'single':
                    s_et.evapotranspiration[pos_snowfree] *= s_et.water_stress_coeff[pos_snowfree]  # eq. (81)
                elif crop_coefficient_type == 'dual':
                    s_et.transpiration[pos_snowfree] *= s_et.water_stress_coeff[pos_snowfree]
                    s_et.evapotranspiration[pos_snowfree] = (
                        s_et.evaporation[pos_snowfree]
                        + s_et.transpiration[pos_snowfree]
                    )
                self._root_zone_water_balance(pos_snowfree)

    def _reference_evapotranspiration(self):
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

    def _single_coeff_crop_et(self, pos):
        s_et = self.model.state.evapotranspiration
        s_et.evapotranspiration[pos] = s_et.crop_coeff[pos] * s_et.et_ref[pos]

    def _dual_coeff_crop_et(self, pos, lcc):
        model = self.model
        s = model.state
        s_et = s.evapotranspiration

        plant_height = DEFAULT_MAX_PLANT_HEIGHTS[lcc]
        min_crop_coeff = model.config.evapotranspiration.min_crop_coefficient

        # Calculate K_c_max (eq. (72))
        # TODO the climate correction term is intended to be calculated using mean values of wind
        # speed and daily-minimum relative humidity over the period of interest - maybe better use
        # 24-hour moving averages than instantaneous values?
        clim_corr = climate_correction(
            s.meteo.wind_speed[pos],
            s.meteo.rel_hum[pos],
            plant_height,
        )
        max_crop_coeff = np.maximum(1.2 + clim_corr, s_et.basal_crop_coeff[pos] + 0.05)

        # Calculate fraction of the soil surface covered by vegetation (eq. (76))
        veg_frac = (
            (s_et.basal_crop_coeff[pos] - min_crop_coeff).clip(min=0.01)
            / (max_crop_coeff - min_crop_coeff)
        )**(1 + 0.5 * plant_height)

        # Fraction of soil surface wetted by irrigation or precipitation (use value for
        # precipitation (= 1.0) from Table 20)
        wetted_frac = 1.

        # Exposed and wetted soil fraction (eq. (75))
        exposed_wetted_frac = np.minimum(1 - veg_frac, wetted_frac)

        # Calculate evaporation reduction coefficient (eq. (74))
        pos2 = s_et.cum_soil_surface_depletion[pos] > s_et.readily_evaporable_water[pos]
        pos3 = model.global_mask(pos2, pos)
        evaporation_reduction_coeff = np.ones(pos.sum())  # K_r = 1 when D_e,i-1 <= REW
        evaporation_reduction_coeff[pos2] = (
            (s_et.total_evaporable_water[pos3] - s_et.cum_soil_surface_depletion[pos3])
            / (s_et.total_evaporable_water[pos3] - s_et.readily_evaporable_water[pos3])
        )

        # Calculate evaporation coefficient (eq. (71))
        s_et.evaporation_coeff[pos] = np.minimum(
            evaporation_reduction_coeff * (max_crop_coeff - s_et.basal_crop_coeff[pos]),
            exposed_wetted_frac * max_crop_coeff,
        )

        s_et.crop_coeff[pos] = s_et.basal_crop_coeff[pos] + s_et.evaporation_coeff[pos]
        s_et.evaporation[pos] = s_et.evaporation_coeff[pos] * s_et.et_ref[pos]
        s_et.transpiration[pos] = s_et.basal_crop_coeff[pos] * s_et.et_ref[pos]
        s_et.evapotranspiration[pos] = s_et.evaporation[pos] + s_et.transpiration[pos]

        # Initialize water balance variables
        precip = s.meteo.rainfall[pos]
        precip_runoff = 0.  # as suggested by [1]
        irrigation = 0.
        soil_transpiration = 0.  # as suggested by [1]

        # Calculate water balance for the surface soil layer
        s_et.deep_percolation_evaporation_layer[pos] = (  # eq. (79)
            precip - precip_runoff
            + irrigation / wetted_frac
            - s_et.deep_percolation_evaporation_layer[pos]
        ).clip(min=0)
        s_et.cum_soil_surface_depletion[pos] = (  # eq. (77)
            s_et.cum_soil_surface_depletion[pos]
            - (precip - precip_runoff)
            - irrigation / wetted_frac
            + s_et.evaporation[pos] / exposed_wetted_frac
            + soil_transpiration
            + s_et.deep_percolation_evaporation_layer[pos]
        ).clip(min=0)
        s_et.cum_soil_surface_depletion[pos] = np.minimum(  # eq. (78)
            s_et.cum_soil_surface_depletion[pos],
            s_et.total_evaporable_water[pos],
        )

    def _water_stress_coefficient(self, pos):
        model = self.model
        s = model.state
        s_et = s.evapotranspiration

        s_et.water_stress_coeff[pos] = (  # eq. (84)
            (s_et.total_available_water[pos] - s_et.cum_root_zone_depletion[pos])
            / (s_et.total_available_water[pos] - s_et.readily_available_water[pos])
        ).clip(min=0, max=1)

    def _root_zone_water_balance(self, pos):
        """
        Calculate water balance for the root zone.
        """
        model = self.model
        s = model.state
        s_et = s.evapotranspiration

        precip = s.meteo.rainfall[pos]
        precip_runoff = 0.  # as suggested by [1]
        irrigation = 0.
        capillary_rise = 0.  # assumed to be zero when the water table is more than about 1 m below the bottom of the root zone [1]

        s_et.deep_percolation[pos] = (  # eq. (88)
            (precip - precip_runoff)
            + irrigation
            - s_et.evapotranspiration[pos]
            - s_et.cum_root_zone_depletion[pos]
        ).clip(min=0)

        s_et.cum_root_zone_depletion[pos] = (  # eq. (85)
            s_et.cum_root_zone_depletion[pos]
            - (precip - precip_runoff)
            - irrigation
            - capillary_rise
            + s_et.evapotranspiration[pos]
            + s_et.deep_percolation[pos]
        ).clip(min=0)
        s_et.cum_root_zone_depletion[pos] = np.minimum(  # eq. (86)
            s_et.cum_root_zone_depletion[pos],
            s_et.total_available_water[pos],
        )


def climate_correction(mean_wind_speed, mean_min_rel_hum, plant_height):
    """
    Calculate the climate correction term for the crop coefficients (right part
    of eq. (70) from [1]).

    Parameters
    ----------
    mean_wind_speed : float or ndarray(float)
        Mean wind speed value during the period of interest (m s-1).

    mean_min_rel_hum : float or ndarray(float)
        Mean value for daily minimum relative humidity during the period of
        interest (%).

    plant_height : float
        Plant height during the period of interest (m).

    Returns
    -------
    clim_corr : float or ndarray(float)
        Climate correction term.

    References
    ----------
    .. [1] Allen, R.G., Pereira, L.S., Raes, D., et al. (1998). Crop
       Evapotranspiration-Guidelines for Computing Crop Water Requirements-FAO
       Irrigation and Drainage Paper 56. FAO, Rome, 300(9): D05109.
       http://www.fao.org/3/x0490e/x0490e00.htm
    """
    return (
        0.04 * (mean_wind_speed - 2)
        - 0.004 * (mean_min_rel_hum - 45)
    ) * (plant_height / 3)**0.3


def crop_coefficient(
    growing_period_day,
    length_ini,
    length_dev,
    length_mid,
    length_late,
    crop_coeff_ini,
    crop_coeff_mid,
    crop_coeff_end,
    crop_coeff_min,
):
    """
    Calculate the crop coefficient Kc or basal crop coefficient K_cb for a given
    day following [1].

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

    crop_coeff_min : float
        Crop coefficient outside of the growing period.

    Returns
    -------
    crop_coeff : float or ndarray(float)
        Crop coefficient for the given day.
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
        crop_coeff = crop_coeff_min
    elif growing_period_day < length_ini:  # initial
        crop_coeff = crop_coeff_ini
    elif growing_period_day < (length_ini + length_dev):  # crop development
        crop_coeff = (  # eq. (66)
            crop_coeff_ini
            + (growing_period_day - length_ini) / length_dev
            * (crop_coeff_mid - crop_coeff_ini)
        )
    elif growing_period_day < (length_ini + length_dev + length_mid):  # mid season
        crop_coeff = crop_coeff_mid
    else:  # late season
        crop_coeff = (  # eq. (66)
            crop_coeff_mid
            + (growing_period_day - (length_ini + length_dev + length_mid)) / length_late
            * (crop_coeff_end - crop_coeff_mid)
        )

    return crop_coeff
