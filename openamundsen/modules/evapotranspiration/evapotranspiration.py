from openamundsen import constants as c, meteo
import numpy as np
from .soiltexture import SoilTextureClass


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

# Reduce depletion fractions for fine textured soils by 5-10% and increase by 5-10% for coarse
# textured soils ([1], p. 167)
DEFAULT_DEPLETION_FRACTION_ADJUSTMENTS = {
    SoilTextureClass.SAND: 1.10,
    SoilTextureClass.LOAMY_SAND: 1.075,
    SoilTextureClass.SANDY_LOAM: 1.05,
    SoilTextureClass.LOAM: 1.025,
    SoilTextureClass.SILT_LOAM: 1.0,
    SoilTextureClass.SILT: 0.975,
    SoilTextureClass.SILT_CLAY_LOAM: 0.95,
    SoilTextureClass.SILTY_CLAY: 0.925,
    SoilTextureClass.CLAY: 0.90,
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
       https://www.researchgate.net/publication/284300773_FAO_Irrigation_and_drainage_paper_No_56
    """

    def __init__(self, model):
        self.model = model
        s = model.state

        s_et = s.add_category('evapotranspiration')
        s_et.add_variable('soil_texture', long_name='Soil texture class', dtype=int)  # TODO move to base or soil group eventually
        s_et.add_variable('evaporation', 'kg m-2', 'Evaporation')
        s_et.add_variable('transpiration', 'kg m-2', 'Transpiration')
        s_et.add_variable('evapotranspiration', 'kg m-2', 'Evapotranspiration')
        s_et.add_variable('et_ref', 'kg m-2', 'Reference evapotranspiration')
        s_et.add_variable('soil_heat_flux', 'W m-2', 'Soil heat flux beneath the grass reference surface')
        s_et.add_variable('crop_coeff', '1', 'Crop coefficient')
        s_et.add_variable('basal_crop_coeff', '1', 'Basal crop coefficient')
        s_et.add_variable('evaporation_coeff', '1', 'Evaporation coefficient')
        s_et.add_variable('water_stress_coeff', '1', 'Water stress coefficient')
        s_et.add_variable('clim_corr', '1', 'Climate correction term')
        s_et.add_variable('cum_soil_surface_depletion', 'kg m-2', 'Cumulative evaporation from the soil surface layer')
        s_et.add_variable('cum_root_zone_depletion', 'kg m-2', 'Cumulative evapotranspiration from the root zone')
        s_et.add_variable('total_evaporable_water', 'kg m-2', 'Total evaporable water')
        s_et.add_variable('total_available_water', 'kg m-2', 'Total available water')
        s_et.add_variable('readily_evaporable_water', 'kg m-2', 'Readily evaporable water')
        s_et.add_variable('readily_available_water', 'kg m-2', 'Readily available water')
        s_et.add_variable('deep_percolation', 'kg m-2', 'Deep percolation')
        s_et.add_variable('deep_percolation_evaporation_layer', 'kg m-2', 'Deep percolation from the evaporation layer')
        s_et.add_variable('sealed_interception', 'kg m-2', 'Interception for sealed surfaces')

    def initialize(self):
        model = self.model
        roi = model.grid.roi
        s = self.model.state
        s_et = s.evapotranspiration

        # Prepare unique land cover classes occurring in the model domain and their associated pixel
        # locations
        lccs = np.unique(s.land_cover.land_cover[roi])
        lccs = lccs[lccs > 0]
        lccs = set(lccs) & set(model.config.land_cover.classes.keys())  # calculate ET only for land cover classes with set parameters
        self.land_cover_class_pixels = {}
        for lcc in lccs:
            self.land_cover_class_pixels[lcc] = (s.land_cover.land_cover == lcc) & roi

        self._climate_correction()

        # Initialize depletion fractions
        depletion_fraction = np.full((model.grid['rows'], model.grid['cols']), np.nan)
        for lcc, pos in self.land_cover_class_pixels.items():
            depletion_fraction[pos] = model.config['land_cover']['classes'][lcc]['depletion_fraction']

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

            # Adjust depletion fraction depending on soil type ([1], p. 167)
            depletion_fraction[pos] *= DEFAULT_DEPLETION_FRACTION_ADJUSTMENTS[stc]

        # Finish calculation of TAW (eq. (82)) and root zone depletion (eq. (87)) and calculate
        # readily available water (eq. (83))
        # (depletion fractions are assumed constant; adjustment using ETc as suggested in [1] (p.
        # 162) is not performed here)
        for lcc, pos in self.land_cover_class_pixels.items():
            rooting_depth = model.config['land_cover']['classes'][lcc]['rooting_depth']
            s_et.total_available_water[pos] *= rooting_depth
            s_et.cum_root_zone_depletion[pos] *= rooting_depth
            s_et.readily_available_water[pos] = (
                depletion_fraction[pos] * s_et.total_available_water[pos]
            )

        # Set D_e to 0 at the start of the model run, i.e., assume the topsoil is near field
        # capacity following a heavy rain or irrigation
        s_et.cum_soil_surface_depletion[roi] = 0.

        s_et.deep_percolation_evaporation_layer[roi] = 0.

        # Initialize sealed interception for sealed surfaces
        for lcc, pos in self.land_cover_class_pixels.items():
            if model.config['land_cover']['classes'][lcc].get('is_sealed', False):
                s_et.sealed_interception[pos] = 0.

    def _climate_correction(self):
        """
        Calculate climate correction term for the crop coefficients.
        """
        model = self.model
        s = self.model.state
        s_et = s.evapotranspiration

        for lcc, pos in self.land_cover_class_pixels.items():
            plant_height = model.config['land_cover']['classes'][lcc]['max_height']
            s_et.clim_corr[pos] = climate_correction(
                model.config.evapotranspiration.mean_wind_speed,
                model.config.evapotranspiration.mean_min_humidity,
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

        model.logger.debug('Calculating evapotranspiration')

        self._reference_evapotranspiration()

        for lcc, pos in self.land_cover_class_pixels.items():
            lcc_params = model.config['land_cover']['classes'][lcc]

            if lcc_params.get('is_sealed', False):
                # Sealed surfaces are treated separately
                self._sealed_evaporation(pos, lcc)
                continue

            crop_coefficient_type = lcc_params['crop_coefficient_type']
            growing_period_day = model.land_cover.growing_period_day(lcc)
            crop_coeff_ini, crop_coeff_mid, crop_coeff_end = lcc_params['crop_coefficients']
            is_water_body = lcc_params.get('is_water_body', False)
            (
                length_ini,
                length_dev,
                length_mid,
                length_late,
            ) = model.land_cover.growth_stage_lengths(lcc)

            # Apply climate correction for Kcb_mid and Kcb_end values >= 0.45 (eq. (70))
            # (convert crop_coeff_mid and crop_coeff_end into fields to allow for possibly
            # non-uniform climate correction values)
            if crop_coeff_mid >= 0.45:
                crop_coeff_mid = np.full(pos.sum(), crop_coeff_mid) + s_et.clim_corr[pos]
            if crop_coeff_end >= 0.45:
                crop_coeff_end = np.full(pos.sum(), crop_coeff_end) + s_et.clim_corr[pos]

            # Adjust Kcb_mid for sparse vegetation
            if lcc_params.get('is_sparse', False):
                crop_coeff_mid = sparse_vegetation_adjustment(
                    model.config.evapotranspiration.min_crop_coefficient,
                    crop_coeff_mid,
                    lcc_params.sparse_vegetation_fraction,
                    lcc_params.max_height,
                    np.deg2rad(model.grid.center_lat),
                    model.sun_params['declination_angle'],
                )

            (crop_coeff, plant_height) = crop_coefficient(
                growing_period_day,
                length_ini,
                length_dev,
                length_mid,
                length_late,
                crop_coeff_ini,
                crop_coeff_mid,
                crop_coeff_end,
                model.config.evapotranspiration.min_crop_coefficient,
                max_plant_height=lcc_params.max_height,
            )

            # If scale_height is True, set the plant height to the calculated value according to the
            # crop coefficient curve, otherwise assume a constant height over the season
            if lcc_params.get('scale_height', True):
                s.land_cover.plant_height[pos] = plant_height
            else:
                s.land_cover.plant_height[pos] = lcc_params.max_height

            # Derive global masks for pixels with the current land cover class which are
            # snow-covered and snow-free
            pos_snow = model.global_mask(pos[roi] & snowies_roi)
            pos_snowfree = model.global_mask(pos[roi] & (~snowies_roi))

            # Ignore snow cover for water
            if is_water_body:
                pos_snowfree = model.global_mask(pos[roi])
                pos_snow = model.global_mask(~pos[roi])

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

            # Adjust ET for soil water stress conditions (except for water bodies)
            if not is_water_body:
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

        soil_heat_flux_factor = 0.1 if model.sun_params['sun_over_horizon'] else 0.5
        s_et.soil_heat_flux[roi] = soil_heat_flux_factor * s.meteo.net_radiation[roi]  # eq. (45-46)

        Wm2_to_MJm2h = 1e-6 * c.SECONDS_PER_HOUR  # conversion factor from W m-2 (= J m-2 s-1) to MJ m-2 h-1

        Rn = s.meteo.net_radiation[roi] * Wm2_to_MJm2h  # net radiation at the grass surface (MJ m-2 h-1)
        G = s_et.soil_heat_flux[roi] * Wm2_to_MJm2h  # soil heat flux density (MJ m-2 h-1)
        T = s.meteo.top_canopy_temp[roi] - c.T0  # air temperature (°C)
        D = 4098 * (0.6108 * np.exp(17.27 * T / (T + 273.3))) / (T + 273.3)**2  # slope of the relationship between saturation vapor pressure and temperature (kPa °C-1) (eq. (13))
        gamma = s.meteo.psych_const[roi] * 1e-3  # psychrometric constant (kPa °C-1)
        es = s.meteo.sat_vap_press[roi] * 1e-3  # saturation vapor pressure (kPa)
        ea = s.meteo.vap_press[roi] * 1e-3  # actual vapor pressure (kPa)

        grass_roughness_length = 0.03  # (m)
        u2 = meteo.log_wind_profile(  # 2 m wind speed (m s-1)
            s.meteo.top_canopy_wind_speed[roi],
            model.config.meteo.measurement_height.wind,
            2,
            grass_roughness_length,
        )

        ET0 = (  # reference evapotranspiration (kg m-2 h-1) (eq. (53))
            (0.408 * D * (Rn - G) + gamma * 37 / (T + 273) * u2 * (es - ea))
            / (D + gamma * (1 + 0.34 * u2))
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

        plant_height = s.land_cover.plant_height[pos]
        min_crop_coeff = model.config.evapotranspiration.min_crop_coefficient

        # Calculate K_c_max (eq. (72))
        # TODO the climate correction term is intended to be calculated using mean values of wind
        # speed and daily-minimum relative humidity over the period of interest - maybe better use
        # 24-hour moving averages than instantaneous values?
        clim_corr = climate_correction(
            s.meteo.top_canopy_wind_speed[pos],
            s.meteo.top_canopy_rel_hum[pos],
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

    def _sealed_evaporation(self, pos, lcc):
        """
        Calculate evaporation for sealed surfaces using the Penman-Monteith
        equation.
        """
        model = self.model
        s = model.state
        s_et = s.evapotranspiration
        lcc_params = model.config.land_cover.classes[lcc]
        max_interception = lcc_params.max_sealed_interception

        s_et.sealed_interception[pos] += s.meteo.rainfall[pos]
        runoff = (s_et.sealed_interception[pos] - max_interception).clip(min=0)
        s_et.sealed_interception[pos] -= runoff
        s_et.deep_percolation[pos] = runoff  # runoff is currently treated as deep percolation for sealed surfaces (should be improved)

        rs = 0.  # stomatal resistance (s m-1)
        zom = 0.123 * lcc_params.max_height  # roughness length governing momentum transfer (m)
        zoh = 0.1 * zom  # roughness length governing heat and vapor (m)
        d = 2./3 * lcc_params.max_height  # zero-plane displacement height (m)
        ra = (  # aerodynamic resistance (s m-1) (eq. (4))
            np.log((model.config.meteo.measurement_height.wind - d) / zom)
            * np.log((model.config.meteo.measurement_height.temperature - d) / zoh)
            / (c.VON_KARMAN**2 * s.meteo.top_canopy_wind_speed[pos])
        )

        Rn = s.meteo.net_radiation[pos]  # net radiation (W m-2)
        G = s_et.soil_heat_flux[pos]  # soil heat flux density (W m-2)
        T = s.meteo.top_canopy_temp[pos] - c.T0  # air temperature (°C)
        D = 1e3 * 4098 * (0.6108 * np.exp(17.27 * T / (T + 273.3))) / (T + 273.3)**2  # slope of the relationship between saturation vapor pressure and temperature (Pa K-1) (eq. (13))

        gamma = s.meteo.psych_const[pos]  # psychrometric constant (Pa K-1)
        es = s.meteo.sat_vap_press[pos]  # saturation vapor pressure (Pa)
        ea = s.meteo.vap_press[pos]  # actual vapor pressure (Pa)

        Tv = 1.01 * s.meteo.top_canopy_temp[pos]  # virtual temperature (K)
        rhoa = s.meteo.atmos_press[pos] / (c.GAS_CONSTANT_DRY_AIR * Tv)  # air density at constant pressure (kg m-3)
        cp = (  # specific heat at constant pressure (J kg-1 K-1) (p. 26)
            gamma * 0.622 * c.LATENT_HEAT_OF_VAPORIZATION
            / s.meteo.atmos_press[pos]
        )

        evaporation_Wm2 = (  # evaporation (W m-2) (eq. (3))
            (D * (Rn - G) + rhoa * cp * (es - ea) / ra)
            / (D + gamma * (1 + rs / ra))
        ).clip(min=0)
        s_et.evaporation[pos] = np.minimum(  # (kg m-2)
            evaporation_Wm2 / c.LATENT_HEAT_OF_VAPORIZATION * model.timestep,
            s_et.sealed_interception[pos],
        )
        s_et.sealed_interception[pos] -= s_et.evaporation[pos]

        s_et.transpiration[pos] = 0.
        s_et.evapotranspiration[pos] = s_et.evaporation[pos] + s_et.transpiration[pos]


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
       https://www.researchgate.net/publication/284300773_FAO_Irrigation_and_drainage_paper_No_56
    """
    mean_wind_speed = np.clip(mean_wind_speed, 1, 6)
    mean_min_rel_hum = np.clip(mean_min_rel_hum, 20, 80)
    plant_height = np.clip(plant_height, 0.1, 10)

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
    max_plant_height=None,
):
    """
    Calculate the crop coefficient K_c or basal crop coefficient K_cb for a
    given day following [1].

    The lengths of the individual crop development stages can either be scalars
    or arrays. The former corresponds to calculation of a single crop
    coefficient curve (see [1], p. 127, "Annual crops"), whereas the latter
    corresponds to a crop coefficient curve composed of a series of subcycles
    (see [1], p. 127, "K_c curves for forage crops").

    If the `max_plant_height` parameter is set, in addition to the crop
    coefficient the plant height for the given day is calculated by multiplying
    the maximum plant height by Kcb/Kcb_mid, while assuming that the plant
    height does not decrease with time (see [1], p. 277, footnote 3).

    Parameters
    ----------
    growing_period_day : int
        Day within the growing period (1 = first day of the period).

    length_ini : int or array-like
        Length(s) of the initial growth stage(s) (days).

    length_dev : int or array-like
        Length(s) of the crop development stage(s) (days).

    length_mid : int or array-like
        Length(s) of the mid-season stage(s) (days).

    length_late : int or array-like
        Length(s) of the late season stage(s) (days).

    crop_coeff_ini : float
        Crop coefficient for the initial stage.

    crop_coeff_mid : float or ndarray(float)
        Crop coefficient for the mid-season stage.

    crop_coeff_end : float or ndarray(float)
        Crop coefficient for the end of the late season stage.

    crop_coeff_min : float
        Crop coefficient outside of the growing period.

    max_plant_height : float, default None
        Maximum plant height during the mid-season stage.

    Returns
    -------
    crop_coeff : float or ndarray(float)
        Crop coefficient for the given day.
        Depending on the data types of crop_coeff_mid and crop_coeff_end, this
        is either a scalar or an array.

    plant_height : float or ndarray(float)
        Plant height for the given day (only returned if max_plant_height is
        not None).

    References
    ----------
    .. [1] Allen, R.G., Pereira, L.S., Raes, D., et al. (1998). Crop
       Evapotranspiration-Guidelines for Computing Crop Water Requirements-FAO
       Irrigation and Drainage Paper 56. FAO, Rome, 300(9): D05109.
       http://www.fao.org/3/x0490e/x0490e00.htm
       https://www.researchgate.net/publication/284300773_FAO_Irrigation_and_drainage_paper_No_56
    """
    length_ini = np.atleast_1d(length_ini)
    length_dev = np.atleast_1d(length_dev)
    length_mid = np.atleast_1d(length_mid)
    length_late = np.atleast_1d(length_late)

    if not (
        length_ini.shape
        == length_dev.shape
        == length_mid.shape
        == length_late.shape
    ):
        raise ValueError('Growth period length arrays have unequal sizes')

    lengths = np.vstack((length_ini, length_dev, length_mid, length_late)).flatten(order='F')
    lengths_cum = lengths.cumsum()
    idx = np.searchsorted(np.concatenate([[0], lengths_cum]), growing_period_day)
    period_num = (idx - 1) % 4

    if idx in (0, len(lengths_cum) + 1):  # outside of growing period
        crop_coeff = crop_coeff_min
    elif period_num == 0:  # initial
        crop_coeff = crop_coeff_ini
    elif period_num == 1:  # crop development
        crop_coeff = (  # eq. (66)
            crop_coeff_ini
            + (growing_period_day - lengths_cum[idx - 2]) / lengths[idx - 1]
            * (crop_coeff_mid - crop_coeff_ini)
        )
    elif period_num == 2:  # mid season
        crop_coeff = crop_coeff_mid
    elif period_num == 3:  # late season
        crop_coeff = (  # eq. (66)
            crop_coeff_mid
            + (growing_period_day - lengths_cum[idx - 2]) / lengths[idx - 1]
            * (crop_coeff_end - crop_coeff_mid)
        )

    if max_plant_height is not None:
        if idx in (0, len(lengths_cum) + 1):  # outside of growing period
            min_plant_height = 0.
        elif period_num == 0:  # initial
            min_plant_height = crop_coeff_min / crop_coeff_mid * max_plant_height
        elif period_num == 1:  # crop development
            min_plant_height = crop_coeff_ini / crop_coeff_mid * max_plant_height
        elif period_num == 2:  # mid season
            min_plant_height = max_plant_height
        elif period_num == 3:  # late season
            min_plant_height = max_plant_height

        plant_height = np.maximum(
            crop_coeff / crop_coeff_mid * max_plant_height,
            min_plant_height,
        )

    if max_plant_height is None:
        return crop_coeff
    else:
        return (crop_coeff, plant_height)


def sparse_vegetation_adjustment(
    crop_coeff_min,
    crop_coeff_mid,
    veg_frac,
    plant_height,
    lat,
    declination_angle,
):
    """
    Adjust the mid-season crop coefficient for sparsely covered vegetation
    following [1] (Chapter 9), assuming round or spherical shaped canopies
    (such as trees).

    Parameters
    ----------
    crop_coeff_min : float
        Crop coefficient outside of the growing period.

    crop_coeff_mid : float or ndarray(float)
        Unadjusted crop coefficient for the mid-season stage.

    veg_frac : float
        Fraction of soil surface covered by vegetation.

    plant_height : float
        Plant height (m).

    lat : float
        Latitude (radians).

    declination_angle : float
        Solar declination angle (radians).

    Returns
    -------
    crop_coeff_mid_adj : float or ndarray(float)
        Adjusted mid-season crop coefficient.

    References
    ----------
    .. [1] Allen, R.G., Pereira, L.S., Raes, D., et al. (1998). Crop
       Evapotranspiration-Guidelines for Computing Crop Water Requirements-FAO
       Irrigation and Drainage Paper 56. FAO, Rome, 300(9): D05109.
       http://www.fao.org/3/x0490e/x0490e00.htm
       https://www.researchgate.net/publication/284300773_FAO_Irrigation_and_drainage_paper_No_56
    """
    # Mean angle above the sun during the period of maximum evapotranspiration
    time_angle = 0.  # calculate for solar noon (12:00), i.e., time angle = 0
    sin_mean_angle_above_sun = (
        np.sin(lat) * np.sin(declination_angle)
        + np.cos(lat) * np.cos(declination_angle) * np.cos(time_angle)
    )

    veg_frac_eff = np.array(veg_frac / sin_mean_angle_above_sun).clip(min=0)
    crop_coeff_mid_adj = (  # eq. (98)
        crop_coeff_min +
        (crop_coeff_mid - crop_coeff_min)
        * min(
            1,
            2 * veg_frac,
            veg_frac_eff**(1 / (1 + plant_height)),
        )
    )
    return crop_coeff_mid_adj
