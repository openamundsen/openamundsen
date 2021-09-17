from openamundsen import modules
import numpy as np


class LandCover:
    def __init__(self, model):
        self.model = model

        s_lc = model.state.add_category('land_cover')
        s_lc.add_variable('land_cover', long_name='Land cover class', dtype=int, retain=True)
        s_lc.add_variable('plant_height', 'm', 'Plant height')
        s_lc.add_variable('lai', 'm2 m-2', 'Leaf area index')
        s_lc.add_variable('lai_eff', 'm2 m-2', 'Effective leaf area index')

    def initialize(self):
        model = self.model
        roi = model.grid.roi
        s = self.model.state

        # Prepare unique land cover classes occurring in the model domain and their associated pixel
        # locations
        lccs = np.unique(s.land_cover.land_cover[roi])
        lccs = lccs[lccs > 0]
        self.class_pixels = {}
        self.forest_classes = []  # LCCs with is_forest == True
        self.lai_classes = []  # LCCs with valid LAI parameters
        for lcc in lccs:
            try:
                lcc_params = model.config['land_cover']['classes'][lcc]
            except KeyError:
                model.logger.warning(f'Unknown land cover class: {lcc}')
                continue

            self.class_pixels[lcc] = (s.land_cover.land_cover == lcc) & roi

            if lcc_params.get('is_forest', False):
                self.forest_classes.append(lcc)

            if 'leaf_area_index' in lcc_params:
                if (
                    'min' in lcc_params['leaf_area_index']
                    and 'max' in lcc_params['leaf_area_index']
                ):
                    self.lai_classes.append(lcc)
                else:
                    model.logger.warning(f'Incomplete LAI parameters for land cover class {lcc}')

    def lai(self):
        model = self.model
        s_lc = self.model.state.land_cover

        for lcc in self.lai_classes:
            lcc_params = model.config['land_cover']['classes'][lcc]
            min_lai = lcc_params['leaf_area_index']['min']
            max_lai = lcc_params['leaf_area_index']['max']
            (
                length_ini,
                length_dev,
                length_mid,
                length_late,
            ) = self.growth_stage_lengths(lcc)
            growing_period_day = self.growing_period_day(lcc)

            lcc_lai = modules.evapotranspiration.crop_coefficient(
                growing_period_day,
                0,  # no initial period with constant LAI
                length_ini + length_dev,
                length_mid,
                length_late,
                min_lai,
                max_lai,
                min_lai,
                min_lai,
            )

            s_lc.lai[self.class_pixels[lcc]] = lcc_lai
            s_lc.lai_eff[self.class_pixels[lcc]] = (
                lcc_lai + lcc_params['leaf_area_index'].get('effective_add', 0)
            )

    def growth_stage_lengths(self, lcc):
        lcc_params = self.model.config['land_cover']['classes'][lcc]
        lcc_gsls = lcc_params['growth_stage_lengths']

        if isinstance(lcc_gsls[0], int):
            length_ini, length_dev, length_mid, length_late = lcc_gsls
        elif isinstance(lcc_gsls[0], list):  # growing season composed of several subcycles
            length_ini = [g[0] for g in lcc_gsls]
            length_dev = [g[1] for g in lcc_gsls]
            length_mid = [g[2] for g in lcc_gsls]
            length_late = [g[3] for g in lcc_gsls]

        return (length_ini, length_dev, length_mid, length_late)

    def growing_period_day(self, lcc):
        lcc_params = self.model.config['land_cover']['classes'][lcc]
        return self.model.date.dayofyear - lcc_params['plant_date'] + 1  # (1-based)
