import numpy as np
from loguru import logger

from openamundsen import constants as c
from openamundsen.modules.snow import CryoLayerID

MIN_ICE_THICKNESS = 10.0  # minimum required ice thickness for applying the delta h function (m)
TERMINUS_MASS_BALANCE_ELEVATION_PERCENTAGE = 10.0  # (%)
TERMINUS_ELEVATION_BUFFER = 50.0  # (m)


class GlacierModel:
    """
    Glacier geometry update model following [1].

    References
    ----------
    .. [1] Huss, M., Jouvet, G., Farinotti, D., & Bauder, A. (2010). Future
       high-mountain hydrology: A new parameterization of glacier retreat.
       Hydrology and Earth System Sciences, 14(5), 815–829.
       https://doi.org/10.5194/hess-14-815-2010
    """

    def __init__(self, model):
        self.model = model
        s = model.state
        sg = s.add_category("glaciers")
        sg.add_variable("glacier", long_name="Glacier ID", dtype=int)
        self.mass_balance_start_we = None

    def initialize(self):
        model = self.model
        roi = model.grid.roi
        s = self.model.state
        sg = s.glaciers
        ids = np.unique(sg.glacier[roi])
        self.glacier_ids = ids[ids > 0]

    def update(self):
        model = self.model
        s = model.state
        timestep_props = model.timestep_props
        mb_start_month = model.config.glaciers.delta_h.mass_balance_year_start_month
        mb_end_month = mb_start_month - 1 if mb_start_month > 1 else 12

        if model.date.month == mb_start_month and timestep_props.strict_first_of_month:
            self.mass_balance_start_we = s.snow.ice_content + s.snow.liquid_water_content
        elif (
            model.date.month == mb_end_month
            and timestep_props.strict_last_of_month
            and self.mass_balance_start_we is not None
        ):
            self._update_geometry()

    def _update_geometry(self):
        model = self.model
        s = model.state

        mass_balance_end_we = s.snow.ice_content + s.snow.liquid_water_content
        mb_layers = mass_balance_end_we - self.mass_balance_start_we
        ice_mb = mb_layers[CryoLayerID.ICE, :]
        firn_mb = mb_layers[CryoLayerID.FIRN, :]
        mb = ice_mb + firn_mb
        old_ice_thickness = self.mass_balance_start_we[CryoLayerID.ICE, :] / c.ICE_DENSITY
        new_ice_thickness = s.snow.thickness[CryoLayerID.ICE, :].copy()

        for gid in self.glacier_ids:
            gpos = (s.glaciers.glacier == gid) & (old_ice_thickness > 0.0)
            num_glacier_pixels = gpos.sum()

            if not gpos.any():
                continue

            logger.debug(f"Updating geometry for glacier {gid}")

            apply_default_delta_h = True

            # TODO: implement glacier advance

            if not apply_default_delta_h:
                continue

            update_pos_local = old_ice_thickness[gpos] >= MIN_ICE_THICKNESS
            update_pos = model.global_mask(update_pos_local, gpos)
            if update_pos_local.sum() < 2:
                # At least 2 pixels are required for calculating the normalized elevation range
                logger.debug("Less than 2 pixels available, skipping glacier")
                continue

            glacier_elevs = model.state.base.dem[gpos]
            glacier_mbs = mb[gpos]
            total_mb = glacier_mbs[update_pos_local].sum()
            glacier_area = num_glacier_pixels * model.grid.resolution**2  # (m2)
            terminus_elev = glacier_elevs.min()

            glacier_elevs_sorted_idxs = np.argsort(glacier_elevs)
            num_terminus_pixels = int(
                np.ceil(num_glacier_pixels * (TERMINUS_MASS_BALANCE_ELEVATION_PERCENTAGE / 100.0))
            )

            # Calculate the highest elevation of the lowest
            # TERMINUS_MASS_BALANCE_ELEVATION_PERCENTAGE percent of glacier pixels
            max_terminus_elev = glacier_elevs[glacier_elevs_sorted_idxs[:num_terminus_pixels]].max()

            # If the highest elevation of the lowest TERMINUS_ELEVATION_BUFFER m (by elevation) of
            # the glacier is higher, take this one instead
            max_terminus_elev = max(max_terminus_elev, terminus_elev + TERMINUS_ELEVATION_BUFFER)

            # Take the lowest mass balance value within this elevation range
            pos = model.global_mask(model.state.base.dem[gpos] <= max_terminus_elev, gpos)
            terminus_mb = mb[pos].min()

            max_allowed_change = min(terminus_mb, 0.0)  # MB must be negative
            mass_change = _ice_mass_change(
                glacier_elevs[update_pos_local],
                total_mb,
                glacier_area,
                max_allowed_change,
            )

            new_ice_thickness[update_pos] = (
                old_ice_thickness[update_pos] + (mass_change - firn_mb[update_pos]) / c.ICE_DENSITY
            ).clip(min=0.0)

        s.snow.thickness[CryoLayerID.ICE, :] = new_ice_thickness
        s.snow.ice_content[CryoLayerID.ICE, :] = new_ice_thickness * c.ICE_DENSITY

        model.snow.update_layers()

        # TODO: when glacier advance is enabled, update the previous ice mass distributions here


def _ice_mass_change(elevs, total_mb, glacier_area, max_allowed_change):
    """
    Calculate the distributed ice mass change for a single glacier.

    Parameters
    ----------
    elevs : ndarray(float)
        Glacier elevations (m).

    total_mb : float
        Total glacier surface mass balance (kg m-2).

    glacier_area : float
        Glacier area (m2).

    max_allowed_change : float
        Maximum allowed surface lowering (kg m-2).

    Returns
    -------
    mass_change : ndarray(float)
        Ice mass changes for the glacier pixels (kg m-2).

    References
    ----------
    .. [1] Huss, M., Jouvet, G., Farinotti, D., & Bauder, A. (2010). Future
       high-mountain hydrology: A new parameterization of glacier retreat.
       Hydrology and Earth System Sciences, 14(5), 815–829.
       https://doi.org/10.5194/hess-14-815-2010
    """
    if total_mb < 0:
        min_elev = elevs.min()
        max_elev = elevs.max()
        normalized_elevs = (max_elev - elevs) / (max_elev - min_elev)
        delta_h = _delta_h(normalized_elevs, glacier_area)
        fs = total_mb / delta_h.sum()
        mass_change = fs * delta_h

        # Restrict the surface lowering to max_allowed_change
        if max_allowed_change < 0:
            clippies = mass_change < max_allowed_change
            if clippies.any() and (~clippies).sum() > 1:
                total_we_change = mass_change.sum()
                mass_change[clippies] = max_allowed_change
                we_change_diff = total_we_change - mass_change.sum()
                mass_change[~clippies] += _mass_loss_surplus_distribution(
                    we_change_diff,
                    elevs[~clippies],
                )
    else:
        # If the mass balance is positive, distribute the mass gain equally over all glacier pixels
        mass_change = total_mb / elevs.size

    return mass_change


def _delta_h(h, glacier_area):
    if glacier_area >= 20e6:
        delta_h = -((h - 0.02) ** 6 + 0.12 * (h - 0.02))
    elif glacier_area >= 5e6:
        delta_h = -((h - 0.05) ** 4 + 0.19 * (h - 0.05) + 0.01)
    else:
        delta_h = -((h - 0.30) ** 2 + 0.60 * (h - 0.30) + 0.09)

    return delta_h.clip(-1, 0)


def _mass_loss_surplus_distribution(mass_loss, elevs):
    min_elev = elevs.min()
    max_elev = elevs.max()
    min_mass = 0.0

    x0 = 0.0
    x1 = mass_loss / elevs.size * 1.5
    y0 = np.sum(_scale_mass_loss_surplus(mass_loss, elevs, min_elev, max_elev, min_mass, x0))
    y1 = np.sum(_scale_mass_loss_surplus(mass_loss, elevs, min_elev, max_elev, min_mass, x1))

    k = (y1 - y0) / (x1 - x0)
    d = y0 - k * x0

    max_mass = (mass_loss - d) / k
    mass_loss_dist = _scale_mass_loss_surplus(
        mass_loss,
        elevs,
        min_elev,
        max_elev,
        min_mass,
        max_mass,
    )

    return mass_loss_dist


def _scale_mass_loss_surplus(mass_loss, elevs, min_elev, max_elev, min_mass, max_mass):
    return (max_mass - min_mass) * (elevs - min_elev) / (max_elev - min_elev) + min_mass
