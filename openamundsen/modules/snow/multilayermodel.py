import numpy as np
from numba import njit, prange
from openamundsen import constants, constants as c, heatconduction
from openamundsen.snowmodel import SnowModel
from . import snow


class MultilayerSnowModel(SnowModel):
    def __init__(self, model):
        self.model = model

        s = model.state.snow
        num_snow_layers = len(model.config.snow.min_thickness)

        s.add_variable('num_layers', '1', 'Number of snow layers', dtype=int, retain=True)
        s.add_variable('thickness', 'm', 'Snow thickness', dim3=num_snow_layers, retain=True)
        s.add_variable('density', 'kg m-3', 'Snow density', 'snow_density', dim3=num_snow_layers)
        s.add_variable('ice_content', 'kg m-2', 'Ice content of snow', dim3=num_snow_layers, retain=True)
        s.add_variable('liquid_water_content', 'kg m-2', 'Liquid water content of snow', 'liquid_water_content_of_snow_layer', dim3=num_snow_layers, retain=True)
        s.add_variable('temp', 'K', 'Snow temperature', dim3=num_snow_layers, retain=True)
        s.add_variable('therm_cond', 'W m-1 K-1', 'Thermal conductivity of snow', dim3=num_snow_layers, retain=True)
        s.add_variable('heat_cap', 'J K-1 m-2', 'Areal heat capacity of snow', dim3=num_snow_layers)

    def initialize(self):
        roi = self.model.grid.roi
        s = self.model.state.snow

        s.swe[roi] = 0
        s.depth[roi] = 0
        s.area_fraction[roi] = 0
        s.num_layers[roi] = 0
        s.sublimation[roi] = 0
        s.therm_cond[:, roi] = self.model.config.snow.thermal_conductivity
        s.thickness[:, roi] = 0
        s.ice_content[:, roi] = 0
        s.liquid_water_content[:, roi] = 0
        s.temp[:, roi] = constants.T0

    def albedo_aging(self):
        snow.albedo(self.model)

    def compaction(self):
        snow.compaction(self.model)

    def accumulation(self):
        model = self.model
        s = model.state
        pos = s.meteo.snowfall > 0
        self.add_snow(
            pos,
            s.meteo.snowfall[pos],
            density=snow.fresh_snow_density(s.meteo.wet_bulb_temp[pos]),
        )

    def heat_conduction(self):
        model = self.model
        s = model.state
        _heat_conduction(
            model.grid.roi_idxs,
            s.snow.num_layers,
            s.snow.thickness,
            s.soil.thickness,
            model.timestep,
            s.snow.temp,
            s.snow.therm_cond,
            s.soil.therm_cond,
            s.surface.heat_flux,
            s.snow.heat_cap,
        )

    def melt(self):
        model = self.model
        s = model.state
        _melt(
            model.grid.roi_idxs,
            model.timestep,
            s.snow.num_layers,
            s.snow.melt,
            s.snow.thickness,
            s.snow.temp,
            s.snow.ice_content,
            s.snow.liquid_water_content,
            s.snow.heat_cap,
        )

    def sublimation(self):
        model = self.model
        s = model.state
        roi = model.grid.roi

        s.snow.sublimation[roi] = 0.
        pos_roi = (s.snow.ice_content[0, roi] > 0) | (s.surface.temp[roi] < constants.T0)
        pos = model.roi_mask_to_global(pos_roi)
        pot_sublim = -1 * s.surface.moisture_flux[pos] * model.timestep
        pot_sublim[np.isnan(pot_sublim)] = 0.
        s.snow.sublimation[pos] = np.minimum(pot_sublim, s.snow.ice_content[:, pos].sum(axis=0))

        # First resublimation
        frost = -np.minimum(s.snow.sublimation, 0)
        pos = frost > 0
        self.add_snow(
            pos,
            frost[pos],
            density=snow.fresh_snow_density(s.meteo.wet_bulb_temp[pos]),
        )

        # Then sublimation
        _sublimation(
            model.grid.roi_idxs,
            model.timestep,
            s.snow.num_layers,
            s.snow.ice_content,
            s.snow.thickness,
            s.snow.sublimation,
        )

    def runoff(self):
        model = self.model
        s = model.state
        _runoff(
            model.grid.roi_idxs,
            snow.max_liquid_water_content(model),
            s.meteo.rainfall,
            s.snow.num_layers,
            s.snow.thickness,
            s.snow.temp,
            s.snow.ice_content,
            s.snow.liquid_water_content,
            s.snow.refreezing,
            s.snow.runoff,
            s.snow.heat_cap,
        )

    def update_layers(self):
        model = self.model
        s = model.state
        _update_layers(
            model.grid.roi_idxs,
            s.snow.num_layers,
            np.array(model.config.snow.min_thickness),
            s.snow.thickness,
            s.snow.ice_content,
            s.snow.liquid_water_content,
            s.snow.heat_cap,
            s.snow.temp,
            s.snow.density,
            s.snow.depth,
        )
        s.snow.albedo[s.snow.num_layers == 0] = np.nan

    def update_properties(self):
        snow.snow_properties(self.model)

    def add_snow(
            self,
            pos,
            ice_content,
            liquid_water_content=0,
            density=None,
            albedo=None,
    ):
        """
        Add snow to the top of the snowpack.
        """
        model = self.model
        s = model.state

        ice_content = np.nan_to_num(ice_content, nan=0., copy=True)

        pos_init = (s.snow.num_layers[pos] == 0) & (ice_content > 0)
        pos_init_global = model.global_mask(pos_init, pos)

        # If albedo is None, set it to the maximum albedo for currently snow-free pixels and keep
        # the current albedo for the other pixels
        if albedo is None:
            albedo = s.snow.albedo[pos]
            albedo[pos_init] = model.config.snow.albedo.max

        s.snow.albedo[pos] = albedo

        # Initialize first snow layer where necessary
        s.snow.num_layers[pos_init_global] = 1
        s.snow.temp[0, pos_init_global] = np.minimum(s.meteo.temp[pos_init_global], constants.T0)

        # Add snow to first layer
        s.snow.ice_content[0, pos] += ice_content
        s.snow.liquid_water_content[0, pos] += liquid_water_content
        s.snow.thickness[0, pos] += ice_content / density


@njit(cache=True, parallel=True)
def _melt(
    roi_idxs,
    timestep,
    num_layers,
    melt,
    thickness,
    temp,
    ice_content,
    liquid_water_content,
    heat_cap,
):
    """
    Calculate snowmelt following [1].

    Parameters
    ----------
    roi_idxs : ndarray(int, ndim=2)
        (N, 2)-array specifying the (row, col) indices within the data arrays
        that should be considered.

    timestep : float
        Model timestep (s).

    num_layers : ndarray(float, ndim=2)
        Number of snow layers.

    melt : ndarray(float, ndim=2)
        Snowmelt (kg m-2).

    thickness : ndarray(float, ndim=3)
        Snow thickness (m).

    temp : ndarray(float, ndim=3)
        Snow temperature (K).

    ice_content : ndarray(float, ndim=3)
        Ice content of snow (kg m-2).

    liquid_water_content : ndarray(float, ndim=3)
        Liquid water content of snow (kg m-2).

    heat_cap : ndarray(float, ndim=3)
        Areal heat capacity of snow (J K-1 m-2).

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    num_pixels = len(roi_idxs)
    for idx_num in prange(num_pixels):
        i, j = roi_idxs[idx_num]

        ice_content_change = melt[i, j]
        actual_melt = 0.  # actual melt without cold content reduction

        for k in range(num_layers[i, j]):
            cold_content = heat_cap[k, i, j] * (c.T0 - temp[k, i, j])
            if cold_content < 0:
                ice_content_change -= cold_content / c.LATENT_HEAT_OF_FUSION
                temp[k, i, j] = c.T0

            if ice_content_change > 0:
                if ice_content_change > ice_content[k, i, j]:  # layer melts completely
                    ice_content_change -= ice_content[k, i, j]
                    thickness[k, i, j] = 0.
                    liquid_water_content[k, i, j] += ice_content[k, i, j]
                    actual_melt += ice_content[k, i, j]
                    ice_content[k, i, j] = 0.
                else:  # layer melts partially
                    thickness[k, i, j] *= (1 - ice_content_change / ice_content[k, i, j])
                    ice_content[k, i, j] -= ice_content_change
                    liquid_water_content[k, i, j] += ice_content_change
                    actual_melt += ice_content_change
                    ice_content_change = 0.

        melt[i, j] = actual_melt


@njit(cache=True, parallel=True)
def _sublimation(
    roi_idxs,
    timestep,
    num_layers,
    ice_content,
    thickness,
    sublimation,
):
    """
    Calculate snow sublimation following [1].

    Parameters
    ----------
    roi_idxs : ndarray(int, ndim=2)
        (N, 2)-array specifying the (row, col) indices within the data arrays
        that should be considered.

    timestep : float
        Model timestep (s).

    num_layers : ndarray(float, ndim=2)
        Number of snow layers.

    ice_content : ndarray(float, ndim=3)
        Ice content of snow (kg m-2).

    thickness : ndarray(float, ndim=3)
        Snow thickness (m).

    sublimation : ndarray(float, ndim=2)
        Snow sublimation (kg m-2).

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    num_pixels = len(roi_idxs)
    for idx_num in prange(num_pixels):
        i, j = roi_idxs[idx_num]

        ice_content_change = max(sublimation[i, j], 0.)

        if ice_content_change > 0:
            for k in range(num_layers[i, j]):
                if ice_content_change > ice_content[k, i, j]:  # complete sublimation of layer
                    ice_content_change -= ice_content[k, i, j]
                    thickness[k, i, j] = 0.
                    ice_content[k, i, j] = 0.
                else:  # partial sublimation
                    thickness[k, i, j] *= (1 - ice_content_change / ice_content[k, i, j])
                    ice_content[k, i, j] -= ice_content_change
                    ice_content_change = 0.


@njit(cache=True, parallel=True)
def _runoff(
    roi_idxs,
    max_liquid_water_content,
    rainfall,
    num_layers,
    thickness,
    temp,
    ice_content,
    liquid_water_content,
    refreezing,
    runoff,
    heat_cap,
):
    """
    Calculate snowmelt runoff following [1].

    Parameters
    ----------
    roi_idxs : ndarray(int, ndim=2)
        (N, 2)-array specifying the (row, col) indices within the data arrays
        that should be considered.

    max_liquid_water_content : ndarray(float, ndim=3)
        Maximum liquid water content (kg m-2).

    rainfall : ndarray(float, ndim=2)
        Rainfall amount (kg m-2).

    num_layers : ndarray(float, ndim=2)
        Number of snow layers.

    thickness : ndarray(float, ndim=3)
        Snow thickness (m).

    temp : ndarray(float, ndim=3)
        Snow temperature (K).

    ice_content : ndarray(float, ndim=3)
        Ice content of snow (kg m-2).

    liquid_water_content : ndarray(float, ndim=3)
        Liquid water content of snow (kg m-2).

    refreezing : ndarray(float, ndim=2)
        Liquid water refreezing (kg m-2).

    runoff : ndarray(float, ndim=2)
        Snow runoff (kg m-2).

    heat_cap : ndarray(float, ndim=3)
        Areal heat capacity of snow (J K-1 m-2).

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    num_pixels = len(roi_idxs)
    for idx_num in prange(num_pixels):
        i, j = roi_idxs[idx_num]

        refreezing[i, j] = 0.

        runoff[i, j] = rainfall[i, j]
        if np.isnan(runoff[i, j]):
            runoff[i, j] = 0.

        for k in range(num_layers[i, j]):
            liquid_water_content[k, i, j] += runoff[i, j]

            if liquid_water_content[k, i, j] > max_liquid_water_content[k, i, j]:
                runoff[i, j] = liquid_water_content[k, i, j] - max_liquid_water_content[k, i, j]
                liquid_water_content[k, i, j] = max_liquid_water_content[k, i, j]
            else:
                runoff[i, j] = 0.

            # Refreeze liquid water
            cold_content = heat_cap[k, i, j] * (c.T0 - temp[k, i, j])
            if cold_content > 0:
                ice_content_change = min(
                    liquid_water_content[k, i, j],
                    cold_content / c.LATENT_HEAT_OF_FUSION,
                )
                liquid_water_content[k, i, j] -= ice_content_change
                ice_content[k, i, j] += ice_content_change
                refreezing[i, j] += ice_content_change
                temp[k, i, j] += c.LATENT_HEAT_OF_FUSION * ice_content_change / heat_cap[k, i, j]


@njit(parallel=True, cache=True)
def _heat_conduction(
    roi_idxs,
    num_layers,
    snow_thickness,
    soil_thickness,
    timestep,
    temp,
    therm_cond_snow,
    therm_cond_soil,
    heat_flux,
    heat_cap,
):
    """
    Update snow layer temperatures.

    Parameters
    ----------
    roi_idxs : ndarray(int, ndim=2)
        (N, 2)-array specifying the (row, col) indices within the data arrays
        that should be considered.

    num_layers : ndarray(float, ndim=2)
        Number of snow layers.

    snow_thickness : ndarray(float, ndim=3)
        Snow thickness (m).

    soil_thickness : ndarray(float, ndim=3)
        Soil thickness (m).

    timestep : float
        Model timestep (s).

    temp : ndarray(float, ndim=3)
        Snow temperature (K).

    therm_cond_snow : ndarray(float, ndim=3)
        Snow thermal conductivity (W m-1 K-1).

    therm_cond_soil : ndarray(float, ndim=3)
        Soil thermal conductivity (W m-1 K-1).

    heat_flux : ndarray(float, ndim=2)
        Surface heat flux (W m-2).

    heat_cap : ndarray(float, ndim=3)
        Areal heat capacity of snow (J K-1 m-2).

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    num_pixels = len(roi_idxs)
    for idx_num in prange(num_pixels):
        i, j = roi_idxs[idx_num]

        ns = num_layers[i, j]

        if ns > 0:
            temp[:ns, i, j] += heatconduction.temp_change(
                snow_thickness[:ns, i, j],
                timestep,
                temp[:ns, i, j],
                therm_cond_snow[:ns, i, j],
                temp[-1, i, j],
                soil_thickness[0, i, j],
                therm_cond_soil[0, i, j],
                heat_flux[i, j],
                heat_cap[:ns, i, j],
            )


@njit(cache=True, parallel=True)
def _update_layers(
    roi_idxs,
    num_layers,
    min_thickness,
    thickness,
    ice_content,
    liquid_water_content,
    heat_cap,
    temp,
    density,
    depth,
):
    """
    Update snow layers.

    Parameters
    ----------
    roi_idxs : ndarray(int, ndim=2)
        (N, 2)-array specifying the (row, col) indices within the data arrays
        that should be considered.

    num_layers : ndarray(float, ndim=2)
        Number of snow layers.

    min_thickness : ndarray(float, ndim=1)
        Minimum snow layer thicknesses (m).

    thickness : ndarray(float, ndim=3)
        Snow thickness (m).

    ice_content : ndarray(float, ndim=3)
        Ice content of snow (kg m-2).

    liquid_water_content : ndarray(float, ndim=3)
        Liquid water content of snow (kg m-2).

    heat_cap : ndarray(float, ndim=3)
        Areal heat capacity of snow (J K-1 m-2).

    temp : ndarray(float, ndim=3)
        Snow temperature (K).

    density : ndarray(float, ndim=3)
        Snow density (kg m-3).

    depth : ndarray(float, ndim=2)
        Snow depth (m).

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    max_num_layers = len(min_thickness)
    num_layers_prev = num_layers.copy()
    thickness_prev = thickness.copy()
    ice_content_prev = ice_content.copy()
    liquid_water_content_prev = liquid_water_content.copy()
    energy_prev = heat_cap * (temp - c.T0)  # energy content (J m-2)

    num_pixels = len(roi_idxs)
    for idx_num in prange(num_pixels):
        i, j = roi_idxs[idx_num]

        num_layers[i, j] = 0
        thickness[:, i, j] = 0.
        ice_content[:, i, j] = 0.
        liquid_water_content[:, i, j] = 0.
        temp[:, i, j] = c.T0
        density[:, i, j] = np.nan
        internal_energy = np.zeros(max_num_layers)

        if depth[i, j] > 0:
            new_thickness = depth[i, j]

            # Update thicknesses and number of layers
            for k in range(max_num_layers):
                thickness[k, i, j] = min_thickness[k]
                new_thickness -= min_thickness[k]

                if new_thickness <= min_thickness[k] or k == max_num_layers - 1:
                    thickness[k, i, j] += new_thickness
                    break

            # Set thin snow layers to 0 to avoid numerical artifacts
            # TODO should this be done at some other location?
            for k in range(max_num_layers):
                if thickness[k, i, j] < 1e-6:
                    thickness[k, i, j] = 0.

            ns = (thickness[:, i, j] > 0).sum()  # new number of layers
            new_thickness = thickness[0, i, j]
            k_new = 0

            # TODO optimize this loop
            for k_old in range(num_layers_prev[i, j]):
                while True:  # TODO replace with normal loop
                    weight = min(new_thickness / thickness_prev[k_old, i, j], 1.)

                    ice_content[k_new, i, j] += weight * ice_content_prev[k_old, i, j]
                    liquid_water_content[k_new, i, j] += weight * liquid_water_content_prev[k_old, i, j]
                    internal_energy[k_new] += weight * energy_prev[k_old, i, j]

                    if weight == 1.:
                        new_thickness -= thickness_prev[k_old, i, j]
                        break

                    thickness_prev[k_old, i, j] *= 1 - weight
                    ice_content_prev[k_old, i, j] *= 1 - weight
                    liquid_water_content_prev[k_old, i, j] *= 1 - weight
                    energy_prev[k_old, i, j] *= 1 - weight

                    k_new += 1

                    if k_new >= ns:
                        break

                    if weight < 1:
                        new_thickness = thickness[k_new, i, j]

            num_layers[i, j] = ns

            # Update areal heat capacity and snow temperature
            heat_cap[:ns, i, j] = (  # TODO use snow_heat_capacity() for this
                ice_content[:ns, i, j] * c.SPEC_HEAT_CAP_ICE
                + liquid_water_content[:ns, i, j] * c.SPEC_HEAT_CAP_WATER
            )
            temp[:ns, i, j] = c.T0 + internal_energy[:ns] / heat_cap[:ns, i, j]

            # Update density
            density[:ns, i, j] = (
                (liquid_water_content[:ns, i, j] + ice_content[:ns, i, j])
                / thickness[:ns, i, j]
            )
