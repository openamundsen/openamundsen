from numba import njit, prange
import numpy as np
from openamundsen import (
    constants as c,
)


def albedo(model, pos=None):
    """
    Update snow albedo.
    Albedo aging is calculated using an exponential decay function with
    different timescales for cold and melting snow [1-3].
    For fresh snow, albedo is either reset to the maximum value when snowfall
    exceeds a preset threshold [2], or continuously increased whenever snow is
    falling with a rate depending on snowfall rate [3-4].

    References
    ----------
    .. [1] Snow Hydrology: Summary Report of the Snow Investigations. Published
       by the North Pacific Division, Corps of Engineers, U.S. Army, Portland,
       Oregon, 1956.  437 pages, 70 pages of plates, maps and figs., 27 cm.
       https://doi.org/10.3189/S0022143000024503

    .. [2] Hanzer, F., Helfricht, K., Marke, T., & Strasser, U. (2016).
       Multilevel spatiotemporal validation of snow/ice mass balance and runoff
       modeling in glacierized catchments. The Cryosphere, 10(4), 1859–1881.
       https://doi.org/10.5194/tc-10-1859-2016

    .. [3] Essery, R., Morin, S., Lejeune, Y., & B Ménard, C. (2013). A
       comparison of 1701 snow models using observations from an alpine site.
       Advances in Water Resources, 55, 131–148.
       https://doi.org/10.1016/j.advwatres.2012.07.013

    .. [4] Dutra, E., Balsamo, G., Viterbo, P., Miranda, P., Beljaars, A.,
       Schär, C., & Elder, K. (2009). New snow scheme in HTESSEL: description and
       offline validation. ECMWF. https://doi.org/10.21957/98x9mrv1y
    """
    albedo_config = model.config.snow.albedo
    s = model.state

    if pos is None:
        pos = model.grid.roi

    if albedo_config.decay_timescale_determination_temperature == 'air':
        temp = s.meteo.temp[pos]
    elif albedo_config.decay_timescale_determination_temperature == 'surface':
        temp = s.surface.temp[pos]

    # Albedo decay
    decay_timescale = np.where(
        temp >= c.T0,
        c.SECONDS_PER_HOUR * albedo_config.melting_snow_decay_timescale,
        c.SECONDS_PER_HOUR * albedo_config.cold_snow_decay_timescale,
    )
    new_albedo = (
        albedo_config.min
        + (s.snow.albedo[pos] - albedo_config.min)
        * np.exp(-1 / decay_timescale * model.timestep)
    )

    # Albedo refresh
    snowfall_rate = s.meteo.snowfall[pos] / model.timestep
    refresh_snowfall_rate = albedo_config.refresh_snowfall / c.SECONDS_PER_HOUR  # kg m-2 s-1
    if albedo_config.refresh_method == 'binary':
        new_albedo[snowfall_rate >= refresh_snowfall_rate] = albedo_config.max
    elif albedo_config.refresh_method == 'continuous':
        new_albedo += (
            (albedo_config.max - new_albedo)
            * np.minimum(1, snowfall_rate / refresh_snowfall_rate)  # minimum() to clip values at albedo_config.max
        )

    s.snow.albedo[pos] = new_albedo


def fresh_snow_density(temp):
    """
    Calculate fresh snow density based on the parameterization by [1] (eq. (4.22)).

    Parameters
    ----------
    temp : numeric
        Wet-bulb temperature (K) in the original formulation by [1]. Can be
        substituted by air temperature if not available.

    Returns
    -------
    rho : numeric
        Fresh snow density (kg m-3).

    References
    ----------
    .. [1] Anderson, E. A. (1976). A point energy and mass balance model of a
       snow cover (NOAA Technical Report NWS 19, pp. 1–172). NOAA.
       https://repository.library.noaa.gov/view/noaa/6392
    """
    min_temp = c.T0 - 15.
    temp = np.array(temp).clip(min_temp)  # the parameterization is only valid for temperatures >= -15 °C
    rho = 50 + 1.7 * (temp - min_temp)**1.5
    rho[np.isnan(rho)] = 100.  # if temperature is not available, set density to 100 kg m-3
    return rho


def compaction(model):
    snow = model.state.snow

    if model.config.snow.compaction.method == 'anderson':
        _compaction_anderson(
            model.grid.roi_idxs,
            model.timestep,
            snow.num_layers,
            snow.thickness,
            snow.ice_content,
            snow.liquid_water_content,
            snow.density,
            model.state.meteo.temp,
        )
    elif model.config.snow.compaction.method == 'empirical':
        timescale = c.SECONDS_PER_HOUR * model.config.snow.compaction.timescale  # snow compaction timescale (s)
        _compaction_empirical(
            model.grid.roi_idxs,
            model.timestep,
            timescale,
            model.config.snow.compaction.max_cold_density,
            model.config.snow.compaction.max_melting_density,
            snow.num_layers,
            snow.temp,
            snow.thickness,
            snow.ice_content,
            snow.liquid_water_content,
            snow.density,
        )
    else:
        raise NotImplementedError


@njit(cache=True, parallel=True)
def _compaction_anderson(
    roi_idxs,
    timestep,
    num_layers,
    thickness,
    ice_content,
    liquid_water_content,
    density,
    air_temp,
):
    """
    Calculate snow compaction following [1].
    Parameter names (c1...c7) are changed compared to the original
    formulation, follwing [2].

    Parameters
    ----------
    roi_idxs : ndarray(int, ndim=2)
        (N, 2)-array specifying the (row, col) indices within the data arrays
        that should be considered.

    timestep : float
        Model timestep (s).

    num_layers : ndarray(float, ndim=2)
        Number of snow layers.

    thickness : ndarray(float, ndim=3)
        Snow thickness (m).

    ice_content : ndarray(float, ndim=3)
        Ice content of snow (kg m-2).

    liquid_water_content : ndarray(float, ndim=3)
        Liquid water content of snow (kg m-2).

    density : ndarray(float, ndim=3)
        Snow density (kg m-3).

    air_temp : ndarray(float, ndim=2)
        Air temperature (K).

    References
    ----------
    .. [1] Anderson, E. A. (1976). A point energy and mass balance model of a
       snow cover (NOAA Technical Report NWS 19, pp. 1–172). NOAA.
       https://repository.library.noaa.gov/view/noaa/6392

    .. [2] Koivusalo, H., Heikinheimo, M., & Karvonen, T. (2001). Test of a
       simple two-layer parameterisation to simulate the energy balance and
       temperature of a snow pack. Theoretical and Applied Climatology, 70(1–4),
       65–79. https://doi.org/10.1007/s007040170006
    """
    timestep_h = timestep / c.SECONDS_PER_HOUR  # model timestep (h)

    c1 = 0.001  # m2 kg-1 h-1
    c2 = 0.08  # K-1
    c3 = 0.021  # m3 kg-1
    c5 = 0.04  # K-1
    c7 = 0.046  # m3 kg-1
    rho_d = 150.  # kg m-3

    num_pixels = len(roi_idxs)
    for idx_num in prange(num_pixels):
        i, j = roi_idxs[idx_num]
        load = 0.  # snow load (mass of layers above + 50% of the current layer) (kg m-2)

        for k in range(num_layers[i, j]):
            if thickness[k, i, j] > 0:  # TODO is this necessary?
                cur_layer_mass = ice_content[k, i, j] + liquid_water_content[k, i, j]
                load += cur_layer_mass / 2.

                # TODO rather update density in update_layers()?
                density[k, i, j] = (ice_content[k, i, j] + liquid_water_content[k, i, j]) / thickness[k, i, j]

                if density[k, i, j] > rho_d:
                    c6 = np.exp(-c7 * (density[k, i, j] - rho_d))
                else:
                    c6 = 1.

                # Parameter c4 has an enhancement factor of 2 for wet snow
                if liquid_water_content[k, i, j] > 0:
                    c4 = 0.02  # h-1
                else:
                    c4 = 0.01  # h-1

                # Densification due to snow load (eq. (3.29) in [1], eq. (12) in [2])
                dens_compact = c1 * load * np.exp(-c2 * (c.T0 - air_temp[i, j]) - c3 * density[k, i, j])

                # Densification due to destructive metamorphism (eq. (3.31) in [1], eq. (13) in [2])
                dens_metamorph = c4 * np.exp(-c5 * (c.T0 - air_temp[i, j])) * c6

                densification_rate = density[k, i, j] * (dens_compact + dens_metamorph)
                if np.isnan(densification_rate):
                    densification_rate = 0.

                density[k, i, j] += densification_rate * timestep_h
                thickness[k, i, j] = (ice_content[k, i, j] + liquid_water_content[k, i, j]) / density[k, i, j]

                # Update snow load for the next layer
                load += cur_layer_mass / 2.


@njit(cache=True, parallel=True)
def _compaction_empirical(
    roi_idxs,
    timestep,
    timescale,
    max_cold_density,
    max_melting_density,
    num_layers,
    temp,
    thickness,
    ice_content,
    liquid_water_content,
    density,
):
    """
    Calculate snow compaction following [1].

    Parameters
    ----------
    roi_idxs : ndarray(int, ndim=2)
        (N, 2)-array specifying the (row, col) indices within the data arrays
        that should be considered.

    timestep : float
        Model timestep (s).

    timescale : float
        Snow compaction timescale (s).

    max_cold_density : float
        Maximum density for cold (T < 0 °C) snow (kg m-2).

    max_melting_density : float
        Maximum density for melting snow (kg m-2).

    num_layers : ndarray(float, ndim=2)
        Number of snow layers.

    temp : ndarray(float, ndim=3)
        Snow temperature (K).

    thickness : ndarray(float, ndim=3)
        Snow thickness (m).

    ice_content : ndarray(float, ndim=3)
        Ice content of snow (kg m-2).

    liquid_water_content : ndarray(float, ndim=3)
        Liquid water content of snow (kg m-2).

    density : ndarray(float, ndim=3)
        Snow density (kg m-3).

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    num_pixels = len(roi_idxs)
    for idx_num in prange(num_pixels):
        i, j = roi_idxs[idx_num]

        for k in range(num_layers[i, j]):
            if thickness[k, i, j] > 0:  # TODO is this necessary?
                # TODO rather update density in update_layers()?
                density[k, i, j] = (ice_content[k, i, j] + liquid_water_content[k, i, j]) / thickness[k, i, j]

                if temp[k, i, j] < c.T0:
                    max_density = max_cold_density
                else:
                    max_density = max_melting_density

                if density[k, i, j] < max_density:
                    # Where the maximum density is already reached, do not increase it anymore
                    # but to avoid jumps do not actually clip the values (because snow might
                    # switch between "cold" and "melting" from one timestep to another)
                    density[k, i, j] = (
                        max_density
                        + (density[k, i, j] - max_density)
                        * np.exp(-timestep / timescale)
                    )

                    thickness[k, i, j] = (ice_content[k, i, j] + liquid_water_content[k, i, j]) / density[k, i, j]


def snow_properties(model):
    """
    Update snow properties (depth, SWE, snow cover fraction, heat capacity).

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    snow = model.state.snow

    snow.depth[:] = snow.thickness.sum(axis=0)
    snow.swe[:] = (snow.ice_content + snow.liquid_water_content).sum(axis=0)

    # Snow cover fraction (eq. (13))
    snow.area_fraction[:] = np.tanh(snow.depth / model.config.snow.snow_cover_fraction_depth_scale)

    # Areal heat capacity of snow (eq. (9))
    snow.heat_cap[:] = (
        snow.ice_content * c.SPEC_HEAT_CAP_ICE
        + snow.liquid_water_content * c.SPEC_HEAT_CAP_WATER
    )


def max_liquid_water_content(model):
    """
    Calculate the maximum liquid water content of snow.

    The following options for calculating the maximum liquid water content are
    supported:
        - 'pore_volume_fraction': maximum LWC is set to a fraction of the snow
          pore volume following [1].
        - 'mass_fraction': maximum LWC is set to a fraction of the snow
          ice mass.

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.

    Returns
    -------
    max_lwc : ndarray(float, ndim=3)
        Maximum liquid water content (kg m-2).

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    s = model.state
    method = model.config.snow.liquid_water_content.method

    if method == 'pore_volume_fraction':
        pos = s.snow.thickness > 0.
        porosity = np.zeros(s.snow.ice_content.shape)

        porosity[pos] = 1 - s.snow.ice_content[pos] / (c.ICE_DENSITY * s.snow.thickness[pos])  # eq. (27)

        max_lwc = (  # eq. (28)
            c.WATER_DENSITY
            * s.snow.thickness
            * porosity
            * model.config.snow.liquid_water_content.max
        )
    elif method == 'mass_fraction':
        max_lwc = model.config.snow.liquid_water_content.max * s.snow.ice_content

    return max_lwc
