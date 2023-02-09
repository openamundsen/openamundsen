import numpy as np
import openamundsen.constants as c
import openamundsen.modules.radiation as rad


def _clear_sky_shortwave_irradiance(
    day_angle,
    sun_vec,
    shadows,
    dem,
    svf,
    normal_vec,
    atmos_press,
    precipitable_water,
    ground_albedo,
    roi=None,
    ozone_layer_thickness=0.0035,
    atmospheric_visibility=25000.,
    single_scattering_albedo=0.9,
    clear_sky_albedo=0.0685,
):
    """
    Calculate potential direct and diffuse shortwave irradiance for a given DEM
    and sun position after Corripio (2002).

    Parameters
    ----------
    day_angle : float
        Day angle (radians).

    sun_vec : array-like
        Solar vector.

    shadows : ndarray
        Array containing 1 for illuminated and 0 for shadowed areas.

    dem : ndarray
        Elevation (m).

    svf : ndarray
        Sky view factor.

    normal_vec : ndarray with dimensions (3, *dem.shape)
        Unit vector perpendicular to the surface.

    atmos_press : ndarray
        Atmospheric pressure (Pa).

    precipitable_water : ndarray
        Precipitable water (kg m-2).

    ground_albedo : float
        Mean surface albedo.

    roi : ndarray, default None
        Boolean array specifying the region of interest.

    ozone_layer_thickness : float, default 0.0035
        Ozone layer thickness (m).

    visibility : float, default 25000
        Atmospheric visibility (m).

    single_scattering_albedo : float, default 0.9
        Single scattering albedo.

    clear_sky_albedo : float, default 0.0685
        Clear sky albedo.

    Returns
    -------
    dir_irr : ndarray
        Potential direct irradiance (W m-2).

    diff_irr : ndarray
        Potential diffuse irradiance (W m-2).

    References
    ----------
    .. [1] Corripio, J. G. (2002). Modelling the energy balance of high
       altitude glacierised basins in the Central Andes. PhD thesis, University of
       Edinburgh.
    """
    zenith_angle = np.arccos(sun_vec[2])

    if roi is None:
        roi = np.ones(dem.shape, dtype=bool)

    # Eccentricity correction (Corripio, 2002)
    ecc_corr = (
        1.000110
        + 0.034221 * np.cos(day_angle)
        + 0.001280 * np.sin(day_angle)
        + 0.000719 * np.cos(2 * day_angle)
        + 0.000077 * np.sin(2 * day_angle)
    )

    # Angle of incidence of the sun on inclined surfaces (dot product of the
    # unit vector normal to the surface and the unit vector in the direction
    # of the sun)
    aoi = (
        sun_vec[0] * normal_vec[0, :] +
        sun_vec[1] * (-normal_vec[1, :]) +
        sun_vec[2] * normal_vec[2, :]
    )
    aoi = aoi.clip(min=0)  # shadows due to steepness of the slope (sun "behind")

    # Relative optical air mass for standard pressure (Iqbal 1983)
    rel_opt_air_mass = 1. / (
        np.cos(zenith_angle)
        + 0.15 * (93.885 - np.rad2deg(zenith_angle))**(-1.253)
    ).clip(max=14.1952)  # function is only valid for zenith angles up to 87°

    # Relative optical air mass pressure corrected
    rel_opt_air_mass_press_corr = rel_opt_air_mass * atmos_press / c.STANDARD_ATMOSPHERE

    ts = _transmittances(
        dem,
        atmos_press,
        precipitable_water,
        rel_opt_air_mass,
        ozone_layer_thickness,
        atmospheric_visibility,
        single_scattering_albedo,
    )

    trans_dir = (
        ts['trans_rayleigh']
        * ts['trans_ozone']
        * ts['trans_gases']
        * ts['trans_vapor']
        * ts['trans_aerosols']
        + ts['elev_corr']
    )
    trans_dir = np.clip(trans_dir, 0, 1)

    top_atmosphere_rad = c.SOLAR_CONSTANT * ecc_corr  # top of atmosphere radiation
    pot_irr = top_atmosphere_rad * aoi * shadows  # potential irradiance including shadows

    dir_irr = 0.9751 * pot_irr * trans_dir

    # Rayleigh-scattered diffuse irradiance
    diff_irr_rayleigh = (
        0.79
        * top_atmosphere_rad * np.cos(zenith_angle)
        * ts['trans_ozone']
        * ts['trans_gases']
        * ts['trans_vapor']
        * ts['trans_aerosol_abs']
        * 0.5 * (1 - ts['trans_rayleigh'])
        / (1 - rel_opt_air_mass_press_corr + rel_opt_air_mass_press_corr**1.02)
    )

    # Aerosol-scattered diffuse irradiance
    scattering_ratio = (
        -0.2562 * zenith_angle**2
        * 0.1409 * zenith_angle + 0.9067
    )
    trans_aerosol_ratio = ts['trans_aerosols'] / ts['trans_aerosol_abs']
    diff_irr_aerosols = (
        0.79
        * top_atmosphere_rad * np.cos(zenith_angle)
        * ts['trans_ozone']
        * ts['trans_gases']
        * ts['trans_vapor']
        * ts['trans_aerosol_abs']
        * scattering_ratio
        * (1 - trans_aerosol_ratio)
        / (1 - rel_opt_air_mass_press_corr + rel_opt_air_mass_press_corr**1.02)
    )

    # Multiply-reflected irradiance between the earth and the atmosphere
    atmospheric_albedo = (
        clear_sky_albedo
        + (1 - scattering_ratio)
        * (1 - trans_aerosol_ratio)
    )
    diff_irr_atmos_ref = (
        (
            dir_irr * np.cos(zenith_angle)
            + diff_irr_rayleigh
            + diff_irr_aerosols
        )
        * ground_albedo * atmospheric_albedo
        / (1 - ground_albedo * atmospheric_albedo)
    )

    diff_irr = diff_irr_rayleigh + diff_irr_aerosols + diff_irr_atmos_ref

    # Diffuse reflected radiation from surrounding slopes
    sun_frac = shadows[roi].mean()
    pos_sun = shadows[roi] == 1
    if pos_sun.sum() > 0:
        mean_sun_aoi = np.nanmean(aoi[roi][pos_sun])
    else:
        mean_sun_aoi = 0.
    diff_irr_terrain_ref = (
        (1 - svf) * (
            pot_irr
            * trans_dir
            * sun_frac
            * mean_sun_aoi
            + diff_irr
        ) * ground_albedo
    )

    # Diffuse irradiance corrected for topography
    diff_irr_topo_corr = diff_irr * svf + diff_irr_terrain_ref

    return dir_irr, diff_irr_topo_corr


def _transmittances(
    elev,
    atmos_press,
    precipitable_water,
    rel_opt_air_mass,
    ozone_layer_thickness,
    visibility,
    single_scattering_albedo,
):
    """
    Calculate atmospheric transmittances after Corripio (2002).

    Parameters
    ----------
    elev : ndarray
        Elevation (m).

    atmos_press : ndarray
        Atmospheric pressure (Pa).

    precipitable_water : ndarray
        Precipitable water (kg m-2).

    rel_opt_air_mass : ndarray
        Relative optical air mass for standard pressure.

    ozone_layer_thickness : float
        Ozone layer thickness (m).

    visibility : float
        Atmospheric visibility (m).

    single_scattering_albedo : float
        Single scattering albedo.

    Returns
    -------
    transmittances : dict
        Dictionary containing the following fields:
        - 'trans_rayleigh': Rayleigh scattering
        - 'trans_ozone': transmittance by ozone
        - 'trans_gases': transmittance by uniformy mixed gases
        - 'trans_vapor': transmittance by water vapor
        - 'trans_aerosols': transmittance by aerosols
        - 'trans_aerosol_abs': transmittance of direct radiation due to aerosol
          absorptance
        - 'elev_corr': correction of atmospheric transmittances for altitude
          after Bintanja (1996)

    References
    ----------
    .. [1] Corripio, J. G. (2002). Modelling the energy balance of high
       altitude glacierised basins in the Central Andes. PhD thesis, University of
       Edinburgh.
    """
    ozone_layer_thickness_cm = ozone_layer_thickness * 100.
    visibility_km = visibility / 1000.

    # Relative optical air mass pressure corrected (eq. (3.10))
    rel_opt_air_mass_press_corr = rel_opt_air_mass * atmos_press / c.STANDARD_ATMOSPHERE

    # Rayleigh scattering
    trans_rayleigh = (
        np.exp(-0.0903 * rel_opt_air_mass_press_corr**0.84)
        * (1 + rel_opt_air_mass_press_corr - rel_opt_air_mass_press_corr**1.01)
    )

    # Transmittance by ozone (constant for the entire area) (eq. (3.13))
    lm = ozone_layer_thickness_cm * rel_opt_air_mass
    trans_ozone = (
        1 - (
            0.1611 * lm * (1 + 139.48 * lm)**(-0.3035)  # -0.3035 seems to be the correct value (in eq. (3.13) in [1] the value is -0.035)
            - 0.002715 * lm * 1. / (1 + 0.044 * lm + 3e-4 * lm**2)
        )
    )

    # Transmittance by uniformly mixed gases (eq. (3.15))
    trans_gases = np.exp(-0.0127 * rel_opt_air_mass_press_corr**0.26)

    # Transmittance by water vapor (eq. (3.16))
    # Note: eq. (3.16) in [1] is missing a closing parenthesis; the exponent 0.6828 should apply to
    # the entire (1 + 79.034 * w * m_r) term, see also e.g. eq. (13) in
    # https://doi.org/10.1029/2003JD003973
    precipitable_water_gcm2 = precipitable_water / 10  # kg m-2 (= mm) to g cm-2 (= cm)
    wm = precipitable_water_gcm2 * rel_opt_air_mass
    trans_vapor = (
        1
        - 2.4959 * wm
        * 1. / ((1 + 79.034 * wm)**0.6828 + 6.385 * wm)
    )

    # Transmittance by aerosols (eq. (3.17))
    trans_aerosols = (0.97 - 1.265 * visibility_km**(-0.66))**(rel_opt_air_mass_press_corr**0.9)

    # Correction of atmospheric transmittances for altitude (eq. (3.8))
    elev_corr = 2.2e-5 * elev.clip(max=3000)

    # Transmittance of direct radiation due to aerosol absorptance (eq. (3.19))
    trans_aerosol_abs = (
        1
        - (1 - single_scattering_albedo)
        * (1 - rel_opt_air_mass_press_corr + rel_opt_air_mass_press_corr**1.06)
        * (1 - trans_aerosols)
    )

    return {
        'trans_rayleigh': trans_rayleigh,
        'trans_ozone': trans_ozone,
        'trans_gases': trans_gases,
        'trans_vapor': trans_vapor,
        'trans_aerosols': trans_aerosols,
        'trans_aerosol_abs': trans_aerosol_abs,
        'elev_corr': elev_corr,
    }
