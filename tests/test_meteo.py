import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import openamundsen.meteo as meteo
import pandas as pd


def test_atmospheric_pressure():
    elevs = [0, 1000, 2000, 3000]
    pressures = [101_325, 89_874.57, 79_495.22, 70_108.54]  # values from https://www.mide.com/air-pressure-at-altitude-calculator
    press_calc = meteo.atmospheric_pressure(elevs)
    assert_allclose(press_calc, pressures, rtol=1e-2)


def test_pressure_to_altitude():
    elevs = np.linspace(0, 3000, 100)
    pressures = meteo.atmospheric_pressure(elevs)
    elevs_calc = meteo.pressure_to_altitude(pressures)
    assert_allclose(elevs, elevs_calc)


def test_latent_heat_of_vaporization():
    assert_allclose(
        meteo.latent_heat_of_vaporization([273.15, 20 + 273.15]),
        [2.501e6, 2.45e6],
        rtol=2e-2,
    )


def test_saturation_vapor_pressure():
    temps = np.array([-40, -20, 0, 20, 40]) + 273.15
    assert_allclose(
        meteo.saturation_vapor_pressure(temps),
        [12.84, 103.24, 611.15, 2339.32, 7384.94],
        rtol=1e-2,
    )


def test_absolute_humidity():
    temps = np.array([30, 20, 10, 0]) + 273.15
    rel_hums = np.array([10, 30, 70, 20])
    abs_hums = np.array([3.0, 5.2, 6.6, 1.0]) * 1e-3  # values from https://www.tis-gdv.de/tis_e/misc/klima-htm/
    vapor_pressures = meteo.vapor_pressure(temps, rel_hums)
    assert_allclose(
        meteo.absolute_humidity(temps, vapor_pressures),
        abs_hums,
        rtol=5e-2,
    )


def test_psychrometric_constant():
    temps = np.array([0, 5, 10, 20]) + 273.15
    psych_consts = np.array([65.5, 65.8, 66.1, 66.8])  # from http://ponce.sdsu.edu/psychrometric_constant.html

    lat_heat_vaps = meteo.latent_heat_of_vaporization(temps)
    atmos_pressures = meteo.atmospheric_pressure(0)
    vapor_pressures = meteo.saturation_vapor_pressure(temps)
    spec_hums = meteo.specific_humidity(atmos_pressures, vapor_pressures)
    spec_heat_caps = meteo.specific_heat_capacity_moist_air(spec_hums)

    assert_allclose(
        meteo.psychrometric_constant(atmos_pressures, spec_heat_caps, lat_heat_vaps),
        psych_consts,
        atol=0.75,
    )


def test_wet_bulb_temperature():
    temps = np.array([5, 10, 20, 30]) + 273.15
    rel_hums = np.array([50, 70, 90, 100])
    wet_bulb_temps = np.array([1.345, 7.393, 18.863, 30]) + 273.15  # from http://www.flycarpet.net/en/PsyOnline
    # TODO also test negative temps

    lat_heat_vaps = meteo.latent_heat_of_vaporization(temps)
    atmos_pressures = meteo.atmospheric_pressure(0)
    vapor_pressures = meteo.vapor_pressure(temps, rel_hums)
    spec_hums = meteo.specific_humidity(atmos_pressures, vapor_pressures)
    spec_heat_caps = meteo.specific_heat_capacity_moist_air(spec_hums)
    psych_consts = meteo.psychrometric_constant(atmos_pressures, spec_heat_caps, lat_heat_vaps)

    assert_allclose(
        meteo.wet_bulb_temperature(temps, rel_hums, vapor_pressures, psych_consts),
        wet_bulb_temps,
        atol=0.1,
    )


def test_relative_humidity():
    temps = np.array([30, 20, 10, 0]) + 273.15
    rel_hums = np.array([10, 30, 70, 20])
    vapor_pressures = meteo.vapor_pressure(temps, rel_hums)
    abs_hums = meteo.absolute_humidity(temps, vapor_pressures)

    assert_allclose(
        rel_hums,
        meteo.relative_humidity(temps, abs_hums),
        atol=1e-6,
    )


def test_dew_point_temperature():
    temps = np.array([-5, 5, 15, 20, 25, 30, 35]) + 273.15
    rel_hums = np.array([30, 50, 50, 70, 90, 100, 30])
    tds = np.array([-19.8, -4.6, 4.65, 14.36, 23.24, 30, 14.84]) + 273.15

    assert_allclose(
        meteo.dew_point_temperature(temps, rel_hums),
        tds,
        atol=0.1,
    )


def test_cloud_fraction():
    temps = [285, 277, 260.3, 268]
    rel_hums = [70, 0.1, 100, 33.77]
    elevs = [100, 500, 3000, 1400]

    cf = meteo.cloud_fraction_from_humidity(
        temps,
        rel_hums,
        elevs,
        -0.0026,
        -0.0044,
    )

    assert cf.min() >= 0
    assert cf.max() <= 1
    assert_allclose(cf[[1, 2]], [0.075224, 0.845268], atol=1e-3)


def test_precipitation_phase():
    temps = np.array([-2, 4, -1.2, 0, 23.4, -24.5, -0.01, 0.01]) + 273.15
    sf = meteo.precipitation_phase(temps, threshold_temp=273.15)
    assert_array_equal(sf, [1, 0, 1, 0, 0, 1, 1, 0])

    temps = np.array([-0.99, -1.01, -1.0, 0., 1., 2.99, 3.01, 3.]) + 273.15
    sf = meteo.precipitation_phase(temps, threshold_temp=1 + 273.15, temp_range=4)
    assert_allclose(sf, [0.9975, 1, 1, 0.75, 0.5, 0.0025, 0, 0], atol=1e-10)


def test_log_wind_profile():
    assert_allclose(meteo.log_wind_profile(1, 10, 2, 0.03), 0.722947444)


def test_wind_uv():
    u = np.array([0, -1, 0, 1, 1, 1, -1, -1])
    v = np.array([-1, 0, 1, 0, 1, -1, 1, -1])
    ws, wd = meteo.wind_from_uv(u, v)
    assert_allclose(ws, [1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    assert_allclose(wd, [0, 90, 180, 270, 225, 315, 135, 45])
    u2, v2 = meteo.wind_to_uv(ws, wd)
    assert_allclose(u, u2, atol=1e-10)
    assert_allclose(v, v2, atol=1e-10)
