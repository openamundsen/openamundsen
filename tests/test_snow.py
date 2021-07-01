import numpy as np
from numpy.testing import assert_allclose, assert_equal
import openamundsen as oa
import pytest
import warnings


single_point_results_all = [
    pytest.lazy_fixture('single_point_results_multilayer'),
    pytest.lazy_fixture('single_point_results_cryolayers'),
]


@pytest.fixture(scope='module')
def point_results_cryolayers(base_config):
    config = base_config.copy()
    config.snow.model = 'cryolayers'
    config.output_data.timeseries.variables.append({'var': 'snow.cold_content'})
    config.output_data.timeseries.variables.append({'var': 'snow.layer_albedo'})
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    return model.point_output.data


@pytest.fixture(scope='function')
def single_point_results_multilayer(base_config_single_point_results):
    return base_config_single_point_results


@pytest.fixture(scope='function')
def single_point_results_cryolayers(point_results_cryolayers):
    return point_results_cryolayers.sel(point='proviantdepot')


def test_default_multilayer(base_config):
    assert base_config.snow.model == 'multilayer'


@pytest.mark.parametrize('ds', single_point_results_all)
def test_swe(ds):
    swe = ds.swe.values
    ice_content = ds.ice_content.values
    liquid_water_content = ds.liquid_water_content.values
    assert np.all(np.isfinite(swe))
    assert swe.min() == 0.
    assert_allclose(
        swe,
        (ice_content + liquid_water_content).sum(axis=1),
        atol=0.1,
    )
    assert np.all(ice_content >= 0.)
    assert np.all(liquid_water_content >= 0.)


def test_swe_max_mean(single_point_results_multilayer, single_point_results_cryolayers):
    swe = single_point_results_multilayer.swe.values
    assert_allclose(swe.max(), 123.95, atol=1)
    assert_allclose(swe.mean(), 67.38, atol=1)

    swe = single_point_results_cryolayers.swe.values
    assert_allclose(swe.max(), 109.6, atol=1)
    assert_allclose(swe.mean(), 56.29, atol=1)


@pytest.mark.parametrize('ds', single_point_results_all)
def test_density(ds):
    swe3d = (ds.ice_content + ds.liquid_water_content).values
    density = ds.snow_density.values
    assert np.all(np.isnan(density[swe3d == 0.]))
    assert np.all(np.isfinite(density[swe3d > 0.]))
    assert not np.any(density < 0.)
    assert not np.any(density > 1000.)


@pytest.mark.parametrize('ds', single_point_results_all)
def test_depth(ds):
    depth = ds.snow_depth.values
    swe = ds.swe.values
    assert np.all(depth[swe == 0.] == 0.)
    assert np.all(depth[swe > 0.] > 0.)
    assert not np.any(depth < 0.)

    swe3d = (ds.ice_content + ds.liquid_water_content).values
    thickness = ds.snow_thickness.values
    assert_allclose(depth, thickness.sum(axis=1), atol=1e-3)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            'invalid value encountered in true_divide',
            RuntimeWarning,
        )
        density_calc = swe3d / thickness

    density = ds.snow_density.values
    assert_allclose(density_calc, density, atol=1.)


@pytest.mark.parametrize('ds', single_point_results_all)
def test_num_layers(ds):
    num_layers = ds.num_snow_layers.values
    thickness = ds.snow_thickness.values
    assert_equal(num_layers, (thickness > 0).sum(axis=1))


@pytest.mark.parametrize('ds', single_point_results_all)
def test_albedo(ds, base_config):
    albedo = ds.snow_albedo.values
    swe = ds.swe.values
    min_albedo = base_config.snow.albedo.min
    max_albedo = base_config.snow.albedo.max
    pos_snow = swe > 0.
    assert np.all(albedo[pos_snow] >= min_albedo)
    assert np.all(albedo[pos_snow] <= max_albedo)
    assert np.all(np.isnan(albedo[~pos_snow]))


@pytest.mark.parametrize('ds', single_point_results_all)
def test_melt(ds):
    assert np.all(ds.melt >= 0)


@pytest.mark.parametrize('ds', single_point_results_all)
def test_runoff(ds):
    assert np.all(ds.runoff >= 0)


def test_cryolayers_layer_albedo(single_point_results_cryolayers, base_config):
    ds = single_point_results_cryolayers
    layer_albedo = ds.layer_albedo.values
    swe3d = (ds.ice_content + ds.liquid_water_content).values
    min_albedo = base_config.snow.albedo.min
    max_albedo = base_config.snow.albedo.max
    pos_snow = swe3d > 0.
    assert np.all(layer_albedo[pos_snow] >= min_albedo)
    assert np.all(layer_albedo[pos_snow] <= max_albedo)
    assert np.all(np.isnan(layer_albedo[~pos_snow]))


def test_cryolayers_cold_content(single_point_results_cryolayers, base_config):
    ds = single_point_results_cryolayers
    cc = ds.cold_content.values
    ic = ds.ice_content.values
    lwc = ds.liquid_water_content.values
    melt = ds.melt.values
    sublimation = ds.sublimation.values
    swe3d = (ds.ice_content + ds.liquid_water_content).values
    cold_holding_capacity = base_config.snow.cryolayers.cold_holding_capacity
    pos_snow = swe3d > 0
    assert np.all(cc >= 0.)
    assert cc.max() > 0.
    assert np.all(cc[~pos_snow] == 0.)

    # Check if the CC is <= swe * cold_colding_capacity (add melt and sublimation because the
    # maximum CC is calculated before reducing SWE by melt and sublimation)
    assert np.all(
        (
            ((ic + lwc).sum(axis=1) + melt + sublimation) * cold_holding_capacity
            - cc.sum(axis=1)
        ) >= -0.1
    )
