from .compare import compare_datasets
from .conftest import base_config
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import openamundsen as oa
import pytest
import warnings


single_point_results_all = [
    pytest.lazy_fixture('single_point_results_multilayer'),
    pytest.lazy_fixture('single_point_results_cryolayers'),
]


def base_config_snow():
    config = base_config()
    config.start_date = '2019-10-01'
    config.end_date = '2020-05-31'
    config.output_data.timeseries.variables = [
        {'var': 'snow.num_layers', 'name': 'num_snow_layers'},
        {'var': 'snow.albedo', 'name': 'snow_albedo'},
    ]
    return config


@pytest.fixture(scope='session')
def multilayer_run():
    config = base_config_snow()
    config.snow.model = 'multilayer'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    return model


@pytest.fixture(scope='session')
def cryolayer_run():
    config = base_config_snow()
    config.snow.model = 'cryolayers'
    config.output_data.timeseries.variables.append({'var': 'snow.cold_content'})
    config.output_data.timeseries.variables.append({'var': 'snow.layer_albedo'})
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    return model


@pytest.fixture(scope='module')
def point_results_cryolayers():
    config = base_config()
    config.snow.model = 'cryolayers'
    config.output_data.timeseries.variables.append({'var': 'snow.cold_content'})
    config.output_data.timeseries.variables.append({'var': 'snow.layer_albedo'})
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    return model.point_output.data


@pytest.fixture(scope='function')
def single_point_results_multilayer(multilayer_run):
    return multilayer_run.point_output.data.sel(point='proviantdepot')


@pytest.fixture(scope='function')
def single_point_results_cryolayers(cryolayer_run):
    return cryolayer_run.point_output.data.sel(point='proviantdepot')


def test_default_multilayer():
    config = base_config_snow()
    assert config.snow.model == 'multilayer'


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.parametrize('ds', single_point_results_all)
def test_density(ds):
    swe3d = (ds.ice_content + ds.liquid_water_content).values
    density = ds.snow_density.values
    assert np.all(np.isnan(density[swe3d == 0.]))
    assert np.all(np.isfinite(density[swe3d > 0.]))
    assert not np.any(density < 0.)
    assert not np.any(density > 1000.)


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.parametrize('ds', single_point_results_all)
def test_num_layers(ds):
    num_layers = ds.num_snow_layers.values
    thickness = ds.snow_thickness.values
    assert_equal(num_layers, (thickness > 0).sum(axis=1))


@pytest.mark.slow
@pytest.mark.parametrize('ds', single_point_results_all)
def test_albedo(ds):
    config = base_config()
    albedo = ds.snow_albedo.values
    swe = ds.swe.values
    min_albedo = config.snow.albedo.min
    max_albedo = config.snow.albedo.max
    pos_snow = swe > 0.
    assert np.all(albedo[pos_snow] >= min_albedo)
    assert np.all(albedo[pos_snow] <= max_albedo)
    assert np.all(np.isnan(albedo[~pos_snow]))


@pytest.mark.slow
@pytest.mark.parametrize('ds', single_point_results_all)
def test_melt(ds):
    assert np.all(ds.melt >= 0)


@pytest.mark.slow
@pytest.mark.parametrize('ds', single_point_results_all)
def test_runoff(ds):
    assert np.all(ds.runoff >= 0)


@pytest.mark.slow
@pytest.mark.parametrize('ds', single_point_results_all)
def test_refreezing(ds):
    assert np.all(ds.refreezing >= 0)


@pytest.mark.slow
def test_cryolayers_layer_albedo(single_point_results_cryolayers):
    config = base_config()
    ds = single_point_results_cryolayers
    layer_albedo = ds.layer_albedo.values
    swe3d = (ds.ice_content + ds.liquid_water_content).values
    min_albedo = config.snow.albedo.min
    max_albedo = config.snow.albedo.max
    pos_snow = swe3d > 0.
    assert np.all(layer_albedo[pos_snow] >= min_albedo)
    assert np.all(layer_albedo[pos_snow] <= max_albedo)
    assert np.all(np.isnan(layer_albedo[~pos_snow]))


@pytest.mark.slow
def test_cryolayers_cold_content(single_point_results_cryolayers):
    config = base_config()
    ds = single_point_results_cryolayers
    cc = ds.cold_content.values
    ic = ds.ice_content.values
    lwc = ds.liquid_water_content.values
    melt = ds.melt.values
    sublimation = ds.sublimation.values
    swe3d = (ds.ice_content + ds.liquid_water_content).values
    cold_holding_capacity = config.snow.cryolayers.cold_holding_capacity
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


@pytest.mark.slow
@pytest.mark.parametrize('method', ['temperature_index', 'enhanced_temperature_index'])
def test_temperature_index(method):
    config = base_config_snow()
    config.snow.model = 'cryolayers'
    config.timestep = 'D'
    config.output_data.timeseries.variables = [{'var': 'snow.cold_content'}]
    config.snow.melt.method = method

    if method == 'temperature_index':
        config.snow.melt.degree_day_factor = 5.
    elif method == 'enhanced_temperature_index':
        config.snow.melt.degree_day_factor = 3.
        config.snow.melt.albedo_factor = 0.1

    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()

    ds = model.point_output.data.sel(point='proviantdepot')
    assert np.all(ds.melt >= 0)
    assert np.any(ds.melt > 0)
    assert np.all(ds.melt[ds.temp <= config.snow.melt.degree_day_factor] == 0.)
    melt_potential = (
        ds.ice_content.sum('snow_layer').shift(time=1)
        + ds.cold_content.sum('snow_layer').shift(time=1)
        + ds.snowfall
    )
    assert np.all(ds.melt <= melt_potential.fillna(np.inf) + 1e-6)


@pytest.mark.slow
@pytest.mark.comparison
def test_compare_multilayer(multilayer_run):
    compare_datasets(
        'snow_multilayer_point',
        multilayer_run.point_output.data,
        point='proviantdepot',
    )


@pytest.mark.slow
@pytest.mark.comparison
def test_compare_cryolayers(cryolayer_run):
    compare_datasets(
        'snow_cryolayers_point',
        cryolayer_run.point_output.data,
        point='proviantdepot',
    )
