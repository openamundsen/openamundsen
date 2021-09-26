from .compare import compare_datasets
from .conftest import base_config
import numpy as np
import openamundsen as oa
from pathlib import Path
import pytest
import tempfile


@pytest.fixture(scope='session')
def canopy_run():
    config = base_config()
    config.start_date = '2019-10-01'
    config.end_date = '2020-05-31'

    model = oa.OpenAmundsen(config)
    model.initialize()

    roi_xs = model.grid.X.flat[model.grid.roi_idxs_flat]
    roi_ys = model.grid.Y.flat[model.grid.roi_idxs_flat]

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)

        for p in Path(config.input_data.grids.dir).glob('*.asc'):
            (tmp_path / p.name).symlink_to(p)

        config.input_data.grids.dir = str(temp_dir)

        soil = np.zeros(model.grid.shape, dtype=int)
        lc = np.zeros(model.grid.shape, dtype=int)

        soil[:] = 5
        lccs = model.config.land_cover.classes.keys()
        config.output_data.timeseries.points = []
        for lcc_num, lcc in enumerate(lccs):
            config.output_data.timeseries.points.append({
                'x': float(roi_xs[lcc_num]),
                'y': float(roi_ys[lcc_num]),
            })
            lc.flat[model.grid.roi_idxs_flat[lcc_num]] = lcc

        rio_meta = {'driver': 'AAIGrid'}
        oa.fileio.write_raster_file(
            oa.util.raster_filename('soil', config),
            soil.astype(np.int32),
            model.grid.transform,
            **rio_meta,
        )
        oa.fileio.write_raster_file(
            oa.util.raster_filename('lc', config),
            lc.astype(np.int32),
            model.grid.transform,
            **rio_meta,
        )

        config.canopy.enabled = True
        config.output_data.timeseries.add_default_points = False
        config.output_data.timeseries.variables = [
            {'var': 'snow.canopy_intercepted_load'},
            {'var': 'snow.canopy_intercepted_snowfall'},
            {'var': 'snow.canopy_sublimation'},
            {'var': 'snow.canopy_melt'},
            {'var': 'meteo.top_canopy_temp'},
        ]

        model = oa.OpenAmundsen(config)
        model.initialize()
        model.run()

    return model


@pytest.mark.slow
def test_canopy_snow(canopy_run):
    model = canopy_run
    ds = model.point_output.data
    lccs = model.config.land_cover.classes.keys()

    for lcc_num, lcc in enumerate(lccs):
        lcc_params = model.config.land_cover.classes[lcc]
        is_forest = lcc_params.get('is_forest', False)

        ds_lcc = ds.isel(point=lcc_num)

        if is_forest:
            assert np.all(ds_lcc.canopy_intercepted_load >= 0.)
            assert np.all(ds_lcc.canopy_intercepted_snowfall >= 0.)
            assert np.all(ds_lcc.canopy_sublimation >= 0.)
            assert np.all(ds_lcc.canopy_melt >= 0.)
            assert ds_lcc.canopy_intercepted_load.max() > 0.
            assert ds_lcc.canopy_intercepted_snowfall.max() > 0.
            assert ds_lcc.canopy_sublimation.max() > 0.
            assert ds_lcc.canopy_melt.max() > 0.
        else:
            assert np.all(np.isnan(ds_lcc.canopy_intercepted_load))
            assert np.all(np.isnan(ds_lcc.canopy_intercepted_snowfall))
            assert np.all(np.isnan(ds_lcc.canopy_sublimation))
            assert np.all(np.isnan(ds_lcc.canopy_melt))


@pytest.mark.slow
def test_canopy_meteorology(canopy_run):
    model = canopy_run
    ds = model.point_output.data
    lccs = model.config.land_cover.classes.keys()

    for lcc_num, lcc in enumerate(lccs):
        lcc_params = model.config.land_cover.classes[lcc]
        is_forest = lcc_params.get('is_forest', False)

        if is_forest:
            ds_lcc = ds.isel(point=lcc_num)
            assert np.all(ds_lcc.rel_hum >= 0.)
            assert np.all(ds_lcc.rel_hum <= 100.)
            assert np.all(ds_lcc.wind_speed >= 0.)
            assert np.all(ds_lcc.sw_in >= 0.)
            assert np.all(ds_lcc.lw_in >= 0.)
            assert np.all(np.abs(ds_lcc.temp - ds_lcc.top_canopy_temp) <= 10.)


def test_no_forest():
    config = base_config()
    config.start_date = '2019-10-01'
    config.end_date = '2019-10-01'

    model = oa.OpenAmundsen(config)
    model.initialize()

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)

        for p in Path(config.input_data.grids.dir).glob('*.asc'):
            (tmp_path / p.name).symlink_to(p)

        config.input_data.grids.dir = str(temp_dir)

        lc = np.zeros(model.grid.shape, dtype=int)
        rio_meta = {'driver': 'AAIGrid'}
        oa.fileio.write_raster_file(
            oa.util.raster_filename('lc', config),
            lc.astype(np.int32),
            model.grid.transform,
            **rio_meta,
        )

        config.canopy.enabled = True
        model = oa.OpenAmundsen(config)
        model.initialize()
        model.run()


@pytest.mark.slow
@pytest.mark.comparison
def test_compare_canopy(canopy_run):
    compare_datasets(
        'canopy_point',
        canopy_run.point_output.data,
        point='point5',  # coniferous forest
    )
