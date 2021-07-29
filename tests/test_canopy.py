from .compare import point_comparison
from .conftest import base_config
import numpy as np
from numpy.testing import assert_allclose
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

        shape = (model.grid.rows, model.grid.cols)
        soil = np.zeros(shape, dtype=int)
        lc = np.zeros(shape, dtype=int)

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
        ]

        model = oa.OpenAmundsen(config)
        model.initialize()
        model.run()

    return model


@pytest.mark.slow
def test_canopy(canopy_run):
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
@pytest.mark.report
def test_plot_canopy(
        canopy_run,
        comparison_data_dir,
        prepare_comparison_data,
        reports_dir,
):
    point_comparison(
        'canopy_point',
        canopy_run.point_output.data,
        'point5',  # coniferous forest
        comparison_data_dir,
        prepare_comparison_data,
        reports_dir,
    )
