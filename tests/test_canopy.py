import numpy as np
from numpy.testing import assert_allclose
import openamundsen as oa
from pathlib import Path


def test_canopy(base_config, tmp_path):
    config = base_config.copy()
    config.start_date = '2020-02-01'
    config.end_date = '2020-02-28'

    model = oa.OpenAmundsen(config)
    model.initialize()

    roi_xs = model.grid.X.flat[model.grid.roi_idxs_flat]
    roi_ys = model.grid.Y.flat[model.grid.roi_idxs_flat]

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
    ds = model.point_output.data

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
