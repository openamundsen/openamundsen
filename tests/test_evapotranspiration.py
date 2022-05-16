from .conftest import base_config
import numpy as np
from numpy.testing import assert_allclose
import openamundsen as oa
from pathlib import Path
import pytest


@pytest.mark.slow
def test_evapotranspiration(tmp_path):
    config = base_config()
    config.start_date = '2020-07-01'
    config.end_date = '2020-07-15'

    model = oa.OpenAmundsen(config)
    model.initialize()

    meteo = model.meteo.copy()
    meteo.temp.values[:, 30] = np.nan  # nan values should not propagate to evapotranspiration variables

    roi_xs = model.grid.X.flat[model.grid.roi_idxs_flat]
    roi_ys = model.grid.Y.flat[model.grid.roi_idxs_flat]

    for p in Path(config.input_data.grids.dir).glob('*.asc'):
        (tmp_path / p.name).symlink_to(p)

    soil = np.zeros(model.grid.shape, dtype=int)
    lc = np.zeros(model.grid.shape, dtype=int)

    # Test with all land cover classes and fixed soil texture
    soil[:] = 5
    lccs = model.config.land_cover.classes.keys()
    config.output_data.timeseries.points = []
    for lcc_num, lcc in enumerate(lccs):
        config.output_data.timeseries.points.append({
            'x': float(roi_xs[lcc_num]),
            'y': float(roi_ys[lcc_num]),
        })
        lc.flat[model.grid.roi_idxs_flat[lcc_num]] = lcc

    config.input_data.grids.dir = str(tmp_path)

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

    config.evapotranspiration.enabled = True
    config.output_data.timeseries.add_default_points = False
    config.output_data.timeseries.variables = [
        {'var': 'evapotranspiration.evaporation'},
        {'var': 'evapotranspiration.transpiration'},
        {'var': 'evapotranspiration.evapotranspiration'},
    ]

    model = oa.OpenAmundsen(config)
    model.initialize()
    model.meteo = meteo
    model.run()
    ds = model.point_output.data

    for lcc_num, lcc in enumerate(lccs):
        lcc_params = model.config.land_cover.classes[lcc]
        crop_coeff_type = lcc_params.get('crop_coefficient_type', None)
        is_sealed = lcc_params.get('is_sealed', False)

        ds_lcc = ds.isel(point=lcc_num)
        assert np.all(ds_lcc.evapotranspiration >= 0)
        assert ds_lcc.evapotranspiration.max() > 0

        if crop_coeff_type == 'dual' or is_sealed:
            assert np.all(ds_lcc.evaporation >= 0)
            assert np.all(ds_lcc.transpiration >= 0)
            assert_allclose(
                ds_lcc.evaporation + ds_lcc.transpiration,
                ds_lcc.evapotranspiration,
                rtol=1e-3,
            )
        elif crop_coeff_type == 'single':
            assert np.all(np.isnan(ds_lcc.evaporation))
            assert np.all(np.isnan(ds_lcc.transpiration))

    # Test with all soil texture classes and fixed land cover
    lc[:] = 9
    soil[:] = 0
    stcs = range(1, 9 + 1)

    config.output_data.timeseries.points = []
    for stc_num, stc in enumerate(stcs):
        config.output_data.timeseries.points.append({
            'x': float(roi_xs[stc_num]),
            'y': float(roi_ys[stc_num]),
        })
        soil.flat[model.grid.roi_idxs_flat[stc_num]] = stc

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

    model = oa.OpenAmundsen(config)
    model.initialize()
    model.meteo = meteo
    model.run()
    ds = model.point_output.data

    for stc_num, stc in enumerate(stcs):
        ds_stc = ds.isel(point=stc_num)
        assert np.all(ds_stc.evapotranspiration >= 0)
        assert ds_lcc.evapotranspiration.max() > 0
        assert np.all(ds_stc.evaporation >= 0)
        assert np.all(ds_stc.transpiration >= 0)
        assert_allclose(
            ds_stc.evaporation + ds_stc.transpiration,
            ds_stc.evapotranspiration,
            rtol=1e-3,
        )
