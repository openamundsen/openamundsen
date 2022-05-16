import numpy as np
from openamundsen.fileio import read_raster_file
import pytest

RASTER_TEMPLATE = '''
ncols        6
nrows        5
xllcorner    630800
yllcorner    5180500
cellsize     1000
nodata_value {val_nodata}
{val1} {val1} {val1} {val1} {val1} {val1}
{val1} {val1} {val2} {val2} {val2} {val2}
{val_nodata} {val2} {val2} {val2} {val2} {val2}
{val1} {val1} {val2} {val2} {val2} {val2}
{val1} {val_nodata} {val2} {val2} {val2} {val2}
'''.strip()


def test_fill_value(tmp_path):
    fn = tmp_path / 'test.asc'
    with open(fn, 'w') as f:
        f.write(RASTER_TEMPLATE.format(val1=0, val2=1, val_nodata=-9999))

    data = read_raster_file(fn)
    assert data[2, 0] == -9999

    data = read_raster_file(fn, fill_value=-1)
    assert data[2, 0] == -1

    with pytest.raises(TypeError):
        data = read_raster_file(fn, fill_value=np.nan)

    data = read_raster_file(fn, fill_value=np.nan, dtype=float)
    assert np.isnan(data[2, 0])

    data = read_raster_file(fn, fill_value=False, dtype=bool)
    assert not data[2, 0]


def test_dtype(tmp_path):
    fn = tmp_path / 'test.asc'
    with open(fn, 'w') as f:
        f.write(RASTER_TEMPLATE.format(val1=0, val2=1, val_nodata=-9999))

    data = read_raster_file(fn)
    assert np.issubdtype(data.dtype, np.integer)

    with pytest.raises(TypeError):
        data = read_raster_file(fn, fill_value=np.nan)

    data = read_raster_file(fn, fill_value=np.nan, dtype=float)
    assert np.issubdtype(data.dtype, float)

    data = read_raster_file(fn, fill_value=False, dtype=bool)
    assert np.issubdtype(data.dtype, bool)
