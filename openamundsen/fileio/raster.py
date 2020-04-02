from openamundsen.errors import RasterFileError
import rasterio


def read_raster_metadata(filename):
    """
    Return metadata for a raster file.

    Parameters
    ----------
    filename : str or pathlib.Path

    Returns
    -------
    meta : dict
        Dictionary containing the following keys 'rows' (number of rows),
        'cols' (number of columns), 'resolution' ((width, height) tuple), and
        'transform' (georeferencing transformation parameters).
    """
    meta = {}

    with rasterio.open(filename) as ds:
        meta['rows'] = ds.meta['height']
        meta['cols'] = ds.meta['width']
        meta['resolution'] = ds.res
        meta['transform'] = ds.meta['transform']

    return meta


def read_raster_file(filename, check_meta=None):
    """
    Read a raster file.

    Parameters
    ----------
    filename : str or pathlib.Path

    check_meta : dict, default None
        A metadata dictionary (as returned by `read_raster_metadata` to compare
        the current raster metadata with. If the metadata of the two rasters
        does not match (e.g., if the number of rows or columns differs) a
        RasterFileError is raised.

    Returns
    -------
    data : np.ndarray
    """
    if check_meta is not None:
        meta = read_raster_metadata(filename)
        if meta != check_meta:
            raise RasterFileError(f'Metadata mismatch for {filename}')

    with rasterio.open(filename) as ds:
        data = ds.read(1)

    return data
