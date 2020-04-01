from openamundsen.errors import RasterFileError
from pathlib import Path
import rasterio


def raster_filename(kind, config):
    """
    Return the filename of an input raster file for a model run.

    Parameters
    ----------
    kind : str
        Type of input file, e.g. 'dem' or 'roi'.

    config : dict
        Model run configuration.

    Returns
    -------
    file : pathlib.Path
    """
    dir = config['input_data']['grids']['dir']
    domain = config['domain']
    resolution = config['resolution']
    extension = 'asc'
    return Path(f'{dir}/{kind}_{domain}_{resolution}.{extension}')


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


def read_input_data(model):
    meta = model.config['raster_meta']

    dem_file = raster_filename('dem', model.config)
    roi_file = raster_filename('roi', model.config)

    if dem_file.exists():
        model.logger.info(f'Reading DEM ({dem_file})')
        model.state.base.dem[:] = read_raster_file(dem_file, check_meta=meta)
    else:
        raise FileNotFoundError(f'DEM file not found: {dem_file}')

    if roi_file.exists():
        model.logger.info(f'Reading ROI ({roi_file})')
        model.state.base.roi[:] = read_raster_file(roi_file, check_meta=meta)
    else:
        model.logger.debug('No ROI file available, setting ROI to entire grid area')
        model.state.base.roi[:] = True


def read_meteo_data(model):
    model.logger.info('Reading meteo data')
    for station_num in range(7):
        model.logger.info(f'Reading station {station_num}')


def update_field_outputs(model):
    model.logger.debug('Updating field outputs')


def update_point_outputs(model):
    model.logger.debug('Updating point outputs')
