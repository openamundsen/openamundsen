import copy
import loguru
from openamundsen import (
    conf,
    constants,
    errors,
    fileio,
    liveview,
    meteo,
    modules,
    statevars,
    terrain,
    util,
)
import pandas as pd
from pathlib import Path
import sys
import time


class Model:
    """
    Class encapsulating the required data and methods for a single
    openAMUNDSEN model run.

    Parameters
    ----------
    config : dict
        Model run configuration.

    Examples
    --------
    >>> model = oa.Model(oa.read_config('config.yml'))
    >>> model.initialize()
    >>> model.run()
    """

    def __init__(self, config):
        self.logger = None
        self.config = None
        self.state = None
        self.dates = None

        self._initialize_logger()

        self.logger.info('Checking configuration')
        full_config = conf.full_config(config)
        self.config = conf.parse_config(full_config)

    def _prepare_time_steps(self):
        """
        Prepare the time steps for the model run according to the run
        configuration (start date, end date, and time step).
        """
        dates = pd.date_range(
            start=self.config['start_date'],
            end=self.config['end_date'],
            freq=pd.DateOffset(seconds=self.config['timestep']),
        )
        self.dates = list(dates.to_pydatetime())

    def _time_step_loop(self):
        """
        Run the main model loop, i.e. iterate over all time steps and call the
        methods for preparing the meteorological fields and the interface to
        the submodules.
        """
        for date in self.dates:
            self.logger.info(f'Processing time step {date:%Y-%m-%d %H:%M}')
            meteo.interpolate_station_data(self, date)
            self._process_meteo_data()
            self._model_interface()
            self._update_gridded_outputs()
            self._update_point_outputs()

            if self.config.liveview.enabled:
                self.liveview.update(date)

    def _model_interface(self):
        """
        Interface for calling the different submodules. This method is called
        in every time step after the meteorological fields have been prepared.
        """
        modules.radiation.irradiance(self)
        modules.snow.update_albedo(self)
        modules.snow.compaction(self)
        modules.snow.add_fresh_snow(self)
        modules.snow.energy_balance(self)

    def _initialize_logger(self):
        """
        Initialize the logger for the model instance.
        """
        loguru.logger.remove()
        logger = copy.deepcopy(loguru.logger)
        log_format = ('<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | ' +
                      '<level>{message}</level>')
        logger.add(sys.stderr, format=log_format, filter='openamundsen', level='DEBUG')
        self.logger = logger

    def _initialize_grid(self):
        """
        Initialize the grid parameters (number of rows and columns, transformation
        parameters) for the Model instance by reading the DEM file associated to the
        model run.
        """
        self.logger.info('Initializing model grid')

        dem_file = util.raster_filename('dem', self.config)
        meta = fileio.read_raster_metadata(dem_file)
        self.logger.info(f'Grid has dimensions {meta["rows"]}x{meta["cols"]}')

        grid = util.ModelGrid(meta)
        grid.prepare_coordinates()
        self.grid = grid

    def _prepare_station_coordinates(self):
        """
        Transform the lon/lat coordinates of the meteorological stations to the
        coordinate system of the model grid. The transformed coordinates are
        stored in the `x` and `y` variables of the meteo dataset.
        """
        ds = self.meteo

        src_crs = 'epsg:4326'  # WGS 84
        dst_crs = self.config.crs
        x, y = util.transform_coords(ds.lon, ds.lat, src_crs, dst_crs)

        x_var = ds.lon.copy()
        x_var.values = x
        x_var.attrs = {
            'standard_name': 'projection_x_coordinate',
            'units': 'm',
        }

        y_var = ds.lat.copy()
        y_var.values = y
        y_var.attrs = {
            'standard_name': 'projection_y_coordinate',
            'units': 'm',
        }

        ds['x'] = x_var
        ds['y'] = y_var

    def _initialize_state_variables(self):
        """
        Initialize the default state variables (i.e., create empty arrays) for
        the model run. Depending on which submodules are activated in the run
        configuration, further state variables might be added at other
        locations.
        """
        self.state = statevars.StateVariableManager(self.grid.rows, self.grid.cols)
        statevars.add_default_state_variables(self)
        self.state.initialize()

    def _calculate_terrain_parameters(self):
        self.logger.info('Calculating terrain parameters')
        slope, aspect = terrain.slope_aspect(self.state.base.dem, self.grid.resolution)
        normal_vec = terrain.normal_vector(self.state.base.dem, self.grid.resolution)
        self.state.base.slope[:] = slope
        self.state.base.aspect[:] = aspect
        self.state.base.normal_vec[:] = normal_vec

    def _read_input_data(self):
        """
        Read the input raster files required for the model run including the
        DEM, ROI (if available), and other files depending on the activated
        submodules.
        """
        dem_file = util.raster_filename('dem', self.config)
        roi_file = util.raster_filename('roi', self.config)

        if dem_file.exists():
            self.logger.info(f'Reading DEM ({dem_file})')
            self.state.base.dem[:] = fileio.read_raster_file(dem_file, check_meta=self.grid)
        else:
            raise FileNotFoundError(f'DEM file not found: {dem_file}')

        if roi_file.exists():
            self.logger.info(f'Reading ROI ({roi_file})')
            self.grid.roi[:] = fileio.read_raster_file(roi_file, check_meta=self.grid)
        else:
            self.logger.debug('No ROI file available, setting ROI to entire grid area')
            self.grid.roi[:] = True

        self.grid.prepare_roi_coordinates()

    def _read_meteo_data(self):
        """
        Read the meteorological data files required for the model run and store
        them in the `meteo` variable.
        """
        if self.config.input_data.meteo.format != 'netcdf':
            raise NotImplementedError('Only NetCDF meteo input currently supported')

        meteo_data_dir = Path(self.config.input_data.meteo.dir)
        nc_files = sorted(list(meteo_data_dir.glob('*.nc')))

        if len(nc_files) == 0:
            raise errors.MeteoDataError('No meteo files found')

        datasets = []

        for nc_file in nc_files:
            self.logger.info(f'Reading meteo file: {nc_file}')

            ds = fileio.read_netcdf_meteo_file(nc_file)
            ds = ds.sel(time=slice(self.config.start_date, self.config.end_date))

            if ds.dims['time'] == 0:
                self.logger.info('File contains no meteo data for the specified period')
            else:
                datasets.append(ds)

        if len(datasets) == 0:
            raise errors.MeteoDataError('No meteo data available for the specified period')

        self.meteo = fileio.combine_meteo_datasets(datasets)
        self._prepare_station_coordinates()

        # reorder variables (only for aesthetic reasons)
        var_order = ['lon', 'lat', 'alt', 'x', 'y'] + list(constants.METEO_VAR_METADATA.keys())
        self.meteo = self.meteo[var_order]

    def _process_meteo_data(self):
        """
        Calculate derived meteorological variables from the interpolated fields.
        """
        self.logger.debug('Calculating derived meteorological variables')

        m = self.state.meteo
        roi = self.grid.roi

        m.atmos_press[roi] = meteo.atmospheric_pressure(self.state.base.dem[roi])
        m.sat_vap_press[roi] = meteo.saturation_vapor_pressure(m.temp[roi])
        m.vap_press[roi] = meteo.vapor_pressure(m.temp[roi], m.rel_hum[roi])
        m.spec_hum[roi] = meteo.specific_humidity(m.atmos_press[roi], m.vap_press[roi])
        m.spec_heat_cap_moist_air[roi] = meteo.specific_heat_capacity_moist_air(m.spec_hum[roi])
        m.lat_heat_vap[roi] = meteo.latent_heat_of_vaporization(m.temp[roi])
        m.psych_const[roi] = meteo.psychrometric_constant(
            m.atmos_press[roi],
            m.spec_heat_cap_moist_air[roi],
            m.lat_heat_vap[roi],
        )
        m.wetbulb_temp[roi] = meteo.wet_bulb_temperature(
            m.temp[roi],
            m.rel_hum[roi],
            m.vap_press[roi],
            m.psych_const[roi],
        )
        m.dewpoint_temp[roi] = meteo.dew_point_temperature(m.vap_press[roi])

    def initialize(self):
        """
        Initialize the model according to the given configuration, i.e. read
        the required input raster files and meteorological input data,
        initialize the model grid and all required state variables, etc.
        """
        self._prepare_time_steps()
        self._initialize_grid()
        self._initialize_state_variables()

        self._read_input_data()
        self._read_meteo_data()
        self._calculate_terrain_parameters()

    def run(self):
        """
        Start the model run. Before calling this method, the model must be
        properly initialized by calling `initialize()`.
        """
        if self.config.liveview.enabled:
            self.logger.info('Creating live view window')
            lv = liveview.LiveView(self.config.liveview, self.state)
            lv.create_window()
            self.liveview = lv

        self.logger.info('Starting model run')
        start_time = time.time()
        self._time_step_loop()
        time_diff = pd.Timedelta(seconds=(time.time() - start_time))
        self.logger.success('Model run finished. Runtime: ' + str(time_diff))

    def _update_gridded_outputs(self):
        """
        Update (in the case of aggregated fields) and potentially write the
        gridded output fields according to the run configuration.
        """
        self.logger.debug('Updating gridded outputs')

    def _update_point_outputs(self):
        """
        Update and write the output time series for the selected variables and
        point locations according to the run configuration.
        """
        self.logger.debug('Updating point outputs')
