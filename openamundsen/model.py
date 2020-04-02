import copy
import loguru
from openamundsen import conf, fileio, meteo, statevars, util
import pandas as pd
import sys


class Model:
    def __init__(self, config):
        self.logger = None
        self.config = None
        self.state = None
        self.dates = None

        self._initialize_logger()
        conf.apply_config(self, config)

    def _prepare_time_steps(self):
        """
        Prepare the time steps for the model run according to the run
        configuration (start date, end date, and time step).
        """
        self.dates = pd.date_range(
            start=self.config['start_date'],
            end=self.config['end_date'],
            freq=pd.DateOffset(seconds=self.config['timestep']),
        )

    def _time_step_loop(self):
        for date in self.dates:
            self.logger.info(f'Processing time step {date}')
            meteo.interpolate_station_data(self, date)
            meteo.process_meteo_data(self)
            self._model_interface()
            self._update_gridded_outputs()
            self._update_point_outputs()

    def _model_interface(self):
        self.logger.debug('Modifying sub-canopy meteorology')
        self.logger.debug('Updating snow albedo')
        self.logger.debug('Adding fresh snow')
        self.logger.debug('Calculating canopy interception')
        self.logger.debug('Calculating melt')

    def _initialize_logger(self):
        loguru.logger.remove()
        logger = copy.deepcopy(loguru.logger)
        log_format = ('<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | ' +
                      '<level>{message}</level>')
        logger.add(sys.stderr, format=log_format, filter='openamundsen', level='DEBUG')
        self.logger = logger

    def _initialize_model_grid(self):
        """
        Initialize the grid parameters (number of rows and columns, transformation
        parameters) for the Model instance by reading the DEM file associated to the
        model run.
        """
        self.logger.info('Initializing model grid')

        dem_file = util.raster_filename('dem', self.config)
        meta = fileio.read_raster_metadata(dem_file)

        self.config['rows'] = meta['rows']
        self.config['cols'] = meta['cols']
        self.config['raster_meta'] = meta

        self.logger.info(f'Grid has dimensions {meta["rows"]}x{meta["cols"]}')

    def _initialize_state_variables(self):
        """
        Initialize the default state variables (i.e., create empty arrays) for
        the model run. Depending on which submodules are activated in the run
        configuration, further state variables might be added at other
        locations.
        """
        statevars.initialize_state_variables(self)

    def read_input_data(self):
        meta = self.config['raster_meta']

        dem_file = util.raster_filename('dem', self.config)
        roi_file = util.raster_filename('roi', self.config)

        if dem_file.exists():
            self.logger.info(f'Reading DEM ({dem_file})')
            self.state.base.dem[:] = fileio.read_raster_file(dem_file, check_meta=meta)
        else:
            raise FileNotFoundError(f'DEM file not found: {dem_file}')

        if roi_file.exists():
            self.logger.info(f'Reading ROI ({roi_file})')
            self.state.base.roi[:] = fileio.read_raster_file(roi_file, check_meta=meta)
        else:
            self.logger.debug('No ROI file available, setting ROI to entire grid area')
            self.state.base.roi[:] = True

    def read_meteo_data(self):
        self.logger.info('Reading meteo data')
        for station_num in range(7):
            self.logger.info(f'Reading station {station_num}')

    def initialize(self):
        self._prepare_time_steps()
        self._initialize_model_grid()
        self._initialize_state_variables()

        self.read_input_data()
        self.read_meteo_data()

    def run(self):
        self._time_step_loop()

    def _update_gridded_outputs(self):
        self.logger.debug('Updating gridded outputs')

    def _update_point_outputs(self):
        self.logger.debug('Updating point outputs')
