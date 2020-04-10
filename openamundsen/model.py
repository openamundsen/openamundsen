import copy
import loguru
from openamundsen import conf, fileio, meteo, statevars, util
from openamundsen import modules
import pandas as pd
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
            meteo.process_meteo_data(self)
            self._model_interface()
            self._update_gridded_outputs()
            self._update_point_outputs()

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
        """
        Read the input raster files required for the model run including the
        DEM, ROI (if available), and other files depending on the activated
        submodules.
        """
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
        """
        Read the meteorological data files required for the model run.
        """
        self.logger.info('Reading meteo data')

    def initialize(self):
        """
        Initialize the model according to the given configuration, i.e. read
        the required input raster files and meteorological input data,
        initialize the model grid and all required state variables, etc.
        """
        self._prepare_time_steps()
        self._initialize_model_grid()
        self._initialize_state_variables()

        self.read_input_data()
        self.read_meteo_data()

    def run(self):
        """
        Start the model run. Before calling this method, the model must be
        properly initialized by calling `initialize()`.
        """
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
