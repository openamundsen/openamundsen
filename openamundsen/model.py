import copy
import loguru
from openamundsen import (
    conf,
    constants,
    errors,
    fileio,
    meteo,
    modules,
    surface,
    statevars,
    terrain,
    util,
)
import numpy as np
import pandas as pd
import rasterio
import sys
import time


class OpenAmundsen:
    """
    Class encapsulating the required data and methods for a single
    openAMUNDSEN model run.

    Parameters
    ----------
    config : dict
        Model run configuration.

    Examples
    --------
    >>> model = oa.OpenAmundsen(oa.read_config('config.yml'))
    >>> model.initialize()
    >>> model.run()
    """

    def __init__(self, config):
        self.config = conf.parse_config(config)
        self.logger = None
        self.state = None
        self.dates = None
        self.date = None
        self.date_idx = None

    def initialize(self):
        """
        Initialize the model according to the given configuration, i.e. read
        the required input raster files and meteorological input data,
        initialize the model grid and all required state variables, etc.
        """
        config = self.config

        self._initialize_logger()

        self._prepare_time_steps()
        self._initialize_grid()
        self._initialize_state_variable_management()

        self.require_soil = config.snow.model == 'multilayer'
        self.require_energy_balance = config.snow.melt.method == 'energy_balance'
        self.require_temperature_index = not self.require_energy_balance

        if config.snow.model == 'multilayer':
            self.snow = modules.snow.MultilayerSnowModel(self)
        elif config.snow.model == 'cryolayers':
            self.snow = modules.snow.CryoLayerSnowModel(self)
        else:
            raise NotImplementedError

        # Create snow redistribution factor state variables
        for precip_corr in config.meteo.precipitation_correction:
            if precip_corr['method'] == 'srf':
                self.state.base.add_variable('srf', '1', 'Snow redistribution factor')
                break  # multiple SRFs are not allowed

        if config.snow_management.enabled:
            try:
                import openamundsen_snowmanagement
            except ImportError:
                raise errors.ConfigurationError('The snow management module must be installed '
                                                'for enabling snow management.')
            self.snow_management = openamundsen_snowmanagement.SnowManagementModel(self)

        self._create_state_variables()

        self._read_input_data()
        self._read_meteo_data()
        self._calculate_terrain_parameters()

        config.results_dir.mkdir(parents=True, exist_ok=True)  # create results directory if necessary
        self._initialize_point_outputs()
        self._initialize_gridded_outputs()

        self._initialize_state_variables()

        if self.config.liveview.enabled:
            self.logger.info('Creating live view window')
            from openamundsen import liveview
            lv = liveview.LiveView(self.config.liveview, self.state, self.grid.roi)
            lv.create_window()
            self.liveview = lv

    def run(self):
        """
        Start the model run. Before calling this method, the model must be
        properly initialized by calling `initialize()`.
        """
        self.logger.info('Starting model run')
        start_time = time.time()

        for _ in range(len(self.dates)):
            self.run_single()

        time_diff = pd.Timedelta(seconds=(time.time() - start_time))
        self.logger.success('Model run finished. Runtime: ' + str(time_diff))

    def run_single(self):
        """
        Process the next time step, i.e., increment the date counter and call the methods for
        preparing the meteorological fields and the interface to the submodules.
        """
        if self.date_idx is None:
            self.date_idx = 0
        elif self.date_idx == len(self.dates) - 1:
            raise errors.RuntimeError('Model run already finished')
        else:
            self.date_idx += 1

        self.date = self.dates[self.date_idx]

        self.logger.info(f'Processing time step {self.date:%Y-%m-%d %H:%M}')

        meteo.interpolate_station_data(self)
        self._process_meteo_data()
        self._model_interface()
        self.point_output.update()
        self.gridded_output.update()

        if self.config.liveview.enabled:
            self.logger.debug('Updating live view window')
            self.liveview.update(self.date)

    def global_mask(self, mask, global_mask=None, global_idxs=None):
        if global_idxs is None:
            if global_mask is None:
                global_idxs = self.grid.roi_idxs_flat
            else:
                global_idxs = np.flatnonzero(global_mask)

        if mask.shape[-1] != len(global_idxs):
            raise Exception('Local mask does not match global mask size')

        if mask.ndim == 1:
            global_mask = np.zeros((self.grid.rows, self.grid.cols), dtype=bool)
            idxs = global_idxs[mask]
            global_mask.flat[idxs] = True
        elif mask.ndim == 2:
            dim3 = mask.shape[0]
            global_mask = np.zeros((dim3, self.grid.rows, self.grid.cols), dtype=bool)

            for i in range(dim3):
                idxs = global_idxs[mask[i, :]]
                global_mask[i, :, :].flat[idxs] = True

        return global_mask

    def roi_mask_to_global(self, mask):
        return self.global_mask(mask)

    def _prepare_time_steps(self):
        """
        Prepare the time steps for the model run according to the run
        configuration (start date, end date, and time step).
        """
        dates = pd.date_range(
            start=self.config['start_date'],
            end=self.config['end_date'],
            freq=self.config['timestep'],
        )
        self.dates = dates

        # Store timestep in seconds in the `timestep` attribute
        self.timestep = util.offset_to_timedelta(self.config['timestep']).total_seconds()

    def _model_interface(self):
        """
        Interface for calling the different submodules. This method is called
        in every time step after the meteorological fields have been prepared.
        """
        modules.radiation.irradiance(self)

        self.snow.compaction()
        self.snow.accumulation()
        self.snow.albedo_aging()

        if self.config.snow_management.enabled:
            self.snow_management.produce()
            self.snow_management.groom()

        self.snow.update_properties()

        if self.require_soil:
            modules.soil.soil_properties(self)

        surface.surface_properties(self)

        if self.require_energy_balance:
            surface.energy_balance(self)
        elif self.require_temperature_index:
            self.snow.temperature_index()

        self.snow.heat_conduction()
        self.snow.melt()
        self.snow.sublimation()
        self.snow.runoff()
        self.snow.update_properties()
        self.snow.update_layers()

        if self.require_soil:
            modules.soil.soil_heat_flux(self)
            modules.soil.soil_temperature(self)

    def _initialize_logger(self):
        """
        Initialize the logger for the model instance.
        """
        loguru.logger.remove()
        logger = copy.deepcopy(loguru.logger)
        log_format = ('<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | ' +
                      '<level>{message}</level>')
        logger.add(sys.stderr, format=log_format, filter='openamundsen', level=self.config.log_level)
        self.logger = logger

    def _initialize_grid(self):
        """
        Initialize the grid parameters (number of rows and columns, transformation
        parameters) for the OpenAmundsen instance by reading the DEM file associated to the
        model run.
        """
        self.logger.info('Initializing model grid')

        dem_file = util.raster_filename('dem', self.config)
        meta = fileio.read_raster_metadata(dem_file, crs=self.config.crs)
        self.logger.info(f'Grid has dimensions {meta["rows"]}x{meta["cols"]}')

        grid = util.ModelGrid(meta)
        grid.prepare_coordinates()
        self.grid = grid

    def _prepare_station_coordinates(self):
        """
        Transform the lon/lat coordinates of the meteorological stations to the
        coordinate system of the model grid. The transformed coordinates are
        stored in the `x` and `y` variables of the meteo dataset.
        Additionally, the row and column indices of the stations within the
        model grid are stored in the `row` and `col` variables, and two boolean
        variables `within_grid_extent` and `within_roi` indicate whether the
        stations lie within the model grid extent and the ROI, respectively.
        """
        ds = self.meteo

        x, y = util.transform_coords(ds.lon, ds.lat, constants.CRS_WGS84, self.config.crs)

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

        bool_var = ds.x.copy().astype(bool)
        bool_var[:] = False
        bool_var.attrs = {}
        int_var = bool_var.copy().astype(int)
        int_var[:] = -1

        rows, cols = rasterio.transform.rowcol(self.grid.transform, x, y)
        row_var = int_var.copy()
        col_var = int_var.copy()
        row_var.values[:] = rows
        col_var.values[:] = cols
        ds['col'] = col_var
        ds['row'] = row_var

        grid = self.grid
        ds['within_grid_extent'] = (
            (col_var >= 0)
            & (col_var < grid.cols)
            & (row_var >= 0)
            & (row_var < grid.rows)
        )

        within_roi_var = bool_var.copy()
        ds['within_roi'] = within_roi_var

        for station in ds.indexes['station']:
            dss = ds.sel(station=station)

            if dss.within_grid_extent:
                row = int(dss.row)
                col = int(dss.col)
                within_roi_var.loc[station] = self.grid.roi[row, col]

    def _initialize_state_variable_management(self):
        """
        Initialize the state variable management and add default state variables.
        """
        self.state = statevars.StateVariableManager(self.grid.rows, self.grid.cols)
        statevars.add_default_state_variables(self)

    def _create_state_variables(self):
        """
        Create the state variables (i.e., empty arrays with the shape of the
        model grid) for the model run.
        """
        self.state.initialize()

    def _initialize_state_variables(self):
        """
        Fill the state variables arrays with initial values.
        """
        self.snow.initialize()
        modules.soil.initialize(self)
        self.state.surface.temp[self.grid.roi] = self.state.soil.temp[0, self.grid.roi]

        if self.config.snow_management.enabled:
            self.snow_management.initialize()

    def _initialize_point_outputs(self):
        self.point_output = fileio.PointOutputManager(self)

    def _initialize_gridded_outputs(self):
        self.gridded_output = fileio.GriddedOutputManager(self)

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
        svf_file = util.raster_filename('svf', self.config)

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

        if svf_file.exists():
            self.logger.info(f'Reading sky view factor ({svf_file})')
            self.state.base.svf[:] = fileio.read_raster_file(svf_file, check_meta=self.grid)
        else:
            self.logger.info('Calculating sky view factor')
            svf = terrain.sky_view_factor(
                self.state.base.dem,
                self.grid.resolution,
                num_sweeps=self.config.meteo.radiation.num_shadow_sweeps,
                logger=self.logger,
            )
            self.state.base.svf[:] = svf

            self.logger.debug(f'Writing sky view factor file ({svf_file})')
            fileio.write_raster_file(svf_file, svf, self.grid.transform)

        # Read snow redistribution factor files
        for precip_corr in self.config.meteo.precipitation_correction:
            if precip_corr['method'] == 'srf':
                if 'file' in precip_corr:
                    srf_file = precip_corr['file']
                else:
                    srf_file = util.raster_filename('srf', self.config)

                self.logger.info(f'Reading snow redistribution factor ({srf_file})')
                self.state.base.srf[:] = fileio.read_raster_file(srf_file, check_meta=self.grid)
                break

        self.grid.prepare_roi_coordinates()

    def _read_meteo_data(self):
        """
        Read the meteorological data files required for the model run and store
        them in the `meteo` variable.
        """
        bounds = self.config.input_data.meteo.bounds
        if bounds == 'grid':
            x_min = self.grid.x_min
            y_min = self.grid.y_min
            x_max = self.grid.x_max
            y_max = self.grid.y_max
        elif bounds == 'global':
            x_min = -np.inf
            y_min = -np.inf
            x_max = np.inf
            y_max = np.inf
        elif isinstance(bounds, list):
            x_min, y_min, x_max, y_max = bounds

        if self.config.input_data.meteo.format == 'csv':
            if self.config.input_data.meteo.crs is None:
                meteo_crs = self.config.crs
            else:
                meteo_crs = self.config.input_data.meteo.crs
        elif self.config.input_data.meteo.format == 'netcdf':
            meteo_crs = None  # no CRS required for NetCDF input

        self.meteo = fileio.read_meteo_data(
            self.config.input_data.meteo.format,
            self.config.input_data.meteo.dir,
            self.config.start_date,
            self.config.end_date,
            meteo_crs=meteo_crs,
            grid_crs=self.config.crs,
            bounds=(x_min, y_min, x_max, y_max),
            exclude=self.config.input_data.meteo.exclude,
            include=self.config.input_data.meteo.include,
            freq=self.config['timestep'],
            aggregate=self.config.input_data.meteo.aggregate_when_downsampling,
            logger=self.logger,
        )

        self._prepare_station_coordinates()

        # reorder variables (only for aesthetic reasons)
        var_order = [
            'station_name',
            'lon',
            'lat',
            'alt',
            'x',
            'y',
            'col',
            'row',
            'within_grid_extent',
            'within_roi',
        ] + list(constants.METEO_VAR_METADATA.keys())
        self.meteo = self.meteo[var_order]

        meteo.correct_station_precipitation(self)

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
        m.wet_bulb_temp[roi] = meteo.wet_bulb_temperature(
            m.temp[roi],
            m.rel_hum[roi],
            m.vap_press[roi],
            m.psych_const[roi],
        )
        m.dew_point_temp[roi] = meteo.dew_point_temperature(m.temp[roi], m.rel_hum[roi])
        m.precipitable_water[roi] = meteo.precipitable_water(
            m.temp[roi],
            m.vap_press[roi],
        )
        m.dry_air_density[roi] = m.atmos_press[roi] / (constants.GAS_CONSTANT_DRY_AIR * m.temp[roi])

        # Calculate precipitation phase
        precip_phase_method = self.config.meteo.precipitation_phase.method
        if precip_phase_method == 'temp':
            pp_temp = m.temp
        elif precip_phase_method == 'wet_bulb_temp':
            pp_temp = m.wet_bulb_temp

        snowfall_frac = meteo.precipitation_phase(
            pp_temp[roi],
            threshold_temp=self.config.meteo.precipitation_phase.threshold_temp,
            temp_range=self.config.meteo.precipitation_phase.temp_range,
        )
        m.snowfall[roi] = snowfall_frac * m.precip[roi]
        m.rainfall[roi] = (1 - snowfall_frac) * m.precip[roi]

        # Redistribute snow
        if 'srf' in self.state.base:
            m.snowfall[roi] *= self.state.base.srf[roi]
            m.precip[roi] = m.snowfall[roi] + m.rainfall[roi]

        m.snowfall_rate[roi] = m.snowfall[roi] / self.timestep
        m.rainfall_rate[roi] = m.rainfall[roi] / self.timestep
        m.precip_rate[roi] = m.precip[roi] / self.timestep

    @property
    def is_first_timestep_of_model_run(self):
        return self.date_idx == 0

    @property
    def is_first_timestep_of_day(self):
        return (
            self.is_first_timestep_of_model_run
            or self.date.day != self.dates[self.date_idx - 1].day
        )


def Model(*args, **kwargs):
    import warnings
    warnings.warn(
        'Using oa.Model is deprecated, please use oa.OpenAmundsen instead',
        DeprecationWarning,
        stacklevel=2,
    )
    return OpenAmundsen(*args, **kwargs)
