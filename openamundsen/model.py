import copy
import loguru
from openamundsen import (
    conf,
    constants,
    errors,
    fileio,
    forcing,
    meteo as oameteo,
    modules,
    surface,
    statevars,
    terrain,
    util,
)
from .landcover import LandCover
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

    def initialize(self, meteo=None, meteo_callback=None):
        """
        Initialize the model according to the given configuration, i.e. read
        the required input raster files and meteorological input data,
        initialize the model grid and all required state variables, etc.
        """
        config = self.config

        self._require_soil = config.snow.model == 'multilayer'
        self._require_energy_balance = config.snow.melt.method == 'energy_balance'
        self._require_temperature_index = not self._require_energy_balance
        self._require_canopy = config.canopy.enabled
        self._require_evapotranspiration = config.evapotranspiration.enabled
        self._require_land_cover = self._require_canopy or self._require_evapotranspiration
        self._require_soil_texture = self._require_evapotranspiration
        self._require_interpolation = config.input_data.meteo.format in ('csv', 'netcdf', 'memory')
        self._require_snow_management = (
            conf.SNOW_MANAGEMENT_AVAILABLE and config.snow_management.enabled
        )

        self._initialize_logger()

        self._prepare_time_steps()
        self._initialize_grid()
        self._initialize_state_variable_management()

        if config.snow.model == 'multilayer':
            self.snow = modules.snow.MultilayerSnowModel(self)
        elif config.snow.model == 'cryolayers':
            self.snow = modules.snow.CryoLayerSnowModel(self)
        else:
            raise NotImplementedError

        if self._require_land_cover:
            self.land_cover = LandCover(self)

        if self._require_canopy:
            self.canopy = modules.canopy.CanopyModel(self)

        if self._require_evapotranspiration:
            self.evapotranspiration = modules.evapotranspiration.EvapotranspirationModel(self)

        if config.meteo.interpolation.wind.method == 'liston':
            self.state.base.add_variable(
                'scaled_curvature',
                '1',
                'Topographic curvature scaled to [-0.5, 0.5]',
                retain=True,
            )

        # Create snow redistribution factor state variables
        for precip_corr in config.meteo.precipitation_correction:
            if precip_corr['method'] == 'srf':
                self.state.base.add_variable('srf', '1', 'Snow redistribution factor', retain=True)
                break  # multiple SRFs are not allowed

        if self._require_snow_management:
            import openamundsen_snowmanagement
            self.snow_management = openamundsen_snowmanagement.SnowManagementModel(self)

        self._create_state_variables()

        self._read_input_data()

        if config.input_data.meteo.format in ('netcdf', 'csv'):
            meteo = self._read_meteo_data()
        elif config.input_data.meteo.format == 'memory':
            if meteo is None:
                raise errors.MeteoDataError('A meteo dataset must be passed to '
                                            'OpenAmundsen.initialize() if the meteo input format '
                                            'is set to "memory"')

            if not forcing.is_valid_point_dataset(meteo, dates=self.dates):
                raise errors.MeteoDataError('Not a valid point forcing dataset')

            meteo = meteo.copy(deep=True)
        elif config.input_data.meteo.format == 'callback':
            if meteo_callback is None:
                raise errors.MeteoDataError('meteo_callback must be passed to '
                                            'OpenAmundsen.initialize() if the meteo input format '
                                            'is set to "callback"')
            elif not callable(meteo_callback):
                raise errors.MeteoDataError('meteo_callback must be callable (i.e., a function '
                                            ' or method)')

            self._meteo_callback = meteo_callback

            # Create a dummy point forcing dataset with 0 stations
            dummyds = forcing.make_empty_point_dataset(
                self.dates,
                'dummy',
                'dummy',
                np.nan,
                np.nan,
                np.nan,
            )
            meteo = forcing.combine_point_datasets([dummyds]).drop_isel(station=0)

        if (
            config.meteo.interpolation.cloudiness.method == 'prescribed'
            and 'cloud_cover' not in meteo
        ):
            raise errors.MeteoDataError(
                'Cloud cover data must be provided for cloudiness method "prescribed"'
            )

        self.meteo = forcing.prepare_point_coordinates(meteo, self.grid, self.config.crs)
        oameteo.correct_station_precipitation(self)

        # Extend ROI with the station positions
        if config.extend_roi_with_stations:
            pos_outside_roi_stations = self.meteo.within_grid_extent & ~self.meteo.within_roi
            any_outside_roi_stations = np.any(pos_outside_roi_stations)
            if any_outside_roi_stations:
                self.meteo.within_roi[pos_outside_roi_stations] = True
                outside_roi_rows = self.meteo.row[pos_outside_roi_stations].values
                outside_roi_cols = self.meteo.col[pos_outside_roi_stations].values
                self.grid.roi[outside_roi_rows, outside_roi_cols] = True
                self.grid.prepare_roi_coordinates()

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

        if self.config.reset_state_variables:
            self.state.reset()

        if self._require_interpolation:
            oameteo.interpolate_station_data(self)

        if self.config.input_data.meteo.format == 'callback':
            self._meteo_callback(self)

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
            global_mask = np.zeros(self.grid.shape, dtype=bool)
            idxs = global_idxs[mask]
            global_mask.flat[idxs] = True
        elif mask.ndim == 2:
            dim3 = mask.shape[0]
            global_mask = np.zeros((dim3, *self.grid.shape), dtype=bool)

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

        if self.config.simulation_timezone is None:
            hour_shift = 0
        else:
            hour_shift = self.config.timezone - self.config.simulation_timezone
        self._simulation_timezone_shift = pd.Timedelta(hours=hour_shift)

    def _model_interface(self):
        """
        Interface for calling the different submodules. This method is called
        in every time step after the meteorological fields have been prepared.
        """
        self._irradiance()

        if self._require_land_cover:
            self.land_cover.lai()

        if self._require_evapotranspiration or self._require_canopy:
            modules.canopy.above_canopy_meteorology(self)

        if self._require_canopy:
            self.canopy.meteorology()
            self.canopy.snow()

        self.snow.compaction()
        self.snow.accumulation()
        self.snow.albedo_aging()

        if self._require_snow_management:
            self.snow_management.produce()
            self.snow_management.groom()

        self.snow.update_properties()

        if self._require_soil:
            modules.soil.soil_properties(self)

        surface.surface_properties(self)

        if self._require_energy_balance:
            surface.energy_balance(self)
        elif self._require_temperature_index:
            self.snow.temperature_index()

        self.snow.heat_conduction()
        self.snow.melt()
        self.snow.sublimation()
        self.snow.runoff()
        self.snow.update_properties()
        self.snow.update_layers()

        if self._require_soil:
            modules.soil.soil_heat_flux(self)
            modules.soil.soil_temperature(self)

        if self._require_evapotranspiration:
            self.evapotranspiration.evapotranspiration()

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

        if self._require_land_cover:
            self.land_cover.initialize()

        if self._require_canopy:
            self.canopy.initialize()

        if self._require_evapotranspiration:
            self.evapotranspiration.initialize()

        if self._require_snow_management:
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

        if self.config.meteo.interpolation.wind.method == 'liston':
            # Calculate topographic curvature and normalize to a [-0.5, 0.5] range following Liston
            # et al. (2007)
            curv = terrain.curvature(
                self.state.base.dem,
                self.grid.resolution,
                'liston',
                L=self.config.meteo.interpolation.wind.curvature_length_scale,
            )
            self.state.base.scaled_curvature[:] = util.normalize_array(curv, -0.5, 0.5)

    def _read_input_data(self):
        """
        Read the input raster files required for the model run including the
        DEM, ROI (if available), and other files depending on the activated
        submodules.
        """
        dem_file = util.raster_filename('dem', self.config)
        roi_file = util.raster_filename('roi', self.config)
        svf_file = util.raster_filename('svf', self.config)

        self._read_extended_grids()

        # If an extendeded DEM+SVF are available, set the model DEM and SVF using them, otherwise
        # (try to) read them from file (or calculate SVF)
        if self.grid.extended_grid.available:
            self.state.base.dem[:] = self.grid.extended_grid.dem[
                self.grid.extended_grid.row_slice,
                self.grid.extended_grid.col_slice,
            ]
            self.state.base.svf[:] = self.grid.extended_grid.svf[
                self.grid.extended_grid.row_slice,
                self.grid.extended_grid.col_slice,
            ]
        else:
            if dem_file.exists():
                self.logger.info(f'Reading DEM ({dem_file})')
                self.state.base.dem[:] = fileio.read_raster_file(
                    dem_file,
                    check_meta=self.grid,
                    fill_value=np.nan,
                    dtype=float,
                )
            else:
                raise FileNotFoundError(f'DEM file not found: {dem_file}')

            if svf_file.exists():
                self.logger.info(f'Reading sky view factor ({svf_file})')
                self.state.base.svf[:] = fileio.read_raster_file(
                    svf_file,
                    check_meta=self.grid,
                    fill_value=np.nan,
                    dtype=float,
                )
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
                fileio.write_raster_file(svf_file, svf, self.grid.transform, decimal_precision=3)

        if roi_file.exists():
            self.logger.info(f'Reading ROI ({roi_file})')
            self.grid.roi[:] = fileio.read_raster_file(
                roi_file,
                check_meta=self.grid,
                fill_value=False,
                dtype=bool,
            )
        else:
            self.logger.debug('No ROI file available, setting ROI to entire grid area')
            self.grid.roi[:] = True

        dem_nan_pos = np.isnan(self.state.base.dem) & self.grid.roi
        if np.any(dem_nan_pos):
            self.logger.debug(f'Excluding {dem_nan_pos.sum()} pixels where DEM is NaN '
                              'from the ROI')
            self.grid.roi[dem_nan_pos] = False

        # Read snow redistribution factor files
        for precip_corr in self.config.meteo.precipitation_correction:
            if precip_corr['method'] == 'srf':
                if 'file' in precip_corr:
                    srf_file = precip_corr['file']
                else:
                    srf_file = util.raster_filename('srf', self.config)

                self.logger.info(f'Reading snow redistribution factor ({srf_file})')
                self.state.base.srf[:] = fileio.read_raster_file(
                    srf_file,
                    check_meta=self.grid,
                    fill_value=np.nan,
                    dtype=float,
                )
                break

        # Read land cover file
        if self._require_land_cover:
            land_cover_file = util.raster_filename('lc', self.config)

            if land_cover_file.exists():
                self.logger.info(f'Reading land cover ({land_cover_file})')
                self.state.land_cover.land_cover[:] = fileio.read_raster_file(
                    land_cover_file,
                    check_meta=self.grid,
                    fill_value=0,
                    dtype=int,
                )
            else:
                raise FileNotFoundError(f'Land cover file not found: {land_cover_file}')

        # Read soil texture file
        if self._require_soil_texture:
            soil_texture_file = util.raster_filename('soil', self.config)

            if soil_texture_file.exists():
                self.logger.info(f'Reading soil texture ({soil_texture_file})')
                self.state.evapotranspiration.soil_texture[:] = fileio.read_raster_file(
                    soil_texture_file,
                    check_meta=self.grid,
                    fill_value=0,
                    dtype=int,
                )
            else:
                raise FileNotFoundError(f'Soil texture file not found: {soil_texture_file}')

        self.grid.prepare_roi_coordinates()

    def _read_extended_grids(self):
        """
        Try to read the extended DEM if available, and read/calculate the extended SVF.
        """
        extended_dem_file = util.raster_filename('extended-dem', self.config)
        extended_svf_file = util.raster_filename('extended-svf', self.config)

        if not extended_dem_file.exists():
            return False

        self.logger.info(f'Reading extended DEM ({extended_dem_file})')
        ext_meta = fileio.read_raster_metadata(extended_dem_file, crs=self.config.crs)
        ext_dem = fileio.read_raster_file(
            extended_dem_file,
            fill_value=np.nan,
            dtype=float,
        )
        grid_transform = self.grid.transform
        ext_transform = ext_meta['transform']
        grid_ul_xy = rasterio.transform.xy(grid_transform, 0, 0, offset='ul')
        grid_lr_xy = rasterio.transform.xy(
            grid_transform,
            self.grid.rows - 1,
            self.grid.cols - 1,
            offset='lr',
        )
        ext_ul_xy = rasterio.transform.xy(ext_transform, 0, 0, offset='ul')
        ext_lr_xy = rasterio.transform.xy(
            ext_transform,
            ext_meta['rows'] - 1,
            ext_meta['cols'] - 1,
            offset='lr',
        )
        ext_offset_ul = rasterio.transform.rowcol(ext_transform, *grid_ul_xy, op=float)
        if not (float.is_integer(ext_offset_ul[0]) and float.is_integer(ext_offset_ul[1])):
            raise errors.RasterFileError('Extended DEM is not aligned correctly')
        if not (
            grid_ul_xy[0] >= ext_ul_xy[0]
            and grid_ul_xy[1] <= ext_ul_xy[1]
            and grid_lr_xy[0] <= ext_lr_xy[0]
            and grid_lr_xy[1] >= ext_lr_xy[1]
        ):
            raise errors.RasterFileError('Extended DEM does not fully cover the model grid')

        if extended_svf_file.exists():
            ext_svf = fileio.read_raster_file(
                extended_svf_file,
                fill_value=np.nan,
                dtype=float,
            )
            if ext_svf.shape != ext_dem.shape:
                raise errors.RasterFileError('Extended DEM and SVF have differing dimensions')
        else:
            self.logger.info('Calculating extended sky view factor')
            ext_svf = terrain.sky_view_factor(
                ext_dem,
                self.grid.resolution,
                num_sweeps=self.config.meteo.radiation.num_shadow_sweeps,
                logger=self.logger,
            )
            self.logger.debug(f'Writing extended sky view factor file ({extended_svf_file})')
            fileio.write_raster_file(extended_svf_file, ext_svf, ext_transform, decimal_precision=3)

        row_offset = int(ext_offset_ul[0])
        col_offset = int(ext_offset_ul[1])
        self.grid.extended_grid.available = True
        self.grid.extended_grid.rows = ext_dem.shape[0]
        self.grid.extended_grid.cols = ext_dem.shape[1]
        self.grid.extended_grid.row_offset = row_offset
        self.grid.extended_grid.col_offset = col_offset
        self.grid.extended_grid.row_slice = slice(row_offset, row_offset + self.grid.rows)
        self.grid.extended_grid.col_slice = slice(col_offset, col_offset + self.grid.cols)
        self.grid.extended_grid.dem = ext_dem
        self.grid.extended_grid.svf = ext_svf
        self.grid.extended_grid.normal_vec = terrain.normal_vector(ext_dem, self.grid.resolution)

        return True

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

        ds = fileio.read_meteo_data(
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

        return ds

    def _process_meteo_data(self):
        """
        Calculate derived meteorological variables from the interpolated fields.
        """
        self.logger.debug('Calculating derived meteorological variables')

        m = self.state.meteo
        roi = self.grid.roi

        m.atmos_press[roi] = oameteo.atmospheric_pressure(self.state.base.dem[roi])
        m.sat_vap_press[roi] = oameteo.saturation_vapor_pressure(m.temp[roi])
        m.vap_press[roi] = oameteo.vapor_pressure(m.temp[roi], m.rel_hum[roi])
        m.spec_hum[roi] = oameteo.specific_humidity(m.atmos_press[roi], m.vap_press[roi])
        m.spec_heat_cap_moist_air[roi] = oameteo.specific_heat_capacity_moist_air(m.spec_hum[roi])
        m.lat_heat_vap[roi] = oameteo.latent_heat_of_vaporization(m.temp[roi])
        m.psych_const[roi] = oameteo.psychrometric_constant(
            m.atmos_press[roi],
            m.spec_heat_cap_moist_air[roi],
            m.lat_heat_vap[roi],
        )
        m.wet_bulb_temp[roi] = oameteo.wet_bulb_temperature(
            m.temp[roi],
            m.rel_hum[roi],
            m.vap_press[roi],
            m.psych_const[roi],
        )
        m.dew_point_temp[roi] = oameteo.dew_point_temperature(m.temp[roi], m.rel_hum[roi])
        m.precipitable_water[roi] = oameteo.precipitable_water(
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

        snowfall_frac = oameteo.precipitation_phase(
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

    def _irradiance(self):
        self.sun_params = modules.radiation.sun_parameters(
            self.date + self._simulation_timezone_shift,
            self.grid.center_lon,
            self.grid.center_lat,
            self.config.timezone,
        )

        modules.radiation.clear_sky_shortwave_irradiance(self)

        if self._require_interpolation:
            modules.radiation.shortwave_irradiance(self)
        else:
            m = self.state.meteo
            roi = self.grid.roi
            m.sw_in[roi] = m.sw_in_clearsky[roi] * m.cloud_factor[roi]

        modules.radiation.longwave_irradiance(self)

    @property
    def timestep_props(self):
        """
        Get the properties of the current timestep.

        The return value is a dataclass with the following fields:

            - `first_of_run`: Whether this is the first timestep of the model
              run.
            - `strict_first_of_year`: Whether this is the first possible
              timestep of the current year.
            - `strict_first_of_month`: Whether this is the first possible
              timestep of the current month.
            - `strict_first_of_day`: Whether this is the first possible timestep
              of the current day.
            - `first_of_year`: Whether this is the first actually processed
              timestep of the current year.
            - `first_of_month`: Whether this is the first actually processed
              timestep of the current month.
            - `first_of_day`: Whether this is the first actually processed
              timestep of the current day.
            - `last_of_run`: Whether this is the last timestep of the model run.
            - `strict_last_of_year`: Whether this is the last possible timestep
              of the current year.
            - `strict_last_of_month`: Whether this is the last possible timestep
              of the current month.
            - `strict_last_of_day`: Whether this is the last possible timestep
              of the current day.
            - `last_of_year`: Whether this is the last actually processed
              timestep of the current year.
            - `last_of_month`: Whether this is the last actually processed
              timestep of the current month.
            - `last_of_day`: Whether this is the last actually processed
              timestep of the current day.

        The difference between the "strict" and the "non-strict" fields is that
        the latter take the model run start/end date into account. For example,
        if the start date is 2020-12-31 23:00, at the first timestep
        `first_of_year`, `first_of_month` and `first_of_day` will all be True
        (since these are the first _processed_ timesteps of the current year,
        month and day), whereas `strict_first_of_year`, `strict_first_of_month`
        and `strict_first_of_day` will only be True for the first time on
        2021-01-01 00:00.
        """
        pot_prev_date = self.date - pd.Timedelta(seconds=self.timestep)
        pot_next_date = self.date + pd.Timedelta(seconds=self.timestep)

        first_of_run = self.date_idx == 0
        strict_first_of_year = pot_prev_date.year != self.date.year
        strict_first_of_month = pot_prev_date.month != self.date.month
        strict_first_of_day = pot_prev_date.day != self.date.day
        first_of_year = strict_first_of_year or first_of_run
        first_of_month = strict_first_of_month or first_of_run
        first_of_day = strict_first_of_day or first_of_run

        last_of_run = self.date_idx == len(self.dates) - 1
        strict_last_of_year = pot_next_date.year != self.date.year
        strict_last_of_month = pot_next_date.month != self.date.month
        strict_last_of_day = pot_next_date.day != self.date.day
        last_of_year = strict_last_of_year or last_of_run
        last_of_month = strict_last_of_month or last_of_run
        last_of_day = strict_last_of_day or last_of_run

        return util.TimestepProperties(
            first_of_run=first_of_run,
            strict_first_of_year=strict_first_of_year,
            strict_first_of_month=strict_first_of_month,
            strict_first_of_day=strict_first_of_day,
            first_of_year=first_of_year,
            first_of_month=first_of_month,
            first_of_day=first_of_day,
            last_of_run=last_of_run,
            strict_last_of_year=strict_last_of_year,
            strict_last_of_month=strict_last_of_month,
            strict_last_of_day=strict_last_of_day,
            last_of_year=last_of_year,
            last_of_month=last_of_month,
            last_of_day=last_of_day,
        )


def Model(*args, **kwargs):
    import warnings
    warnings.warn(
        'Using oa.Model is deprecated, please use oa.OpenAmundsen instead',
        DeprecationWarning,
        stacklevel=2,
    )
    return OpenAmundsen(*args, **kwargs)
