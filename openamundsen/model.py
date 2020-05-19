import copy
import loguru
from openamundsen import (
    conf,
    constants,
    errors,
    fileio,
    interpolation,
    liveview,
    meteo,
    modules,
    statevars,
    terrain,
    util,
)
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
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
            self._calculate_irradiance(date)
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

        rows, cols = rasterio.transform.rowcol(self.grid.transform, x, y)
        row_var = bool_var.copy()
        col_var = bool_var.copy()
        row_var.values = rows
        col_var.values = cols
        ds['col'] = col_var
        ds['row'] = row_var

        grid = self.grid
        ds['within_grid_extent'] = (
            (x_var >= grid.x_min)
            & (x_var <= grid.x_max)
            & (y_var >= grid.y_min)
            & (y_var <= grid.y_max)
        )

        within_roi_var = bool_var.copy()
        ds['within_roi'] = within_roi_var

        for station in ds.indexes['station']:
            dss = ds.sel(station=station)

            if dss.within_grid_extent:
                row = int(dss.row)
                col = int(dss.col)
                within_roi_var.loc[station] = self.grid.roi[row, col]

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

        # TODO replace this eventually
        self.state.surface.albedo[self.grid.roi] = constants.SNOWFREE_ALBEDO
        self.state.surface.temp[self.grid.roi] = constants.T0
        self.state.snow.swe[self.grid.roi] = 0

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
                logger=self.logger,
            )
            self.state.base.svf[:] = svf

            self.logger.debug(f'Writing sky view factor file ({svf_file})')
            fileio.write_raster_file(svf_file, svf, self.grid.transform)

        self.grid.prepare_roi_coordinates()

    def _read_meteo_data(self):
        """
        Read the meteorological data files required for the model run and store
        them in the `meteo` variable.
        """
        meteo_format = self.config.input_data.meteo.format
        if meteo_format == 'netcdf':
            self.meteo = fileio.read_meteo_data_netcdf(
                self.config.input_data.meteo.dir,
                self.config.start_date,
                self.config.end_date,
                logger=self.logger,
            )
        elif meteo_format == 'csv':
            self.meteo = fileio.read_meteo_data_csv(
                self.config.input_data.meteo.dir,
                self.config.start_date,
                self.config.end_date,
                self.config.input_data.meteo.crs,
                logger=self.logger,
            )
        else:
            raise NotImplementedError('Unsupported meteo format')

        self._prepare_station_coordinates()

        # reorder variables (only for aesthetic reasons)
        var_order = [
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
        m.dewpoint_temp[roi] = meteo.dew_point_temperature(m.temp[roi], m.rel_hum[roi])
        m.precipitable_water[roi] = meteo.precipitable_water(
            m.temp[roi],
            m.vap_press[roi],
        )

    def _calculate_irradiance(self, date):
        sun_params = modules.radiation.sun_parameters(
            date,
            self.grid.center_lon,
            self.grid.center_lat,
            self.config.timezone,
        )

        day_angle = sun_params['day_angle']
        sun_vec = sun_params['sun_vector']
        zenith_angle = np.rad2deg(np.arccos(sun_vec[2]))
        sun_over_horizon = zenith_angle < 90

        self._calculate_clear_sky_shortwave_irradiance(day_angle, sun_vec, sun_over_horizon)
        self._calculate_shortwave_irradiance(date, sun_over_horizon)
        self._calculate_longwave_irradiance()

    def _calculate_clear_sky_shortwave_irradiance(self, day_angle, sun_vec, sun_over_horizon):
        roi = self.grid.roi

        mean_surface_albedo = self.state.surface.albedo[roi].mean()

        if sun_over_horizon:
            self.logger.debug('Calculating shadows')
            shadows = modules.radiation.shadows(
                self.state.base.dem,
                self.grid.resolution,
                sun_vec,
            )

            self.logger.debug('Calculating clear-sky shortwave irradiance')
            dir_irr, diff_irr = modules.radiation.clear_sky_shortwave_irradiance(
                day_angle,
                sun_vec,
                shadows,
                self.state.base.dem,
                self.state.base.svf,
                self.state.base.normal_vec,
                self.state.meteo.atmos_press,
                self.state.meteo.precipitable_water,
                mean_surface_albedo,
                roi=roi,
            )
        else:
            dir_irr = np.zeros((self.grid.rows, self.grid.cols))
            diff_irr = np.zeros((self.grid.rows, self.grid.cols))

        pot_irr = dir_irr + diff_irr
        self.state.meteo.sw_in_clearsky[roi] = pot_irr[roi]
        self.state.meteo.dir_in_clearsky[roi] = dir_irr[roi]
        self.state.meteo.diff_in_clearsky[roi] = diff_irr[roi]

    def _calculate_shortwave_irradiance(self, date, sun_over_horizon):
        self.logger.debug('Calculating actual shortwave irradiance')
        cloud_config = self.config.meteo.interpolation.cloudiness
        roi = self.grid.roi
        m = self.state.meteo

        # Select stations within the grid extent and with shortwave radiation measurements
        # for the current time step
        ds_rad = (
            self.meteo
            .isel(station=self.meteo.within_grid_extent)
            .sel(time=date)
            .dropna('station', subset=['sw_in'])
        )
        num_rad_stations = len(ds_rad.station)

        # Select stations with temperature and humidity measurements for the current time step
        ds_temp_hum = (
            self.meteo
            .sel(time=date)
            .dropna('station', how='any', subset=['temp', 'rel_hum'])
        )
        num_temp_hum_stations = len(ds_temp_hum.station)

        method = 'constant'
        if sun_over_horizon:
            if cloud_config['day_method'] == 'clear_sky_fraction':
                if num_rad_stations > 0:
                    method = 'clear_sky_fraction'
                else:
                    if cloud_config['allow_fallback'] and num_temp_hum_stations > 0:
                        self.logger.debug('No radiation measurements available, falling back '
                                          'to humidity-based cloudiness calculation')
                        method = 'humidity'
                    else:
                        self.logger.debug('No radiation measurements available, using cloudiness '
                                          'from previous time step')
                        method = 'constant'
            else:
                method = cloud_config['day_method']
        else:
            if cloud_config['night_method'] == 'humidity':
                if num_temp_hum_stations > 0:
                    method = 'humidity'
                else:
                    self.logger.debug('No temperature and humidity measurements available, using '
                                      'cloudiness from previous time step')
                    method = 'constant'

        if method == 'constant':
            pass  # use cloudiness from previous time step, i.e., do nothing
        elif method == 'humidity':
            lr_t = self.config.meteo.interpolation.temperature.lapse_rates[date.month - 1]
            lr_td = self.config.meteo.interpolation.humidity.lapse_rates[date.month - 1]
            # TODO use here also the same settings for regression/fixed gradients as for the interpolation
            cloud_fracs = meteo.cloud_fraction_from_humidity(
                ds_temp_hum.temp,
                ds_temp_hum.rel_hum,
                ds_temp_hum.alt,
                lr_t,
                lr_td,
            )
            cloud_factors = meteo.cloud_factor_from_cloud_fraction(cloud_fracs)
            cloud_factor_interpol = interpolation.idw(
                ds_temp_hum.x,
                ds_temp_hum.y,
                cloud_factors,
                self.grid.roi_points[:, 0],
                self.grid.roi_points[:, 1],
            )
            m.cloud_factor[roi] = cloud_factor_interpol.clip(0, 1)
        elif method == 'clear_sky_fraction':
            cloud_factors = ds_rad.sw_in.values / m.sw_in_clearsky[ds_rad.row, ds_rad.col]
            cloud_factor_interpol = interpolation.idw(
                ds_rad.x,
                ds_rad.y,
                cloud_factors,
                self.grid.roi_points[:, 0],
                self.grid.roi_points[:, 1],
            )
            m.cloud_factor[roi] = cloud_factor_interpol.clip(0, 1)

        m.cloud_fraction[roi] = meteo.cloud_fraction_from_cloud_factor(m.cloud_factor[roi])
        m.sw_in[roi] = m.sw_in_clearsky[roi] * m.cloud_factor[roi]
        m.sw_out[roi] = self.state.surface.albedo[roi] * m.sw_in[roi]

    def _calculate_longwave_irradiance(self):
        self.logger.debug('Calculating longwave irradiance')
        roi = self.grid.roi
        m = self.state.meteo
        clear_sky_emissivity = meteo.clear_sky_emissivity(m.precipitable_water[roi])

        # TODO these are parameters
        snow_emissivity = 0.99
        cloud_emissivity = 0.976  # emissivity of totally overcast skies (Greuell et al., 1997)
        rock_emission_factor = 0.01  # (K W-1 m2) temperature of emitting rocks during daytime is assumed to be higher than the air temperature by this factor multiplied by the incoming shortwave radiation (Greuell et al., 1997)

        # Incoming longwave radiation from the clear sky
        lw_in_clearsky = (
            clear_sky_emissivity
            * constants.STEFAN_BOLTZMANN
            * m.temp[roi]**4
            * self.state.base.svf[roi]
            * (1 - m.cloud_fraction[roi]**2)
        )

        # Incoming longwave radiation from clouds
        lw_in_clouds = (
            cloud_emissivity
            * constants.STEFAN_BOLTZMANN
            * m.temp[roi]**4
            * self.state.base.svf[roi]
            * m.cloud_fraction[roi]**2
        )

        # Incoming longwave radiation from surrounding slopes
        snowfree_count = (self.state.snow.swe[roi] == 0).sum()
        rock_fraction = snowfree_count / self.grid.roi.sum()
        lw_in_slopes = (
            constants.STEFAN_BOLTZMANN
            * (1 - self.state.base.svf[roi]) * (
                rock_fraction * (m.temp[roi] + rock_emission_factor * m.sw_in[roi])**4
                + (1 - rock_fraction) * self.state.surface.temp[roi]**4
            )
        )

        # Total incoming/outgoing longwave radiation
        m.lw_in[roi] = lw_in_clearsky + lw_in_clouds + lw_in_slopes
        m.lw_out[roi] = snow_emissivity * constants.STEFAN_BOLTZMANN * self.state.surface.temp[roi]**4

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
