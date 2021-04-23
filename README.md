# openAMUNDSEN

openAMUNDSEN is a modular snow and hydrological modeling framework written in
Python.

<p align="center">
  <img src="https://user-images.githubusercontent.com/17546246/115751189-3afe4c00-a399-11eb-8bfa-87d0a86c2119.gif" />
</p>

## Overview

openAMUNDSEN is a fully distributed model, designed primarily for resolving the mass and energy
balance of snow and ice covered surfaces in mountain regions.
Typically, it is applied in areas ranging from the point scale to the regional scale (i.e., up to
some hundreds to thousands of square kilometers), using a spatial resolution of 10–100 m and a
temporal resolution of 1–3 h, however its potential applications are very versatile.

Main features include:

* Spatial interpolation of scattered meteorological point measurements using a combined lapse
  rate–inverse distance weighting scheme
* Calculation of solar radiation taking into account terrain slope and orientation, hill shading
  and transmission losses and gains due to scattering, absorption, and reflections
* Adjustment of precipitation using several correction functions for wind-induced undercatch and
  redistribution of snow using terrain-based parameterizations
* Simulation of the snow and ice mass and energy balance using either a multilayer scheme (based on
  Essery (2015)) or a bulk-layering scheme using separate layers for new snow, old snow, firn and
  ice
* Calculation of snowmelt using the surface energy balance or a temperature index/enhanced
  temperature index method
* Usage of arbitrary timesteps (e.g. 10-minutely, daily) while resampling forcing data to the
  desired time resolution if necessary
* Flexible output of time series including arbitrary model variables for selected point locations in
  NetCDF or CSV format
* Flexible output of gridded model variables, either for specific dates or periodically (e.g., daily
  or monthly), optionally aggregated to averages or sums
* Live view window for displaying the model state in real time

## Quick start

### Installation

openAMUNDSEN requires Python 3.6+.
To install it including the required dependencies, run

`pip install git+https://github.com/openamundsen/openamundsen.git`

(preferably in a virtualenv or a conda environment).

### Examples

Example data sets for running the model can be downloaded from
https://github.com/openamundsen/openamundsen-examples.

## Setting up a model run

### Input data

Required input data for running the model is at the least:

* a digital elevation model (DEM) as an Arc/Info ASCII Grid (.asc) file in a projected coordinate
  reference system, with the same spatial resolution in which the model should be run,
* and time series of the meteorological variables air temperature, precipitation, relative humidity,
  global radiation and wind speed in NetCDF or CSV format.

Additionally, a region of interest (ROI) file can be supplied defining a subset of the DEM area in
which the model should be applied.

#### Spatial input data

The DEM file must be named `dem_{domain}_{resolution}.asc`, where `{domain}` refers to the (freely
selectable) name of the respective model domain, and `{resolution}` to the spatial resolution in m.
Accordingly, the ROI file (if available) is named `roi_{domain}_{resolution}.asc`.

#### Meteorological input data

Meteorological input time series must be provided in the same or higher temporal resolution in which
the model should be run.
For each point location, a CSV or NetCDF file covering the entire time series must be provided.

When using CSV as input format, the input files should have one or more of the following columns
(columns for variables not available can be omitted):

* `date`: timestamp as a `pd.to_datetime`-compatible string (e.g. `YYYY-MM-DD HH:MM`)
* `temp`: air temperature (K)
* `precip`: precipitation sum (kg m<sup>-2</sup>)
* `rel_hum`: relative humidity (%)
* `sw_in`: global radiation (W m<sup>-2</sup>)
* `wind_speed`: wind speed (m s<sup>-1</sup>)

Additionally, a `stations.csv` file containing the metadata of the point locations must be specified
containing the following columns:

* `id`: station ID, corresponding to the filename of the respective data file
* `name`: station name
* `x`: longitude or projected x coordinate
* `y`: latitude or projected y coordinate
* `alt`: altitude (m)

When using NetCDF as input format, for each station a NetCDF file with the following schema should be
provided (unavailable variables can be omitted):

```
netcdf dummy {
dimensions:
        time = UNLIMITED ;
variables:
        double alt ;
                alt:_FillValue = NaN ;
                alt:standard_name = "surface_altitude" ;
                alt:units = "m" ;
        float hurs(time) ;
                hurs:_FillValue = NaNf ;
                hurs:standard_name = "relative_humidity" ;
                hurs:units = "%" ;
        double lat ;
                lat:_FillValue = NaN ;
                lat:standard_name = "latitude" ;
                lat:units = "degree_north" ;
        double lon ;
                lon:_FillValue = NaN ;
                lon:standard_name = "longitude" ;
                lon:units = "degree_east" ;
        float pr(time) ;
                pr:_FillValue = NaNf ;
                pr:standard_name = "precipitation_flux" ;
                pr:units = "kg m-2 s-1" ;
        float rsds(time) ;
                rsds:_FillValue = NaNf ;
                rsds:standard_name = "surface_downwelling_shortwave_flux_in_air" ;
                rsds:units = "W m-2" ;
        float tas(time) ;
                tas:_FillValue = NaNf ;
                tas:standard_name = "air_temperature" ;
                tas:units = "K" ;
        int64 time(time) ;
                time:standard_name = "time" ;
                time:units = "hours since 1999-01-01 00:00:00" ;
                time:calendar = "proleptic_gregorian" ;
        float wss(time) ;
                wss:_FillValue = NaNf ;
                wss:standard_name = "wind_speed" ;
                wss:units = "m s-1" ;

// global attributes:
                :Conventions = "CF-1.6" ;
                :station_name = "dummy" ;
}
```

### Configuration

The configuration of an openAMUNDSEN model run can either be read in from a
[YAML](https://en.wikipedia.org/wiki/YAML) file or be passed directly as a dictionary from within
Python.

This is an example of a model run configuration file:

```yaml
domain: rofental # name of the model domain (corresponding to the domain part of the spatial input data filenames)
start_date: "2020-10-01"
end_date: "2021-03-31"
resolution: 50  # spatial resolution (m)
timestep: H  # temporal resolution as a pandas-compatible frequency string (e.g., "H", "3H", "D")
crs: "epsg:32632"  # CRS of the input grids
timezone: 1  # timezone of the model domain (difference to UTC in h)
results_dir: results  # directory for storing the model outputs

# Input data configuration
input_data:
  grids:
    dir: input/grid  # location of the input grids (DEM, ROI etc.)
  meteo:
    dir: input/meteo  # location of the meteorological input data
    format: csv  # input format (CSV or NetCDF)
    crs: "epsg:4326"  # CRS of the station coordinates (when using CSV)

# Output data configuration
output_data:
  # Time series (point) outputs configuration
  timeseries:
    # List of points to be written
    points:
      - x: 642579 # x coordinate in the domain CRS
        y: 5193069 # y coordinate in the domain CRS
        name: testpoint # point name (optional)

    add_default_variables: true # write default point output variables
    variables: # optional additional output variables not written by default
      - var: surface.turbulent_exchange_coeff

  # Configuration for gridded outputs
  grids:
    format: netcdf # NetCDF or ASCII
    variables:
      - var: meteo.precip # internal variable name
        name: precip_month # NetCDF output variable name
        freq: M # write frequency (if not specified, write every timestep)
        agg: sum # aggregation function ("sum", "mean" or empty)
      - var: snow.melt
        freq: M
        agg: sum
      - var: snow.swe
        freq: D

meteo:
  # Spatial interpolation parameters
  interpolation:
    temperature:
      trend_method: fixed # use fixed monthly temperature lapse rates

    precipitation:
      trend_method: fractional # use fixed monthly fractional precipitation gradients
      lapse_rate: # (m-1)
        - 0.00100 # J
        - 0.00095 # F
        - 0.00085 # M
        - 0.00069 # A
        - 0.00059 # M
        - 0.00052 # J
        - 0.00050 # J
        - 0.00052 # A
        - 0.00059 # S
        - 0.00069 # O
        - 0.00085 # N
        - 0.00095 # D

    humidity:
      trend_method: fixed # use fixed monthly dew point temperature lapse rates

  # Precipitation phase determination parameters
  precipitation_phase:
    method: wet_bulb_temp # use wet-bulb temperature for precipitation phase determination
    threshold_temp: 273.65 # threshold temperature (K) in which 50% of precipitation falls as snow
    temp_range: 1. # temperature range in which mixed precipitation can occur

  # Parameters for adjusting precipitation for wind-induced undercatch and snow redistribution
  precipitation_correction:
    - method: wmo
      gauge: hellmann

snow:
  model: multilayer # snow scheme ("multilayer" or "cryolayers")

  # Number of layers and minimum thicknesses (m) when using the multilayer model
  min_thickness:
    - 0.1
    - 0.2
    - 0.4

  albedo:
    min: 0.55 # minimum snow albedo
    max: 0.85 # maximum snow albedo
    cold_snow_decay_timescale: 480 # albedo decay timescale for cold (T < 0 °C) snow (h)
    melting_snow_decay_timescale: 200 # albedo decay timescale for melting snow (h)
    refresh_snowfall: 0.5 # snowfall amount for resetting albedo to the maximum value (kg m-2 h-1)
```

Only few configuration parameters (`domain`, `start_date`, `end_date`, `resolution`, `timezone` and
the input data directories) are mandatory, for all other parameters default values are used
otherwise.
A detailed documentation of all model parameters will be available soon (in
the meantime, the available parameters and their default values can be looked up in
[configschema.yml](./openamundsen/data/configschema.yml)).

### Running the model

When the input data and the model configuration have been prepared, a model run can be started either
using the `openamundsen` command line utility (`openamundsen config_file.yml`), or from within
Python using the following syntax:

```python
import openamundsen as oa

config = oa.read_config('config_file.yml')  # read in configuration file
model = oa.OpenAmundsen(config)  # create OpenAmundsen object and populate unspecified parameters with default values
model.initialize()  # read in input data files, initialize state variables etc.
model.run()  # run the model
```
