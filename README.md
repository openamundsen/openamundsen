# openAMUNDSEN

openAMUNDSEN is a modular snow and hydrological modeling framework written in
Python. It can be used both as a Python library as well as a standalone command
line utility.

## Quick start

 1. Clone the git repository to your local hard disk, either using `git clone
    https://git.uibk.ac.at/c716657/openamundsen.git` or using a GUI tool such
    as SourceTree or GitHub Desktop.
 2. Install the required dependencies. This can be done either globally or in a
    virtualenv or a conda environment. For Anaconda/Miniconda, a global install of the
    missing dependencies can be done using the following command[^1]:

        conda install -c conda-forge \
            munch                    \
            loguru                   \
            netCDF4                  \
            numba                    \
            numpy                    \
            pandas                   \
            pyproj                   \
            pyqt                     \
            pyqtgraph                \
            ruamel.yaml              \
            rasterio                 \
            scipy                    \
            xarray
 3. Run the included sample model setup for the Rofental domain, either by
    running `./bin/openamundsen examples/rofental.yml` from the main
    openamundsen directory, or by running the rofental_test.py script located in
    the examples directory (if you are running the script from an IDE, make sure
    that the working directory is set to the main openamundsen directory).

[^1]: In case of dependency resolving problems, a `conda update --all` might be
  necessary before the `conda install`.

## Overview

### Configuration

The configuration of an openAMUNDSEN model run is given as a collection of
key-value pairs (i.e., corresponding to a dictionary in Python). When reading
the configuration from a file the [YAML](https://en.wikipedia.org/wiki/YAML)
format is used. A simple configuration file might look like this:

```yaml
domain: rofental
start_date: 2019-11-01
end_date: 2020-04-30
resolution: 50
timestep: H

input_data:
  grids:
    dir: data/rofental
  meteo:
    dir: data/rofental/meteo
```

Only few configuration parameters (such as `start_date` and `end_date`) are
absolutely necessary; for most parameters default values are used if no value
is explicitly specified. The "reference" configuration containing the default
values is located under openamundsen/data/defaultconfig.yml.

### Code structure

The `Model` class encapsulates the required data and functionality for an
openAMUNDSEN model run. The object oriented nature allows to, e.g., run several
model runs in parallel from a single Python script. When instantiating a
`Model`, the configuration for the model run must be passed in the constructor.
After instantiation, the `initialize()` method must be called in order to
create and initialize all required state variables, read the input files and
meteorological data, etc. Then, the `run()` method can be called to perform the
actual model run.

Hence, a simple model run from within Python would look like this:

```python
import openamundsen as oa

config = oa.read_config('config_file.yml')
model = oa.Model(config)
model.initialize()
model.run()
```

The state variables of a model run are stored in the `state` attribute of the
respective `Model` object. They are organized in categories such as `base`
(containing basic data such as the DEM and derived variables (slope, aspect,
…), the ROI, etc.), `meteo` (containing the meteorological fields), `snow`
(containing the snow-specific variables), etc. For example, the DEM array can
be accessed as `model.state.base.dem`, the temperature field as
`model.state.meteo.temp`, and the total SWE as `model.state.snow.swe`. This
modular nature makes it easy for submodules to create and access their own
state variables (e.g., when the glacier module is activated the
glacier-specific state variables would be stored under `model.state.glacier`).

From within the `run()` method, the internal `_time_step_loop()` method is
called. This is where the main loop over all time steps happens. Within the
loop, first the meteorological fields are prepared, and subsequently the
`_model_interface()` method is called. This is the location where the
individual submodules are plugged in. After `_model_interface()` returns, the
grid and point outputs are updated and potentially written, before proceeding
to the next time step.

## Release Notes/Changelog

### v0.1 (2020-07-30)

- Calculate precipitation phase
- Implement point outputs (to NetCDF or CSV)
- Implement field outputs (2D/3D variables, for single dates/regular
  intervals/temporally aggregated (sum/mean), to NetCDF or ASCII)
- Implement Cox et al. (1999) soil model
- Implement Essery (2015) snow model
- Implement AMUNDSEN snow albedo (Rohrer, 1992) and densification (Anderson,
  1976) parameterizations

### v0.0.3 (2020-05-20)

- Show color bars in live view, allow to set min/max range for each variable.
- Calculate atmospheric variables (atmospheric pressure, vapor pressure,
  absolute/specific humidity, wet-bulb/dew point temperature, cloud fraction,
  etc.).
- Calculate sun-related parameters (day angle, hour angle, declination angle,
  equation of time, sun vector).
- Calculate terrain parameters (slope, aspect, normal vector, sky view
  factor).
- Calculate shortwave and longwave irradiance.
- Interpolate relative humidity not directly but via dew point temperature.
- Added CSV meteo data reader.

### v0.0.2 (2020-04-22)

- Meteorological station data in NetCDF format can be read in.
- IDW interpolation function has been implemented.
- Air temperature, precipitation, humidity and wind speed measurements are
  interpolated to the model grid using IDW with elevation detrending (with
  automatic calculation of lapse rates in each time step).
- Live view window for showing state variable fields during a model run has
  been implemented.
- Metadata for state variables ([CF](http://cfconventions.org)-compliant
  attributes `standard_name`, `long_name`, `units`) can be specified (used e.g.
  for labeling plots in the live view window and subsequently for annotating
  output data).
- Some first unit tests have been created.

### v0.0.1 (2020-04-06)

- Basic model structure has been established.
- Utility functions (reading config file, reading raster files, preparing time
  steps, …) have been created.
- DEM and ROI are read in.
- Some modules have been created and filled with dummy functions.
