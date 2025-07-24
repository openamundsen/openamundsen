import pytest

import openamundsen as oa

from .compare import compare_datasets
from .conftest import base_config


@pytest.mark.slow
@pytest.mark.comparison
def test_compare_interpolation_grids():
    bc = base_config()
    bc.start_date = "2019-12-01"
    bc.end_date = "2019-12-01"
    bc.output_data.grids.format = "memory"

    config = bc.copy()
    config.output_data.grids.variables = [{"var": "meteo.temp"}]
    config.meteo.interpolation.temperature.trend_method = "regression"
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    compare_datasets("interpolation_temp_regression", model.gridded_output.data)

    config.meteo.interpolation.temperature.trend_method = "fixed"
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    compare_datasets("interpolation_temp_fixed", model.gridded_output.data)

    config.meteo.interpolation.temperature.trend_method = "regression"
    config.meteo.interpolation.temperature.regression_params.max_segments = 2
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    compare_datasets("interpolation_temp_piecewise-regression", model.gridded_output.data)

    config = bc.copy()
    config.output_data.grids.variables = [{"var": "meteo.precip"}]
    config.meteo.interpolation.precipitation.trend_method = "regression"
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    compare_datasets("interpolation_precip_regression", model.gridded_output.data)

    config.meteo.interpolation.precipitation.trend_method = "adjustment_factor"
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    compare_datasets("interpolation_precip_adjustment_factor", model.gridded_output.data)

    config.meteo.interpolation.precipitation.trend_method = "fractional"
    config.meteo.interpolation.precipitation.lapse_rate = [
        0.00048,
        0.00046,
        0.00041,
        0.00033,
        0.00028,
        0.00025,
        0.00024,
        0.00025,
        0.00028,
        0.00033,
        0.00041,
        0.00046,
    ]
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    compare_datasets("interpolation_precip_fractional", model.gridded_output.data)

    config = bc.copy()
    config.output_data.grids.variables = [{"var": "meteo.rel_hum"}, {"var": "meteo.dew_point_temp"}]
    config.meteo.interpolation.humidity.trend_method = "regression"
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    compare_datasets("interpolation_hum_regression", model.gridded_output.data)

    config.meteo.interpolation.humidity.trend_method = "fixed"
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    compare_datasets("interpolation_hum_fixed", model.gridded_output.data)

    config.meteo.interpolation.humidity.trend_method = "regression"
    config.meteo.interpolation.humidity.regression_params.max_segments = 2
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    compare_datasets("interpolation_hum_piecewise-regression", model.gridded_output.data)

    config = bc.copy()
    config.output_data.grids.variables = [{"var": "meteo.wind_speed"}]
    config.meteo.interpolation.wind.trend_method = "regression"
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    compare_datasets("interpolation_wind_regression", model.gridded_output.data)

    # config.output_data.grids.variables = [{"var": "meteo.wind_speed"}, {"var": "meteo.wind_dir"}]
    # config.meteo.interpolation.wind.method = "liston"
    # model = oa.OpenAmundsen(config)
    # model.initialize()
    # model.run()
    # compare_datasets("interpolation_wind_liston", model.gridded_output.data)
