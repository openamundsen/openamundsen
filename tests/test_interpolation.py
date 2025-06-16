import pytest

import openamundsen as oa

from .compare import compare_datasets
from .conftest import base_config


@pytest.mark.slow
@pytest.mark.comparison
def test_compare_interpolation_grids():
    config = base_config()
    config.start_date = "2019-12-01"
    config.end_date = "2019-12-10"

    grid_cfg = config.output_data.grids
    grid_cfg.format = "memory"
    grid_cfg.variables = [
        {"var": "meteo.temp"},
        {"var": "meteo.rel_hum"},
        {"var": "meteo.precip"},
        {"var": "meteo.sw_in"},
        {"var": "meteo.wind_speed"},
        # {"var": "meteo.wind_dir"},
    ]

    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()

    compare_datasets(
        "interpolation_grids",
        model.gridded_output.data,
    )
