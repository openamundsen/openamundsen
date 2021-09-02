from .conftest import base_config
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import openamundsen as oa
import openamundsen.errors as errors
import pandas as pd
from pathlib import Path
import pytest
import xarray as xr


def compare_tsp(tsp, **kwargs):
    d = dict(
        first_of_run=False,
        strict_first_of_year=False,
        strict_first_of_month=False,
        strict_first_of_day=False,
        first_of_year=False,
        first_of_month=False,
        first_of_day=False,
        last_of_run=False,
        strict_last_of_year=False,
        strict_last_of_month=False,
        strict_last_of_day=False,
        last_of_year=False,
        last_of_month=False,
        last_of_day=False,
    )
    d.update(kwargs)

    for key in d.keys():
        assert getattr(tsp, key) == d[key], f'{key} should be {d[key]}'


def test_timestep_properties():

    config = base_config()
    config.start_date = '2015-07-28 00:00'
    config.end_date = '2015-12-31'
    config.timestep = 'H'

    model = oa.OpenAmundsen(config)
    model.initialize()

    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_run=True,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        strict_first_of_day=True,
    )

    model.run_single()
    compare_tsp(model.timestep_props)

    while model.date < pd.Timestamp('2015-07-29 00:00'):
        model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_day=True,
        strict_first_of_day=True,
    )

    while model.date < pd.Timestamp('2015-08-01 00:00'):
        model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_month=True,
        first_of_day=True,
        strict_first_of_month=True,
        strict_first_of_day=True,
    )

    config.start_date = '2015-07-28 23:00'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_run=True,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        last_of_day=True,
        strict_last_of_day=True,
    )

    config.start_date = '2015-08-01 00:00'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_run=True,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        strict_first_of_month=True,
        strict_first_of_day=True,
    )

    config.start_date = '2014-12-31 23:00'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_run=True,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        last_of_year=True,
        last_of_month=True,
        last_of_day=True,
        strict_last_of_year=True,
        strict_last_of_month=True,
        strict_last_of_day=True,
    )

    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        strict_first_of_year=True,
        strict_first_of_month=True,
        strict_first_of_day=True,
    )

    config.end_date = '2015-12-15 12:00'
    config.start_date = config.end_date
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_run=True,
        last_of_run=True,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        last_of_year=True,
        last_of_month=True,
        last_of_day=True,
    )
