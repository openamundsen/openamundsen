import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import openamundsen as oa
import pandas as pd


def test_slope():
    slope, _ = oa.terrain.slope_aspect(np.zeros((3, 3)), 1)
    assert_almost_equal(slope[1][1], 0)

    dem = np.array([
        [50, 45, 50],
        [30, 30, 30],
        [8, 10, 10],
    ], dtype=float)  # from https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm
    slope, _ = oa.terrain.slope_aspect(dem, 5)
    assert_almost_equal(slope[1][1], 75.25765769)


def test_aspect():
    dem = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
    ], dtype=float)

    _, aspect = oa.terrain.slope_aspect(dem, 1)
    assert aspect[1][1] == 0  # north facing

    _, aspect = oa.terrain.slope_aspect(np.rot90(dem), 1)
    assert aspect[1][1] == 270  # west facing

    _, aspect = oa.terrain.slope_aspect(np.rot90(dem, k=2), 1)
    assert aspect[1][1] == 180  # south facing

    _, aspect = oa.terrain.slope_aspect(np.rot90(dem, k=3), 1)
    assert aspect[1][1] == 90  # east facing

    dem = np.array([
        [101, 92, 85],
        [101, 92, 85],
        [101, 91, 84],
    ], dtype=float)  # from https://desktop.arcgis.com/en/arcmap/10.3/tools/3d-analyst-toolbox/how-aspect-works.htm
    _, aspect = oa.terrain.slope_aspect(dem, 1)
    assert_almost_equal(aspect[1][1], 92.64254529)
