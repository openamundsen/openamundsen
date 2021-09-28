import numpy as np
import pytest
import xarray as xr
import warnings


def compare_datasets(
    name,
    ds_dev,
    point=None,
    variables=None,
    data_dir=pytest.COMPARISON_DATA_DIR,
    prepare_comparison_data=pytest.PREPARE_COMPARISON_DATA,
    reports_dir=pytest.REPORTS_DIR,
):
    plot = reports_dir is not None
    nc_file = data_dir / f'{name}.nc'

    if prepare_comparison_data:
        ds_dev.to_netcdf(nc_file)
        return

    ds_base = xr.load_dataset(nc_file)

    if point is not None:
        ds_base = ds_base.sel(point=point)
        ds_dev = ds_dev.sel(point=point)

    # Check if we are comparing time series or gridded results
    is_time_series = 'point' in ds_base.coords

    if variables is None:
        base_vars = [v for v in ds_base.variables if v not in ds_base.coords]
        dev_vars = [v for v in ds_dev.variables if v not in ds_dev.coords]
        variables = base_vars
        variables.extend([v for v in dev_vars if v not in variables])

    compare_vars = []
    for v in variables:
        if not (v in ds_base.variables and v in ds_dev.variables):
            warnings.warn(f'Variable {v} not in both datasets, cannot compare')
        elif ds_base[v].shape != ds_dev[v].shape:
            warnings.warn(f'Non-matching shapes for variable {v}')
        else:
            compare_vars.append(v)

    if plot:
        plot_vars = []

        for v in compare_vars:
            if len(ds_base[v].dims) > 3:
                warnings.warn(f'Variable {v} is 3-dimensional, plotting not supported')
            else:
                plot_vars.append(v)

        if is_time_series:
            changed_vars = []
        else:
            changed_vars = {}

    for v in compare_vars:
        data_base = ds_base[v].values
        data_dev = ds_dev[v].values

        if is_time_series:
            if not np.allclose(data_base, data_dev, equal_nan=True):
                max_abs_diff, max_rel_diff = _max_diff(data_base, data_dev)
                warnings.warn(f'Non-matching values for variable {v}. '
                              f'Max abs/rel diff: {max_abs_diff:g} / {max_rel_diff:g}')

                if plot:
                    changed_vars.append(v)
        else:
            time_dim = ds_base[v].dims[0]
            time_vals = ds_base[v].coords[time_dim].to_index()

            for date_num in range(data_base.shape[0]):
                data_base_cur = data_base[date_num, :, :]
                data_dev_cur = data_dev[date_num, :, :]
                date = time_vals[date_num]

                if not np.allclose(data_base_cur, data_dev_cur, equal_nan=True):
                    max_abs_diff, max_rel_diff = _max_diff(data_base_cur, data_dev_cur)
                    warnings.warn(f'Non-matching values for variable {v} ({date}). '
                                  f'Max abs/rel diff: {max_abs_diff:g} / {max_rel_diff:g}')

                    if plot:
                        if v not in changed_vars:
                            changed_vars[v] = []

                        changed_vars[v].append(date_num)

    if plot:
        from .plot import plot_point_comparison, plot_gridded_comparison, fig_to_html
        # (import here because plotly is only required for report generation)

        if is_time_series:
            fig = plot_point_comparison(
                ds_base,
                ds_dev,
                plot_vars,
                changed_vars,
            )
        else:
            fig = plot_gridded_comparison(
                ds_base,
                ds_dev,
                plot_vars,
                changed_vars,
            )

        fig_to_html(fig, reports_dir / f'{name}.html')


def _max_diff(x, y):
    pos = np.isfinite(x) & np.isfinite(y)
    x = x[pos]
    y = y[pos]
    nonzero = (x != 0)
    abs_diff = np.abs(x - y)
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(abs_diff[nonzero] / np.abs(x[nonzero]))
    return max_abs_diff, max_rel_diff
