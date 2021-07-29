import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings


def make_point_comparison_plot(
    ds_base,
    ds_dev,
    variables=None,
    num_cols=3,
    plot_height=400,
    base_color='#1f77b4',
    dev_color='#d62728',
):
    if variables is None:
        base_vars = [v for v in ds_base.variables if v not in ds_base.coords]
        dev_vars = [v for v in ds_dev.variables if v not in ds_dev.coords]
        variables = [v for v in base_vars if v in dev_vars]

    prelim_vars = variables
    variables = []
    for v in prelim_vars:
        if v in ds_base.variables and v in ds_dev.variables:
            variables.append(v)
        else:
            warnings.warn(f'Variable {v} not in both datasets, cannot compare')
            variables.remove(v)

    num_rows = int(np.ceil(len(variables) / num_cols))

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=variables,
        shared_xaxes=True,
    )

    highlight_subplots = []

    for plot_num, nc_var in enumerate(variables):
        data_base = ds_base[nc_var].to_pandas()
        data_dev = ds_dev[nc_var].to_pandas()

        if data_base.shape != data_dev.shape:
            warnings.warn(f'Differing shapes for variable {nc_var}: '
                          f'{data_base.shape} vs. {data_dev.shape}')
            continue

        row = plot_num // num_cols + 1
        col = plot_num % num_cols + 1

        num_dims = len(ds_base[nc_var].dims)

        if num_dims == 1:
            data_base = data_base.to_frame()
            data_dev = data_dev.to_frame()

        for dim in range(num_dims):
            fig.add_trace(
                go.Scatter(
                    x=data_base.index,
                    y=data_base[dim],
                    name='base',
                    mode='lines',
                    line=dict(color=base_color),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        if not np.allclose(data_base, data_dev, equal_nan=True):
            warnings.warn(f'Differing values for variable {nc_var}')
            highlight_subplots.append(plot_num)

            for dim in range(num_dims):
                fig.add_trace(
                    go.Scatter(
                        x=data_dev.index,
                        y=data_dev[dim],
                        name='dev',
                        mode='lines',
                        line=dict(color=dev_color),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

    for plot_num in highlight_subplots:
        fig['layout']['annotations'][plot_num]['font']['color'] = 'red'

    fig.update_layout(height=plot_height * num_rows)

    return fig
