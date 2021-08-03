import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plot_point_comparison(
    ds_base,
    ds_dev,
    plot_vars,
    changed_vars,
    num_cols=3,
    plot_height=400,
    base_color='#1f77b4',
    dev_color='#d62728',
):
    num_rows = int(np.ceil(len(plot_vars) / num_cols))

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=plot_vars,
        shared_xaxes=True,
    )

    highlight_subplots = []

    for plot_num, v in enumerate(plot_vars):
        data_base = ds_base[v].to_pandas()
        data_dev = ds_dev[v].to_pandas()

        row = plot_num // num_cols + 1
        col = plot_num % num_cols + 1

        num_dims = len(ds_base[v].dims)

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

        if v in changed_vars:
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


def plot_gridded_comparison(
    ds_base,
    ds_dev,
    plot_vars,
    changed_vars,
    plot_height=400,
):
    titles = []
    num_rows = 0
    for v in plot_vars:
        time_dim = ds_base[v].dims[0]
        dates = ds_base[v].coords[time_dim].to_index()
        num_rows += len(dates)
        for d in dates:
            titles.append(f'{v} ({d})')
            titles.append('')  # right column

    fig = make_subplots(
        rows=num_rows,
        cols=2,
        subplot_titles=titles,
        shared_xaxes=True,
        # shared_yaxes=True,
    )

    row = 1
    for v in plot_vars:
        data_base = ds_base[v].values
        data_dev = ds_dev[v].values

        for var_plot_num in range(data_base.shape[0]):
            data_base_cur = data_base[var_plot_num, :, :]
            data_dev_cur = data_dev[var_plot_num, :, :]

            min_val = min(
                np.nanmin(data_base_cur),
                np.nanmin(data_dev_cur),
            )
            max_val = max(
                np.nanmax(data_base_cur),
                np.nanmax(data_dev_cur),
            )

            fig.add_trace(
                go.Heatmap(
                    z=data_base_cur,
                    zmin=min_val,
                    zmax=max_val,
                    x=ds_base.x,
                    y=ds_base.y,
                    colorscale='viridis',
                    showscale=False,
                ),
                row=row,
                col=1,
            )

            if v in changed_vars and var_plot_num in changed_vars[v]:
                fig.add_trace(
                    go.Heatmap(
                        z=data_dev_cur,
                        zmin=min_val,
                        zmax=max_val,
                        x=ds_base.x,
                        y=ds_base.y,
                        colorscale='viridis',
                        showscale=False,
                    ),
                    row=row,
                    col=2,
                )

            row += 1

    fig.for_each_yaxis(lambda ax: ax.update(scaleanchor=ax.anchor))
    fig.update_layout(height=plot_height * num_rows)

    return fig


def fig_to_html(fig, filename, create_dir=True):
    if create_dir:
        filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'w') as f:
        f.write(fig.to_html(include_plotlyjs='cdn'))
