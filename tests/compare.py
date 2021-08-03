from .plot import make_point_comparison_plot, fig_to_html
import xarray as xr


def point_comparison(
    name,
    ds_dev,
    point,
    comparison_data_dir,
    prepare_comparison_data,
    reports_dir,
):
    point_filename = f'{name}.nc'

    if prepare_comparison_data:
        ds_dev.to_netcdf(comparison_data_dir / point_filename)
        return

    ds_base = xr.load_dataset(comparison_data_dir / point_filename)
    fig = make_point_comparison_plot(ds_base.sel(point=point), ds_dev.sel(point=point))
    if reports_dir is not None:
        fig_to_html(fig, reports_dir / f'{name}.html')
