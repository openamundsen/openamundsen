import openamundsen as oa
import numpy as np


try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Qt5Agg')
    mpl_available = True
except ImportError:
    mpl_available = False


class LiveView:
    """
    Class for managing a live view window of a model run.

    Parameters
    ----------
    config : dict
        The "liveview" section of the model run configuration (i.e.,
        model.config.liveview).

    state : StateVariableManager
        StateVariableManager (i.e., the model.state attribute) of the
        respective OpenAmundsen instance.

    roi : ndarray
        Boolean array specifying the region of interest.
    """
    def __init__(self, config, state, roi):
        if not mpl_available:
            raise ModuleNotFoundError('matplotlib and/or PyQt5 not available')

        self.config = config
        self.state = state
        self.roi = roi

    def create_window(self):
        """
        Prepare and open the live view window.
        """
        plt.style.use('dark_background')
        plt.rcParams['toolbar'] = 'None'

        num_cols = self.config.cols
        num_rows = int(np.ceil(len(self.config.variables) / num_cols))
        fig, axarr = plt.subplots(ncols=num_cols, nrows=num_rows)
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.03, right=0.97)

        fields = []
        imgs = []

        for field_num, d in enumerate(self.config.variables):
            field = d['var']
            min_range = d['min']
            max_range = d['max']

            data = self.state[field]

            ax = axarr.flat[field_num]
            img = ax.imshow(
                data,
                vmin=min_range,
                vmax=max_range,
                interpolation='None',
                cmap=self.config.cmap,
            )

            cbar = fig.colorbar(img, ax=ax)
            cbar.ax.tick_params(labelsize='x-small')

            ax.set_title(self._var_label(field), fontsize='small')
            ax.axis('off')

            imgs.append(img)
            fields.append(field)

        # Hide unused subplot spaces
        for ax in axarr.flat[len(self.config.variables):]:
            ax.axis('off')

        plt.text(
            x=0.03,
            y=0.95,
            s=f'openAMUNDSEN v{oa.__version__}',
            fontsize='xx-large',
            ha='left',
            transform=fig.transFigure,
        )
        time_label = plt.text(
            x=0.97,
            y=0.95,
            s='',
            fontsize='large',
            ha='right',
            transform=fig.transFigure,
            bbox=dict(  # set the text background color to ensure that old values are overplotted
                facecolor=plt.rcParams['figure.facecolor'],
                linewidth=0,
            ),
        )

        self.fields = fields
        self.fig = fig
        self.imgs = imgs
        self.axarr = axarr
        self.time_label = time_label

        if self.config.blit:
            fig.canvas.mpl_connect('draw_event', self._on_draw)

        plt.show(block=False)
        fig.canvas.draw()
        mgr = plt.get_current_fig_manager()
        mgr.set_window_title('openAMUNDSEN')
        mgr.resize(self.config.width, self.config.height)

    def update(self, date):
        """
        Update the live view window with the current values of the state
        variables.

        Parameters
        ----------
        date : datetime
            Current model time step.
        """
        fig = self.fig

        if self.config.blit:
            fig.canvas.restore_region(self.fig_bg)

        # Update time label
        time_label = self.time_label
        time_label.set_text(f'{date:%Y-%m-%d %H:%M}')

        if self.config.blit:
            fig.draw_artist(time_label)
            label_extent = time_label.get_window_extent(renderer=fig.canvas.get_renderer())
            label_bbox = matplotlib.transforms.TransformedBbox(
                label_extent.transformed(fig.transFigure.inverted()),
                fig.transFigure,
            )
            fig.canvas.blit(label_bbox)

        # Update images
        for field, ax, img in zip(self.fields, self.axarr.flat, self.imgs):
            data = self.state[field]

            # Hide non-ROI pixels (conversion to float is required for integer fields)
            data = data.astype(float, copy=False)
            data[~self.roi] = np.nan

            # Perform downsampling
            data = data[::self.config.downsample]

            img.set_data(data)

            if self.config.blit:
                ax.draw_artist(img)
                fig.canvas.blit(ax.bbox)

        if not self.config.blit:
            fig.canvas.draw()

        fig.canvas.flush_events()

    def close(self):
        """
        Close the window.
        """
        plt.close(self.fig)

    def __del__(self):
        """
        Close the window when the object is deleted.
        """
        self.close()

    def _on_draw(self, event):
        fig = self.fig
        self.fig_bg = fig.canvas.copy_from_bbox(fig.bbox)

    def _var_label(self, field):
        """
        Return the label for a variable to be shown in the live view window.

        Parameters
        ----------
        field : str
            "<category>.<var_name>" string.

        Returns
        -------
        label : str
        """
        category, var_name = self.state.parse(field)
        meta = self.state[category]._meta[var_name]

        if meta.long_name is not None:
            label = meta.long_name
        elif meta.standard_name is not None:
            label = meta.standard_name
        else:
            label = var_name

        if meta.units is not None:
            label += f'\n({meta.units})'

        return label
