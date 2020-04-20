import openamundsen as oa
import numpy as np


try:
    import pyqtgraph as pg
    import pyqtgraph.multiprocess as mp
    from PyQt5 import QtCore, QtGui, QtWidgets
    pg_available = True
except ModuleNotFoundError:
    pg_available = False


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
        respective Model instance.
    """
    def __init__(self, config, state):
        if not pg_available:
            raise ModuleNotFoundError('pyqtgraph and/or PyQt5 not available')

        self.config = config
        self.state = state

    def create_window(self):
        """
        Prepare and open the live view window.
        """
        pg.mkQApp()

        # Create remote process
        proc = mp.QtProcess()
        rpg = proc._import('pyqtgraph')

        # Interpret image data as row-major instead of col-major
        rpg.setConfigOptions(imageAxisOrder='row-major')

        win = rpg.GraphicsLayoutWidget()
        win.setWindowTitle('openAMUNDSEN')

        win.addLabel(f'openAMUNDSEN v{oa.__version__}', colspan=2)
        time_label = win.addLabel()

        gei = rpg.GradientEditorItem()
        gei.loadPreset('viridis')
        lut = gei.getLookupTable(50)

        plot_items = []
        imgs = []
        labels = []

        for field_num, field in enumerate(self.config.fields):
            if field_num % self.config.cols == 0:
                win.nextRow()

            vb = rpg.ViewBox(enableMouse=False, enableMenu=False, lockAspect=True)
            vb.invertY(True)  # y axis points downward (otherwise images are plotted upside down)

            pi = rpg.PlotItem(viewBox=vb)
            for ax in ('left', 'right', 'top', 'bottom'):
                pi.hideAxis(ax)

            img = rpg.ImageItem()
            img.setLookupTable(lut)

            pi.addItem(img)
            win.addItem(pi)

            plot_items.append(pi)
            imgs.append(img)
            labels.append(self._var_label(field))

        self.proc = proc
        self.rpg = rpg
        self.win = win
        self.time_label = time_label
        self.imgs = imgs
        self.plot_items = plot_items
        self.labels = labels

        win.show()

    def update(self, date):
        """
        Update the live view window with the current values of the state
        variables.

        Parameters
        ----------
        date : datetime
            Current model time step.
        """
        self.time_label.setText(f'{date:%Y-%m-%d %H:%M}')

        for field, pi, img, label in zip(self.config.fields, self.plot_items, self.imgs, self.labels):
            data = self.state[field]
            img.setImage(data)

            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            label += f'<br>(min={data_min:g} max={data_max:g})'
            pi.setTitle(label)

    def close(self):
        """
        Close the window and kill the underlying Qt process.
        """
        self.proc.close()

    def __del__(self):
        """
        Close the window when the object is deleted.
        """
        self.close()

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
            label += f' ({meta.units})'

        return label
