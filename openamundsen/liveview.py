import openamundsen as oa


try:
    import pyqtgraph as pg
    import pyqtgraph.multiprocess as mp
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

        self.img_height, self.img_width = self.state.base.dem.shape
        self.cbar_width = 20
        self.cbar_spacing = 30  # spacing between image and colorbar

    def create_window(self):
        """
        Prepare and open the live view window.
        """
        pg.mkQApp()

        # Create remote process
        proc = mp.QtProcess()
        rpg = proc._import('pyqtgraph')

        self.proc = proc
        self.rpg = rpg

        # Interpret image data as row-major instead of col-major
        rpg.setConfigOptions(imageAxisOrder='row-major')

        win = rpg.GraphicsLayoutWidget()
        win.setWindowTitle('openAMUNDSEN')

        win.addLabel(f'openAMUNDSEN v{oa.__version__}', colspan=2)
        time_label = win.addLabel()

        gei = rpg.GradientEditorItem()
        gei.loadPreset('viridis')
        lut = gei.getLookupTable(50)
        self.cmap = gei.colorMap()

        fields = []
        imgs = []

        for field_num, d in enumerate(self.config.fields):
            field = d['var']
            min_range = d['min']
            max_range = d['max']

            if field_num % self.config.cols == 0:
                win.nextRow()

            vb = rpg.ViewBox(enableMouse=False, enableMenu=False, lockAspect=True)
            vb.invertY(True)  # y axis points downward (otherwise images are plotted upside down)

            pi = rpg.PlotItem(title=self._var_label(field), viewBox=vb)

            for ax in ('left', 'right', 'top', 'bottom'):
                pi.hideAxis(ax)

            img = rpg.ImageItem(lut=lut, levels=(min_range, max_range))

            pi.addItem(img)
            win.addItem(pi)

            min_label = f'{min_range:g}'
            max_label = f'{max_range:g}'
            self._colorbar(vb, min_label, max_label)

            imgs.append(img)
            fields.append(field)

        self.fields = fields
        self.win = win
        self.time_label = time_label
        self.imgs = imgs

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

        for field, img in zip(self.fields, self.imgs):
            data = self.state[field]
            img.setImage(data, autoLevels=False)

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

    def _colorbar(self, vb, min_label, max_label):
        rpg = self.rpg

        gradient = self.cmap.getGradient()
        gradient.setStart(self.img_width + self.cbar_spacing, 0 + self.img_height)
        gradient.setFinalStop(self.img_width + self.cbar_spacing, 0)

        rect = rpg.QtGui.QGraphicsRectItem(
            self.img_width + self.cbar_spacing,
            0,
            self.cbar_width,
            self.img_height,
        )
        rect.setPen(rpg.mkPen('w'))
        rect.setBrush(rpg.QtGui.QBrush(gradient))

        label_html = '<span style="color: #CCC; font-size: 10pt;">{}</span>'

        ti_min = rpg.TextItem(html=label_html.format(min_label), anchor=(0.5, 0))
        ti_min.setPos(
            self.img_width + self.cbar_spacing + self.cbar_width / 2.,
            self.img_height,
        )

        ti_max = rpg.TextItem(html=label_html.format(max_label), anchor=(0.5, 1))
        ti_max.setPos(
            self.img_width + self.cbar_spacing + self.cbar_width / 2.,
            0,
        )

        vb.addItem(rect)
        vb.addItem(ti_min)
        vb.addItem(ti_max)
