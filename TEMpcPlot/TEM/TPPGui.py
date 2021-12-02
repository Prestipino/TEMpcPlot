import sys
import numpy as np
import pickle
import os

from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow, QAction,
                             QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QSpacerItem, QSizePolicy, QPushButton, QComboBox,
                             QFileDialog, QMessageBox)

from PyQt5.QtGui import QIcon

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt  # noqa
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar  # noqa

# from PyQt5.QtCore import QSize

from .IntrctPeaks import MossMod  # noqa


Intrct = MossMod()


def setconstrain(par_n, expr=None):
    Intrct.FitModel.set_param_hint(par_n, expr=expr)


def setvalue(par_n, expr):
    Intrct.FitModel.set_param_hint(par_n, value=expr)


def setlimits(par_n, minmax, value):
    if value is None:
        del Intrct.FitModel.param_hints[par_n][minmax]
    Intrct.FitModel.param_hints[par_n][minmax] = value


def printParams():
    print('\n' * 2, '#' * 5)
    for i in Intrct.FitModel.make_params().values():
        print(i)
    print('#' * 5, '\n' * 2)


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.title = 'Moss fitting softly'
        self.left = 50
        self.top = 50
        self.width = 960
        self.height = 960
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        #icon = __file__[:-10] + 'Moss.svg'
        #self.setWindowIcon(QIcon(icon))

        self.statusBar().showMessage('Ready')

        # menubar
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(True)
        fileMenu = mainMenu.addMenu('File')
        helpMenu = mainMenu.addMenu('Help')

        openHelp = QAction("&Help", self)
        openHelp.setShortcut("Ctrl+H")
        openHelp.setStatusTip('Mouse Help')
        openHelp.triggered.connect(self.showHelp)
        helpMenu.addAction(openHelp)

        openFile = QAction("&Open File", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip('Open File')
        openFile.triggered.connect(lambda x: self.file_open(open=True))
        fileMenu.addAction(openFile)

        exportButton = QAction('&Export curves', self)
        exportButton.setShortcut('Ctrl+Z')
        exportButton.setStatusTip('Saves plot curves as dat files')
        exportButton.triggered.connect(self.save_curves)
        fileMenu.addAction(exportButton)

        repButton = QAction('&Save Fit report', self)
        repButton.setShortcut('Ctrl+R')
        repButton.setStatusTip('Save Fit report')
        repButton.triggered.connect(self.save_report)
        fileMenu.addAction(repButton)

        LoPButton = QAction('&Load Project', self)
        LoPButton.setShortcut('Ctrl+L')
        LoPButton.setStatusTip('Load Project')
        LoPButton.triggered.connect(self.LoadProject)
        fileMenu.addAction(LoPButton)

        SaPButton = QAction('&Save Project', self)
        SaPButton.setShortcut('Ctrl+S')
        SaPButton.setStatusTip('Save Project')
        SaPButton.triggered.connect(self.SaveProject)
        fileMenu.addAction(SaPButton)

        # exitButton = QAction(QIcon('exit.png'), 'Exit', self)
        exitButton = QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        widget = QWidget(self)
        self.setCentralWidget(widget)
        hlay = QVBoxLayout(widget)
        # vlay = QHBoxLayout()
        # vlay.addLayout(hlay)

        # nameLabel = QLabel('Prefix:', self)
        # self.prefix_LE = QLineEdit(self)
        # self.prefix_LE.setText('g1_')
        # # self.nameLabel2 = QLabel('Result', self)

        # self.cb_psh = QComboBox()
        # self.cb_psh.addItem("Lorentzian")
        # self.cb_psh.addItem("PseudoVoig")
        # self.cb_psh.addItem("Gaussian")

        # multLabel = QLabel('   Quadrupolar:', self)
        # self.cb_mult = QComboBox()
        # self.cb_mult.addItem("singlet")
        # self.cb_mult.addItem("doublet")
        # # self.cb_psh.currentIndexChanged.connect(self.selectionchange)

        # msLabel = QLabel('   Hyperfine:', self)
        # self.cb_ms = QComboBox()
        # self.cb_ms.addItem("singlet")
        # self.cb_ms.addItem("sextet")
        # # self.cb_psh.currentIndexChanged.connect(self.selectionchange)

        # addP = QPushButton('Add +', self)
        # addP.setStyleSheet("background-color:rgb(150, 205, 220)")
        # addP.clicked.connect(self.AddPeak)

        # remP = QPushButton('Remove', self)
        # remP.setStyleSheet("background-col or:rgb(200, 200, 220)")
        # remP.clicked.connect(self.RemPeak)

        # FitB = QPushButton('Fit', self)
        # FitB.setStyleSheet("background-color:rgb(50, 205, 50)")
        # FitB.clicked.connect(self.FitMethod)

        # hlay.addWidget(nameLabel)
        # hlay.addWidget(self.prefix_LE)
        # hlay.addWidget(self.cb_psh)
        # hlay.addWidget(multLabel)
        # hlay.addWidget(self.cb_mult)
        # hlay.addWidget(msLabel)
        # hlay.addWidget(self.cb_ms)
        # hlay.addWidget(addP)
        # hlay.addItem(QSpacerItem(10, 1, QSizePolicy.Preferred))

        # hlay.addWidget(remP)
        # hlay.addItem(QSpacerItem(1000, 1, QSizePolicy.Expanding))
        # hlay.addWidget(FitB)

        self.WP = WidgetPlot(self)
        self.pick_id = self.WP.canvas.mpl_connect('pick_event', self.onpick)

        vlay.addWidget(self.WP)

        self.instruction_LE = QLineEdit(self)
        font = self.instruction_LE.font()      # lineedit current font
        font.setPointSize(10)               # change it's size
        self.instruction_LE.setFont(font)      # set font
        commands = ['setconstrain(\'g1_H\', \'g2_H*2\');',
                    'setvalue(\'g1_H\', 3);',
                    'setlimits(\'g1_H\', \'min\', 5);',
                    'setlimits(\'g1_H\', \'vary\', False);',
                    'printParams()']
        Example = 'Expr..  ' + ' '.join(commands)
        self.instruction_LE.setText(Example)
        expP = QPushButton('Evaluate', self)
        expP.clicked.connect(self.EvalInstr)
        hlay2 = QHBoxLayout()
        hlay2.addWidget(self.instruction_LE)
        hlay2.addWidget(expP)
        vlay.addLayout(hlay2)

        self.keypressed = None

    def showHelp(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Mouse interaction\n"
                    "LMB HM move centroid\n"
                    "RMB HM + change FWHM\n"
                    "LMB HM + 'q' move quadrupolar interaction\n"
                    "LMB HM + 'g' move ground zeeman splitting\n"
                    "LMB HM + 'e' move excited zeeman splitting\n"
                    "MB  VM move amplitude\n")
        msg.setInformativeText("Curve help\n"
                               "PVoight Pseudo Voight\n"
                               "Q quadrupolar interaction splitting\n"
                               "M magnetic interaction splitting\n"
                               "MQ  magnetic interaction splitting +\n"
                               "         quadrupolar interaction shift")
        msg.setWindowTitle("Help")
        helpo_det = ("LMB left mouse button\n"
                     "RMB right mouse button\n"
                     "VM vertical motion\n"
                     "HM horizontal motion\n")
        msg.setDetailedText(helpo_det)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def EvalInstr(self):
        try:
            instruct = str(self.instruction_LE.text()).strip()
            print(instruct, '\n')
            eval(instruct)
        except Exception as err:
            print(err)
            print('??????????????????\n Unknown instruction')
            return

        commands = ['setconstrain(\'g1_H\', \'g2_H*2\');',
                    'setvalue(\'g1_H\', 3);',
                    'setlimits(\'g1_H\', \'min\', 5);',
                    'setlimits(\'g1_H\', \'vary\', False);',
                    'printParams()']
        Example = 'Expr..  ' + ' '.join(commands)
        self.instruction_LE.setText(Example)

    def save_curves(self):
        header = 'x exp_y yerr fit residual'
        fit = Intrct.FitModelEval(x=self.x)
        residual = self.y - fit
        data = np.vstack([self.x,
                          self.y,
                          self.yerr,
                          fit,
                          residual])
        components = Intrct.FitModel.eval_components(x=self.x,
                                                     params=Intrct.FitModel.make_params())
        for c_name, c_y in components.items():
            data = np.vstack([data, c_y])
            header += ' ' + c_name
        name, filters = QFileDialog.getSaveFileName(self, 'Open File')
        np.savetxt(name, data.T, header=header, fmt='%10.5f')

    def save_report(self):
        out = ['Inintial params']
        for i in Intrct.FitModel.make_params().values():
            out.append(str(i))
        out.append('!!!!!' * 5)
        out.append('\n\n\n')
        out = '\n'.join(out)
        name, filters = QFileDialog.getSaveFileName(self, 'Open File')
        with open(name, 'w') as savefile:
            savefile.write(out)
            savefile.write(self.out.fit_report())
        return

    def file_open(self, open=None):
        if open:
            name, filters = QFileDialog.getOpenFileName(self, 'Open File')
            data = np.loadtxt(str(name)).T
            self.x = data[0]
            self.y = data[1]
            if data.shape[0] > 2:
                self.yerr = data[2]
            else:
                self.yerr = np.sqrt(np.abs(data[1]))
        if hasattr(self, 'exp_line'):
            self.exp_line.remove()
            del self.exp_line
        self.exp_line = self.WP.canvas.ax.errorbar(self.x, self.y,
                                                   yerr=self.yerr,
                                                   label='exp.',
                                                   fmt='k',
                                                   capsize=1,
                                                   zorder=0)
        if hasattr(self, 'fit_line'):
            self.WP.canvas.draw()
            return

        Intrct.FitModel.guess(self.y)
        y = Intrct.FitModelEval(self.x)
        self.fit_line, = self.WP.canvas.ax.plot(self.x, y, 'r', label='fit',
                                                linewidth=3)
        bkg = Intrct.FitModel.components[0]
        bkg.line, = self.WP.canvas.ax.plot(self.x, y, label='bkg_')

        self.WP.canvas.ax.legend()
        self.WP.canvas.draw()

    def FitMethod(self):
        out = Intrct.Fit(self.y, self.x, self.yerr)
        bkg = Intrct.bkg_val()
        comps = out.eval_components()
        Intrct.FitModel.components[0].line.set_ydata(bkg)
        for c in Intrct.FitModel.components[1:]:
            c.line.set_ydata(comps[c.prefix] + bkg)
        self.fit_line.set_ydata(out.best_fit)
        self.WP.canvas.draw()
        print('\n' * 3, '!' * 20)
        print(out.fit_report())
        print('!' * 20, '\n' * 3)
        fig = out.plot_fit()
        # fig, gr = out.plot()
        residual = (out.best_fit - self.y) + 1.1 * \
            max(self.y) - 0.1 * min(self.y)
        plt.hlines(y=1.1 * max(self.y) - 0.1 * min(self.y),
                   xmin=min(self.x), xmax=max(self.x), label='')
        plt.plot(out.userkws['x'], residual, label='residual')
        title = ' '.join([i.prefix for i in Intrct.FitModel.components])
        # fig.axes[0].set_title(title)
        plt.title(title)
        plt.legend()
        plt.show()
        self.out = out

    def AddPeak(self):
        PS = str(self.cb_psh.currentText())
        MULT = str(self.cb_mult.currentText())
        MAG = str(self.cb_ms.currentText())
        prefix = self.prefix_LE.text()
        maxim, center = 0, 0
        split, H = 0, 0
        lipo = 0

        def add1(event):
            nonlocal maxim, center, lipo, MULT, MAG
            maxim, center = event.ydata - Intrct.bkg_val(), event.xdata
            lipo, = self.WP.canvas.ax.plot([center], [event.ydata], 'or')
            self.WP.canvas.draw()
            if MAG == "sextet":
                print('\nsecond click on the other peak')
            elif MULT == "doublet":
                print('\nsecond click on second peak')
            else:
                print('\nsecond click FWHM')
            self.WP.canvas.mpl_disconnect(self.p_id)
            self.p_id = self.WP.canvas.mpl_connect('button_press_event', add2)

        def add2(event):
            nonlocal center, split, H, lipo
            self.WP.canvas.mpl_disconnect(self.p_id)
            xpos = event.xdata
            ypos = event.ydata
            lipo.set_data(np.append(lipo.get_xdata(), xpos),
                          np.append(lipo.get_ydata(), ypos))
            self.WP.canvas.draw()
            if (MAG == "sextet") or (MULT == "doublet"):
                center = (center + xpos) / 2
                split = abs(center - xpos) * 2
                self.p_id = self.WP.canvas.mpl_connect('button_press_event',
                                                       add3)
                print('\nthird click FWHM')
                return
            else:
                H = 2 * np.abs(event.xdata - center)
                create()

        def add3(event):
            nonlocal split, H
            self.WP.canvas.mpl_disconnect(self.p_id)
            xpos = event.xdata
            H = min(abs(center - split / 2 - xpos),
                    abs(center + split / 2 - xpos))
            if (MAG == "sextet") and (MULT == "doublet"):
                create(mes=split / 5, mgs=split / 2.5, qs=0)
            elif MAG == "sextet":
                create(mes=split / 5, mgs=split / 2.5)
            elif MULT == "doublet":
                create(qs=split)

        def create(**kwargs):
            nonlocal lipo
            lipo.remove()
            del lipo
            Intrct.add_comp(PS, MULT, MAG, prefix, center, maxim, H, **kwargs)
            Intrct.pre2mod(prefix).iplot(self.x, Intrct.bkg_val(),
                                         self.WP.canvas)
            y = Intrct.FitModelEval(self.x)
            self.fit_line.set_ydata(y)
            self.WP.canvas.draw()
            print('')

        self.p_id = self.WP.canvas.mpl_connect('button_press_event', add1)
        if MAG == "sextet":
            print('\nclick on the first external peak')
        else:
            print('\nclick on the top of a peak')
        return

    def SaveProject(self):
        fname, filters = QFileDialog.getSaveFileName(self, 'Open File')
        out = []
        comps = Intrct.FitModel.components
        out.append({i.prefix: type(i).__name__ for i in comps})
        out.append(Intrct.FitModel.param_hints)
        out.append(self.x)
        out.append(self.y)
        out.append(self.yerr)
        with open(fname, 'wb') as fname:
            pickle.dump(out, fname)
            fname.close()
        pass

    def LoadProject(self):
        fname, filters = QFileDialog.getOpenFileName(self, 'Open File')
        with open(fname, 'rb') as fname:
            out = pickle.load(fname)
            fname.close()

        Intrct.loadFitmodel(out[:2])
        self.x = out[2]
        self.y = out[3]
        self.yerr = out[4]

        self.WP.canvas.ax.cla()

        y = Intrct.FitModelEval(self.x)
        self.fit_line, = self.WP.canvas.ax.plot(self.x, y, 'r', label='fit',
                                                linewidth=3)

        comps = Intrct.FitModel.eval_components(x=self.x)
        for c in Intrct.FitModel.components:
            c.line, = self.WP.canvas.ax.plot(
                self.x, comps[c.prefix], label=c.prefix)
        self.file_open(open=False)

    def RemPeak(self):
        def rem1(event):
            if isinstance(event.artist, plt.Line2D):
                thisline = event.artist
            else:
                return
            prefix = thisline.get_label()
            thisline.remove()
            Intrct.remove_comp(prefix)
            self.WP.canvas.mpl_disconnect(self.pick_id)
            self.pick_id = self.WP.canvas.mpl_connect(
                'pick_event', self.onpick)
            self.fit_line.set_ydata(Intrct.FitModelEval(x=self.x))
            self.WP.canvas.draw()

        self.WP.canvas.mpl_disconnect(self.pick_id)
        self.pick_id = self.WP.canvas.mpl_connect('pick_event', rem1)
        print('Clicked Rem button.')

    def onpick(self, event):

        def move(event, nbutton):
            delta_m = ((event.ydata - bkg - inten_1) / inten_1) + 1
            set_val('amplitude', A_1 * delta_m)

            if nbutton == 1:
                delta = event.xdata - position
                if self.keypressed is None:
                    set_val('center', delta + C_1)

                elif self.keypressed == 'g':
                    if not ('mgs' in comp.param_hints):
                        return
                    if position - C_1 > 0:
                        set_val('center', C_1 + delta / 2)
                        set_val('mgs', M_1 + delta)
                    else:
                        set_val('center', (C_1 + delta / 2))
                        set_val('mgs', M_1 - delta)

                elif self.keypressed == 'e':
                    if not ('mgs' in comp.param_hints):
                        return
                    if position - C_1 > 0:
                        set_val('mgs', M_1 + 3 * delta)
                        set_val('mes', M_2 - delta)
                    else:
                        set_val('mgs', M_1 - 3 * delta)
                        set_val('mes', M_2 + delta)

                elif self.keypressed == 'q':
                    if not ('qs' in comp.param_hints):
                        return
                    elif 'mgs' in comp.param_hints:
                        set_val('center', C_1 + delta / 2)
                        set_val('qs', Q_1 - delta)
                    else:
                        if position - C_1 > 0:
                            set_val('center', C_1 + delta / 2)
                            set_val('qs', Q_1 + delta)
                        else:
                            set_val('center', (C_1 + delta / 2))
                            set_val('qs', Q_1 - delta)

            elif nbutton == 3:
                delta_m = event.xdata - position
                set_val('H', H_1 + delta_m)

            else:
                print('nbutton', nbutton)
                return
            try:
                comp.iplot(x=self.x, bkg_val=bkg, canvas=self.WP.canvas)
            except Exception:
                print('constrain block graph update')
            self.fit_line.set_ydata(Intrct.FitModelEval(x=self.x))
            self.WP.canvas.draw()

        # testget values
        bkg = Intrct.bkg_val()
        comp = Intrct.pre2mod(event.artist.get_label())

        def get_val(x):
            return comp.param_hints[x]['value']

        def set_val(x, y):
            comp.param_hints[x]['value'] = y

        position = event.mouseevent.xdata
        inten_1 = event.mouseevent.ydata - bkg
        C_1 = get_val('center')
        A_1 = get_val('amplitude')
        H_1 = get_val('H')
        if 'qs' in comp.param_hints:
            Q_1 = get_val('qs')
        if 'mes' in comp.param_hints:
            M_1 = get_val('mgs')
            M_2 = get_val('mes')
        but_n = event.mouseevent.button
        self.mid = self.WP.canvas.mpl_connect('motion_notify_event',
                                              lambda x: move(x, but_n.value))
        self.rid = self.WP.canvas.mpl_connect(
            'button_release_event', self.clk_r)
        self.WP.canvas.mpl_disconnect(self.pick_id)

    def clk_r(self, event):
        Intrct.FitParams = Intrct.FitModel.make_params()
        # for i in Intrct.FitParams:
        #     print(Intrct.FitParams[i])
        self.WP.canvas.mpl_disconnect(self.rid)
        self.WP.canvas.mpl_disconnect(self.mid)
        self.pick_id = self.WP.canvas.mpl_connect('pick_event', self.onpick)
        self.keypressed = None

    def keyPressEvent(self, event):
        self.keypressed = event.text()

    def keyReleaseEvent(self, event):
        self.keypressed = None


class WidgetPlot(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = PlotCanvas(self, width=10, height=8)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        fig = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.ax = self.figure.add_subplot(111)


def MossLauncher():
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


def create_icon():
    from pyshortcuts import make_shortcut
    print('\n' * 5, 'create icon', '\n' * 5)

    uname = sys.platform.lower()
    bindir = 'Scripts' if 'win' in uname else 'bin'
    script = os.path.join(sys.exec_prefix, bindir, 'Moss')
    icon = __file__[:-10] + 'Moss.ico'
    make_shortcut(script, name='Moss', icon=icon)


if __name__ == "__main__":
    MossLauncher()
