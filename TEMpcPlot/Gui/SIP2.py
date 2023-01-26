import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5, QtGui
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure




class C3_slider():
    def __init__(self, parent, layout, label, minimum=None, maximum=None, value=None):
        self.min = minimum if minimum else 0
        self.max = maximum if maximum else 100
        self.frame = QtWidgets.QFrame(parent)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(400, 35))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label = QtWidgets.QLabel(self.frame)
        self.horizontalLayout_9.addWidget(self.label)

        self.Slider = QtWidgets.QSlider(self.frame)
        self.Slider.setRange(0, 1000)
        self.Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Slider.setObjectName("Slider")
        self.Slider.valueChanged.connect(self.update)
        self.horizontalLayout_9.addWidget(self.Slider)

        # creating a label
        self.labelx = QtWidgets.QLabel("", self.frame)
        # setting geometry to the label
        self.labelx.setGeometry(200, 100, 300, 80)
        # getting current position of the slider
        value = self.Slider.sliderPosition()
        self.horizontalLayout_9.addWidget(self.labelx)

        #self.lcdNumber = QtWidgets.QLCDNumber(self.frame)
        #self.lcdNumber.setObjectName("lcdNumber_2")
        #self.horizontalLayout_9.addWidget(self.lcdNumber)
        layout.addWidget(self.frame)
        self.label.setText(label)
        #self.Slider.valueChanged['int'].connect(self.lcdNumber.display)

    def update(self, value):
        step = (self.max - self.min) / 1000
        self.labelx.setText(f'{self.min + value * step:.2f}')

    def set_Range(self, minimum, maximum):
        self.min = minimum
        self.max = maximum

    def get_value(self):
        step = (self.max - self.min) / 1000
        value = self.Slider.sliderPosition()
        return self.min + value * step

    def set_value(self, value):
        step = (self.max - self.min) / 1000
        self.Slider.setValue((value - self.min) / step)









class SeqImaPlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layoutH1 = QtWidgets.QHBoxLayout(self._main)

        ###################################################
        layout_Figure = QtWidgets.QVBoxLayout()
        layoutH1.addLayout(layout_Figure)
        # a figure instance to plot on
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        layout_Figure.addWidget(self.toolbar)
        layout_Figure.addWidget(self.canvas)


        layout_Commands = QtWidgets.QVBoxLayout()
        layoutH1.addLayout(layout_Commands)

        layout_GBut = QtWidgets.QHBoxLayout()
        layout_GBut.setSpacing(0)
        layout_Commands.addLayout(layout_GBut)
        self.downBut = QtWidgets.QPushButton(QtGui.QIcon("down.png"), '')
        layout_GBut.addWidget(self.downBut)
        self.upBut = QtWidgets.QPushButton(QtGui.QIcon("up.png"), '')
        layout_GBut.addWidget(self.upBut)
        layout_GBut.addSpacing(5)
        self.PeaksBut = QtWidgets.QPushButton(QtGui.QIcon("PlotP.png"), '')
        layout_GBut.addWidget(self.PeaksBut)
        self.RemBut = QtWidgets.QPushButton(QtGui.QIcon("RemP.png"), '')
        layout_GBut.addWidget(self.RemBut)
        self.RanBut = QtWidgets.QPushButton(QtGui.QIcon("RanP.png"), '')
        layout_GBut.addWidget(self.RanBut)
        layout_GBut.addSpacing(5)
        self.lenghBut = QtWidgets.QPushButton(QtGui.QIcon("lenght.png"), '')
        layout_GBut.addWidget(self.lenghBut)        
        self.angleBut = QtWidgets.QPushButton(QtGui.QIcon("angle.png"), '')
        layout_GBut.addWidget(self.angleBut)        

        # Sliders
        self.Int_sl = C3_slider(None, layout_Commands, 'Inte  ', 0.01, 10.0,5)
        self.dist_sl = C3_slider(None, layout_Commands,'Dist  ', 0.0, 1.0, 0.9) 
        self.rad_sl = C3_slider(None, layout_Commands, 'Radi  ', 0.01, 10.0, 1) 
        self.sym_sl = C3_slider(None, layout_Commands, 'Symm ', 0, 20.0, 0) 

        self.applyalButton = QtWidgets.QPushButton("apply to all")
        #self.applyalButton.setSizePolicy(QtWidgets.QSizePolicy.Minimum,QtWidgets.QSizePolicy.Minimum,)
        self.applyalButton.setMaximumSize(QtCore.QSize(75, 167))
        layout_Commands.addWidget(self.applyalButton)

        vspace = QtWidgets.QSpacerItem(5, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        layout_Commands.addItem(vspace)

        self.vmax_sl = C3_slider(None, layout_Commands, 'Contrast  ', 0.01, 100, 50)
        layout_Commands.addSpacing(15)


        self.RecBut = QtWidgets.QPushButton('Build Reciprocal space')
        layout_Commands.addWidget(self.RecBut)


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = SeqImaPlot()
    app.show()
    app.activateWindow()
    app.raise_()
    #qapp.exec()