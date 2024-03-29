import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure

from matplotlib.backends.qt_compat import QtCore, QtWidgets,  QtGui

from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)





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
        self.frame.setMinimumSize(QtCore.QSize(400, 45))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label_25")
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


class Stepslider(QtWidgets.QSlider):
    floatValueChanged = QtCore.pyqtSignal(float)
    def __init__(self, minimum, maximum, step, parent=None):
        super(StepDial, self).__init__(parent)
        self.scaleValues = []
        self.minimumFloat = minimum
        self.step = step
        self.setMaximum((maximum - minimum) // step)
        self.valueChanged.connect(self.computeFloat)

    def computeFloat(self, value):
        self.floatValueChanged.emit(value * self.step + self.minimumFloat)

    def setFloatValue(self, value):
        # compute the index by subtracting minimum from the value, divided by the
        # step value, then round provide a *rounded* value, otherwise values like
        # 0.9999[...] will be interpreted as 0
        index = (value - self.minimumFloat) / self.step
        self.setValue(int(round(index)))


class StepDial(QtWidgets.QDial):
    floatValueChanged = QtCore.pyqtSignal(float)
    def __init__(self, minimum, maximum, step, parent=None):
        super(StepDial, self).__init__(parent)
        self.scaleValues = []
        self.minimumFloat = minimum
        self.step = step
        self.setMaximum((maximum - minimum) // step)
        self.valueChanged.connect(self.computeFloat)

    def computeFloat(self, value):
        self.floatValueChanged.emit(value * self.step + self.minimumFloat)

    def setFloatValue(self, value):
        # compute the index by subtracting minimum from the value, divided by the
        # step value, then round provide a *rounded* value, otherwise values like
        # 0.9999[...] will be interpreted as 0
        index = (value - self.minimumFloat) / self.step
        self.setValue(int(round(index)))


class FloatDial(QtWidgets.QDial):
    floatValueChanged = QtCore.pyqtSignal(float)
    def __init__(self, minimum, maximum, stepCount=1001, parent=None):
        super(FloatDial, self).__init__(parent)
        self.minimumFloat = minimum
        self.maximumFloat = maximum
        self.floatRange = maximum - minimum
        # since QDial (as all QAbstractSlider subclasses), includes its maximum()
        # in its value range; to get the expected step count, we subtract 1 from
        # it: maximum = minimum + (stepCount - 1)
        # also, since the default minimum() == 0, the maximum is stepCount - 1.
        # Note that QDial is usually symmetrical, using 240 degrees
        # counterclockwise (in cartesian coordinates, as in 3-o'clock) for the
        # minimum, and -60 degrees for its maximum; with this implementation we
        # assume all of that, and get a correct "middle" value, but this requires
        # an *odd* stepCount number so that the odd "middle" index value is
        # actually what it should be.
        self.stepCount = stepCount
        self.setMaximum(stepCount - 1)
        self.valueChanged.connect(self.computeFloat)

    def computeFloat(self, value):
        ratio = float(value) / self.maximum()
        self.floatValueChanged.emit(self.floatRange * ratio + self.minimumFloat)

    def setFloatValue(self, value):
        # compute the "step", based on the stepCount then use the same concept
        # as in the StepDial.setFloatValue function
        step = (self.maximumFloat - self.minimumFloat) / self.stepCount
        index = (value - self.minimumFloat) // step
        self.setValue(int(round(index)))


import numpy as np
class mplwidget(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.main = QtWidgets.QWidget()
        self.setCentralWidget(self.main)
        layout = QtWidgets.QVBoxLayout(self.main)

        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.


        self._static_ax = static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._static_ax.plot(t, np.tan(t), ".")
        static_canvas.draw()
        #layout.addWidget(NavigationToolbar(static_canvas, self))
        layout.addWidget(static_canvas)
