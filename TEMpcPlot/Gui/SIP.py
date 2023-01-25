import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5, QtGui
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvas
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5, QtGui
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


class   SeqImaPlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layoutH1 = QtWidgets.QHBoxLayout(self._main)

        layout_fig = QtWidgets.QVBoxLayout(self._main)
        self.canvas = FigureCanvas(Figure(figsize=(4, 4)))
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        layout_fig.addWidget(NavigationToolbar(self.canvas, self))
        layout_fig.addWidget(self.canvas)


        layout_commands = QtWidgets.QVBoxLayout()
        
        self.frame_3 = QtWidgets.QFrame()
        self.frame_3 = QtWidgets.QFrame()
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSpacing(1)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.downBut = QtWidgets.QPushButton(self.frame_3)
        self.downBut.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.downBut.sizePolicy().hasHeightForWidth())
        self.downBut.setText("")
        self.downBut.setIcon(QtGui.QIcon("down.png"))
        self.downBut.setObjectName("downBut")
        self.horizontalLayout_5.addWidget(self.downBut)
        self.upBut = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.upBut.sizePolicy().hasHeightForWidth())
        self.upBut.setSizePolicy(sizePolicy)
        self.upBut.setMinimumSize(QtCore.QSize(30, 30))
        self.upBut.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("UP.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.upBut.setIcon(icon1)
        self.upBut.setObjectName("upBut")
        self.horizontalLayout_5.addWidget(self.upBut)
        self.line = QtWidgets.QFrame(self.frame_3)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_5.addWidget(self.line)
        self.PeaksBut = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PeaksBut.sizePolicy().hasHeightForWidth())
        self.PeaksBut.setSizePolicy(sizePolicy)
        self.PeaksBut.setMinimumSize(QtCore.QSize(30, 30))
        self.PeaksBut.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("PlotP.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.PeaksBut.setIcon(icon2)
        self.PeaksBut.setObjectName("PeaksBut")
        self.horizontalLayout_5.addWidget(self.PeaksBut)
        self.RemBut = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RemBut.sizePolicy().hasHeightForWidth())
        self.RemBut.setSizePolicy(sizePolicy)
        self.RemBut.setMinimumSize(QtCore.QSize(30, 30))
        self.RemBut.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("RemP.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.RemBut.setIcon(icon3)
        self.RemBut.setObjectName("RemBut")
        self.horizontalLayout_5.addWidget(self.RemBut)
        self.RanBut = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RanBut.sizePolicy().hasHeightForWidth())
        self.RanBut.setSizePolicy(sizePolicy)
        self.RanBut.setMinimumSize(QtCore.QSize(30, 30))
        self.RanBut.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("RanP.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.RanBut.setIcon(icon4)
        self.RanBut.setObjectName("RanBut")
        self.horizontalLayout_5.addWidget(self.RanBut)
        self.line_2 = QtWidgets.QFrame(self.frame_3)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout_5.addWidget(self.line_2)
        self.lenghBut = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lenghBut.sizePolicy().hasHeightForWidth())
        self.lenghBut.setSizePolicy(sizePolicy)
        self.lenghBut.setMinimumSize(QtCore.QSize(30, 30))
        self.lenghBut.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("lenght.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lenghBut.setIcon(icon5)
        self.lenghBut.setObjectName("lenghBut")
        self.horizontalLayout_5.addWidget(self.lenghBut)
        self.angleBut = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.angleBut.sizePolicy().hasHeightForWidth())
        self.angleBut.setSizePolicy(sizePolicy)
        self.angleBut.setMinimumSize(QtCore.QSize(30, 30))
        self.angleBut.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("angle.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.angleBut.setIcon(icon6)
        self.angleBut.setObjectName("angleBut")
        self.horizontalLayout_5.addWidget(self.angleBut)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.frame_Int = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_Int.sizePolicy().hasHeightForWidth())

        # Sliders
        self.Int_sl = C3_slider(self.frame_3, self.verticalLayout, 'Inte')
        self.dist_sl = C3_slider(self.frame_3, self.verticalLayout, 'Dist') 
        self.rad_sl = C3_slider(self.frame_3, self.verticalLayout, 'Radi') 
        self.sym_sl = C3_slider(self.frame_3, self.verticalLayout, 'Symm') 

        self.applyalButton = QtWidgets.QPushButton(self.frame_3)
        self.applyalButton.setMaximumSize(QtCore.QSize(75, 16777215))
        self.applyalButton.setObjectName("applyalButton")
        self.verticalLayout.addWidget(self.applyalButton)

        spacerItem = QtWidgets.QSpacerItem(20, 337, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.frame_contrast = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_contrast.sizePolicy().hasHeightForWidth())
        self.frame_contrast.setSizePolicy(sizePolicy)
        self.frame_contrast.setMinimumSize(QtCore.QSize(400, 45))
        self.frame_contrast.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_contrast.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_contrast.setObjectName("frame_contrast")

        self.vmax_sl = C3_slider(self.frame_contrast, self.verticalLayout, 'Contrast') 

        self.frame_3D = QtWidgets.QFrame(self.frame_3)
        self.frame_3D.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3D.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3D.setObjectName("frame_3D")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame_3D)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.pushButton = QtWidgets.QPushButton(self.frame_3D)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_8.addWidget(self.pushButton)
        self.label = QtWidgets.QLabel(self.frame_3D)
        self.label.setObjectName("label")
        self.horizontalLayout_8.addWidget(self.label)
        self.spinBox = QtWidgets.QSpinBox(self.frame_3D)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout_8.addWidget(self.spinBox)
        self.label_2 = QtWidgets.QLabel(self.frame_3D)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_8.addWidget(self.label_2)
        self.spinBox_2 = QtWidgets.QSpinBox(self.frame_3D)
        self.spinBox_2.setObjectName("spinBox_2")
        self.horizontalLayout_8.addWidget(self.spinBox_2)
        self.checkBox = QtWidgets.QCheckBox(self.frame_3D)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_8.addWidget(self.checkBox)
        self.verticalLayout.addWidget(self.frame_3D)

        layoutH1.addLayout(layout_fig)
        layoutH1.addLayout(layout_commands)


if __name__ == "__main__":
    import sys
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = SeqImaPlot()
    app.show()
    app.activateWindow()
    app.raise_()
    #sys.exit(app.exec_())
    qapp.exec()


