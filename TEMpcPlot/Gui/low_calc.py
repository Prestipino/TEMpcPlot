# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'low_calc.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

class C3_slider():
    def __init__(self, parent, layout, label, minimum=None, maximum=None, value=None):
        self.min = minimum if minimum else 0
        self.max = maximum if maximum else 100
        self.frame = QtWidgets.QFrame(parent)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
        self.frame.sizePolicy().hasHeightForWidth())
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
        self.horizontalLayout_9.addWidget(self.labelx)

        #self.lcdNumber = QtWidgets.QLCDNumber(self.frame)
        # self.lcdNumber.setObjectName("lcdNumber_2")
        # self.horizontalLayout_9.addWidget(self.lcdNumber)
        layout.addWidget(self.frame)
        self.label.setText(label)
        if value is not None:
            self.set_value(value)
        # self.Slider.valueChanged['int'].connect(self.lcdNumber.display)

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
        val = (value - self.min) / step
        self.Slider.setValue(val)

class Bottom_create(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Minimum)

        self.setSizePolicy(sizePolicy)
        self.setMaximumSize(QtCore.QSize(5000, 120))
        horiLay = QtWidgets.QHBoxLayout(self)
        groupBox = QtWidgets.QGroupBox('hkl layer')
        groupBox.setSizePolicy(sizePolicy)
        horiLay.addWidget(groupBox)

        horLay_hkl = QtWidgets.QHBoxLayout(groupBox)
        lab1 = QtWidgets.QLabel("h/k/l")
        horLay_hkl.addWidget(lab1)
        self.liEd_hkl = QtWidgets.QLineEdit()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        self.liEd_hkl.setMaximumSize(QtCore.QSize(30, 20))
        self.liEd_hkl.setSizePolicy(sizePolicy)
        horLay_hkl.addWidget(self.liEd_hkl)

        lab2 = QtWidgets.QLabel("   n.")
        horLay_hkl.addWidget(lab2)
        self.liEd_hkln = QtWidgets.QLineEdit()
        self.liEd_hkln.setSizePolicy(sizePolicy)
        self.liEd_hkln.setMaximumSize(QtCore.QSize(30, 20))
        horLay_hkl.addWidget(self.liEd_hkln)

        self.But_hkl = QtWidgets.QPushButton("Create")
        horiLay.addWidget(self.But_hkl)

        self.cBox_peak = QtWidgets.QCheckBox("show peak   \nposition")
        horiLay.addWidget(self.cBox_peak)
        self.cBox_int = QtWidgets.QCheckBox('show\n intensity')
        horiLay.addWidget(self.cBox_int)

        self.cBox_mir = QtWidgets.QCheckBox('mirror')
        horiLay.addWidget(self.cBox_mir)

        self.dial = self.Int_sl = C3_slider(None, horiLay, 'spot size', 0.0, 2.0, 0.4)
        