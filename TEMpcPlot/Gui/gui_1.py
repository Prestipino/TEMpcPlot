# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1003, 799)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.main_frame = QtWidgets.QFrame(self.centralwidget)
        self.main_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.main_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_frame.setObjectName("main_frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.main_frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_2 = QtWidgets.QFrame(self.main_frame)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.graphicsView = QtWidgets.QGraphicsView(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayout_2.addWidget(self.graphicsView)
        self.frame_3 = QtWidgets.QFrame(self.frame_2)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_Int = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_Int.sizePolicy().hasHeightForWidth())
        self.frame_Int.setSizePolicy(sizePolicy)
        self.frame_Int.setMinimumSize(QtCore.QSize(400, 45))
        self.frame_Int.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_Int.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_Int.setObjectName("frame_Int")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_Int)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.frame_5 = QtWidgets.QFrame(self.frame_Int)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.label_5 = QtWidgets.QLabel(self.frame_5)
        self.label_5.setGeometry(QtCore.QRect(20, 10, 21, 16))
        self.label_5.setObjectName("label_5")
        self.horizontalSlider_2 = QtWidgets.QSlider(self.frame_5)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(50, 10, 84, 22))
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.label_6 = QtWidgets.QLabel(self.frame_5)
        self.label_6.setGeometry(QtCore.QRect(300, 10, 47, 13))
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_3.addWidget(self.frame_5)
        self.label_3 = QtWidgets.QLabel(self.frame_Int)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.horizontalSlider = QtWidgets.QSlider(self.frame_Int)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_3.addWidget(self.horizontalSlider)
        self.label_4 = QtWidgets.QLabel(self.frame_Int)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.verticalLayout_2.addWidget(self.frame_Int)
        self.frame_Int_6 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_Int_6.sizePolicy().hasHeightForWidth())
        self.frame_Int_6.setSizePolicy(sizePolicy)
        self.frame_Int_6.setMinimumSize(QtCore.QSize(400, 45))
        self.frame_Int_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_Int_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_Int_6.setObjectName("frame_Int_6")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.frame_Int_6)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.frame_10 = QtWidgets.QFrame(self.frame_Int_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_10.sizePolicy().hasHeightForWidth())
        self.frame_10.setSizePolicy(sizePolicy)
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.label_23 = QtWidgets.QLabel(self.frame_10)
        self.label_23.setGeometry(QtCore.QRect(20, 10, 21, 16))
        self.label_23.setObjectName("label_23")
        self.horizontalSlider_11 = QtWidgets.QSlider(self.frame_10)
        self.horizontalSlider_11.setGeometry(QtCore.QRect(50, 10, 84, 22))
        self.horizontalSlider_11.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_11.setObjectName("horizontalSlider_11")
        self.label_24 = QtWidgets.QLabel(self.frame_10)
        self.label_24.setGeometry(QtCore.QRect(300, 10, 47, 13))
        self.label_24.setObjectName("label_24")
        self.horizontalLayout_9.addWidget(self.frame_10)
        self.label_25 = QtWidgets.QLabel(self.frame_Int_6)
        self.label_25.setObjectName("label_25")
        self.horizontalLayout_9.addWidget(self.label_25)
        self.horizontalSlider_12 = QtWidgets.QSlider(self.frame_Int_6)
        self.horizontalSlider_12.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_12.setObjectName("horizontalSlider_12")
        self.horizontalLayout_9.addWidget(self.horizontalSlider_12)
        self.label_26 = QtWidgets.QLabel(self.frame_Int_6)
        self.label_26.setObjectName("label_26")
        self.horizontalLayout_9.addWidget(self.label_26)
        self.verticalLayout_2.addWidget(self.frame_Int_6)
        self.frame_Int_2 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_Int_2.sizePolicy().hasHeightForWidth())
        self.frame_Int_2.setSizePolicy(sizePolicy)
        self.frame_Int_2.setMinimumSize(QtCore.QSize(400, 45))
        self.frame_Int_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_Int_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_Int_2.setObjectName("frame_Int_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_Int_2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_6 = QtWidgets.QFrame(self.frame_Int_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.label_7 = QtWidgets.QLabel(self.frame_6)
        self.label_7.setGeometry(QtCore.QRect(20, 10, 21, 16))
        self.label_7.setObjectName("label_7")
        self.horizontalSlider_3 = QtWidgets.QSlider(self.frame_6)
        self.horizontalSlider_3.setGeometry(QtCore.QRect(50, 10, 84, 22))
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.label_8 = QtWidgets.QLabel(self.frame_6)
        self.label_8.setGeometry(QtCore.QRect(300, 10, 47, 13))
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_4.addWidget(self.frame_6)
        self.label_9 = QtWidgets.QLabel(self.frame_Int_2)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_4.addWidget(self.label_9)
        self.horizontalSlider_4 = QtWidgets.QSlider(self.frame_Int_2)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.horizontalLayout_4.addWidget(self.horizontalSlider_4)
        self.label_10 = QtWidgets.QLabel(self.frame_Int_2)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_4.addWidget(self.label_10)
        self.verticalLayout_2.addWidget(self.frame_Int_2)
        self.frame_Int_3 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_Int_3.sizePolicy().hasHeightForWidth())
        self.frame_Int_3.setSizePolicy(sizePolicy)
        self.frame_Int_3.setMinimumSize(QtCore.QSize(400, 45))
        self.frame_Int_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_Int_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_Int_3.setObjectName("frame_Int_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_Int_3)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.frame_7 = QtWidgets.QFrame(self.frame_Int_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.label_11 = QtWidgets.QLabel(self.frame_7)
        self.label_11.setGeometry(QtCore.QRect(20, 10, 21, 16))
        self.label_11.setObjectName("label_11")
        self.horizontalSlider_5 = QtWidgets.QSlider(self.frame_7)
        self.horizontalSlider_5.setGeometry(QtCore.QRect(50, 10, 84, 22))
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setObjectName("horizontalSlider_5")
        self.label_12 = QtWidgets.QLabel(self.frame_7)
        self.label_12.setGeometry(QtCore.QRect(300, 10, 47, 13))
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_6.addWidget(self.frame_7)
        self.label_13 = QtWidgets.QLabel(self.frame_Int_3)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_6.addWidget(self.label_13)
        self.horizontalSlider_6 = QtWidgets.QSlider(self.frame_Int_3)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")
        self.horizontalLayout_6.addWidget(self.horizontalSlider_6)
        self.label_14 = QtWidgets.QLabel(self.frame_Int_3)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_6.addWidget(self.label_14)
        self.verticalLayout_2.addWidget(self.frame_Int_3)
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_2.setMaximumSize(QtCore.QSize(75, 16777215))
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.pushButton_2)
        spacerItem = QtWidgets.QSpacerItem(20, 337, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.frame_Int_4 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_Int_4.sizePolicy().hasHeightForWidth())
        self.frame_Int_4.setSizePolicy(sizePolicy)
        self.frame_Int_4.setMinimumSize(QtCore.QSize(400, 45))
        self.frame_Int_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_Int_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_Int_4.setObjectName("frame_Int_4")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_Int_4)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.frame_8 = QtWidgets.QFrame(self.frame_Int_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_8.sizePolicy().hasHeightForWidth())
        self.frame_8.setSizePolicy(sizePolicy)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.label_15 = QtWidgets.QLabel(self.frame_8)
        self.label_15.setGeometry(QtCore.QRect(20, 10, 21, 16))
        self.label_15.setObjectName("label_15")
        self.horizontalSlider_7 = QtWidgets.QSlider(self.frame_8)
        self.horizontalSlider_7.setGeometry(QtCore.QRect(50, 10, 84, 22))
        self.horizontalSlider_7.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_7.setObjectName("horizontalSlider_7")
        self.label_16 = QtWidgets.QLabel(self.frame_8)
        self.label_16.setGeometry(QtCore.QRect(300, 10, 47, 13))
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_7.addWidget(self.frame_8)
        self.label_17 = QtWidgets.QLabel(self.frame_Int_4)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_7.addWidget(self.label_17)
        self.horizontalSlider_8 = QtWidgets.QSlider(self.frame_Int_4)
        self.horizontalSlider_8.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_8.setObjectName("horizontalSlider_8")
        self.horizontalLayout_7.addWidget(self.horizontalSlider_8)
        self.label_18 = QtWidgets.QLabel(self.frame_Int_4)
        self.label_18.setObjectName("label_18")
        self.horizontalLayout_7.addWidget(self.label_18)
        self.verticalLayout_2.addWidget(self.frame_Int_4)
        self.horizontalLayout_2.addWidget(self.frame_3)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame = QtWidgets.QFrame(self.main_frame)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.spinBox = QtWidgets.QSpinBox(self.frame)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout.addWidget(self.spinBox)
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.spinBox_2 = QtWidgets.QSpinBox(self.frame)
        self.spinBox_2.setObjectName("spinBox_2")
        self.horizontalLayout.addWidget(self.spinBox_2)
        self.checkBox = QtWidgets.QCheckBox(self.frame)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout.addWidget(self.checkBox)
        spacerItem1 = QtWidgets.QSpacerItem(667, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addWidget(self.frame)
        self.horizontalLayout_5.addWidget(self.main_frame)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1003, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionopen_sqi = QtWidgets.QAction(MainWindow)
        self.actionopen_sqi.setObjectName("actionopen_sqi")
        self.actionsave = QtWidgets.QAction(MainWindow)
        self.actionsave.setObjectName("actionsave")
        self.menuFile.addAction(self.actionopen_sqi)
        self.menuFile.addAction(self.actionsave)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_5.setText(_translate("MainWindow", "Int"))
        self.label_6.setText(_translate("MainWindow", "TextLabel"))
        self.label_3.setText(_translate("MainWindow", "Int"))
        self.label_4.setText(_translate("MainWindow", "TextLabel"))
        self.label_23.setText(_translate("MainWindow", "Int"))
        self.label_24.setText(_translate("MainWindow", "TextLabel"))
        self.label_25.setText(_translate("MainWindow", "Int"))
        self.label_26.setText(_translate("MainWindow", "TextLabel"))
        self.label_7.setText(_translate("MainWindow", "Int"))
        self.label_8.setText(_translate("MainWindow", "TextLabel"))
        self.label_9.setText(_translate("MainWindow", "Int"))
        self.label_10.setText(_translate("MainWindow", "TextLabel"))
        self.label_11.setText(_translate("MainWindow", "Int"))
        self.label_12.setText(_translate("MainWindow", "TextLabel"))
        self.label_13.setText(_translate("MainWindow", "Int"))
        self.label_14.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_2.setText(_translate("MainWindow", "PushButton"))
        self.label_15.setText(_translate("MainWindow", "Int"))
        self.label_16.setText(_translate("MainWindow", "TextLabel"))
        self.label_17.setText(_translate("MainWindow", "Int"))
        self.label_18.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton.setText(_translate("MainWindow", "PushButton"))
        self.label.setText(_translate("MainWindow", "px"))
        self.label_2.setText(_translate("MainWindow", "ref"))
        self.checkBox.setText(_translate("MainWindow", "Optimize \n"
" scale"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionopen_sqi.setText(_translate("MainWindow", "open sqi"))
        self.actionsave.setText(_translate("MainWindow", "save "))