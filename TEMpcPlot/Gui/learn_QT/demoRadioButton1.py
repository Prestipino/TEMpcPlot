# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demoRadioButton1.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(90, 20, 161, 16))
        self.label.setObjectName("label")
        self.labelFare = QtWidgets.QLabel(Dialog)
        self.labelFare.setGeometry(QtCore.QRect(80, 270, 161, 16))
        self.labelFare.setText("")
        self.labelFare.setObjectName("labelFare")
        self.radioButtonFirstClass = QtWidgets.QRadioButton(Dialog)
        self.radioButtonFirstClass.setGeometry(QtCore.QRect(70, 80, 151, 17))
        self.radioButtonFirstClass.setObjectName("radioButtonFirstClass")
        self.radioButtonBusinessClass = QtWidgets.QRadioButton(Dialog)
        self.radioButtonBusinessClass.setGeometry(QtCore.QRect(70, 100, 131, 17))
        self.radioButtonBusinessClass.setObjectName("radioButtonBusinessClass")
        self.radioButtonEconomyClass = QtWidgets.QRadioButton(Dialog)
        self.radioButtonEconomyClass.setGeometry(QtCore.QRect(70, 120, 161, 17))
        self.radioButtonEconomyClass.setObjectName("radioButtonEconomyClass")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Choose the flight type"))
        self.radioButtonFirstClass.setText(_translate("Dialog", "First Class $150,"))
        self.radioButtonBusinessClass.setText(_translate("Dialog", "Business Class $125"))
        self.radioButtonEconomyClass.setText(_translate("Dialog", "Economy Class $100"))
