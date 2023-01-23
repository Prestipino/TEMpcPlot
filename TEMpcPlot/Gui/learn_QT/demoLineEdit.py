# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demoLineEdit.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.labelResponse = QtWidgets.QLabel(Dialog)
        self.labelResponse.setGeometry(QtCore.QRect(0, 20, 131, 16))
        self.labelResponse.setObjectName("labelResponse")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 80, 47, 13))
        self.label.setObjectName("label")
        self.lineEditName = QtWidgets.QLineEdit(Dialog)
        self.lineEditName.setGeometry(QtCore.QRect(120, 20, 113, 20))
        self.lineEditName.setObjectName("lineEditName")
        self.ButtonClickMe = QtWidgets.QPushButton(Dialog)
        self.ButtonClickMe.setGeometry(QtCore.QRect(100, 130, 75, 23))
        self.ButtonClickMe.setObjectName("ButtonClickMe")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.labelResponse.setText(_translate("Dialog", "Enter your Name"))
        self.label.setText(_translate("Dialog", "TextLabel"))
        self.ButtonClickMe.setText(_translate("Dialog", "Click"))
