from bisect import bisect_left
from PyQt5 import QtCore, QtGui, QtWidgets



class FixedValuesDial(QtWidgets.QDial):
    floatValueChanged = QtCore.pyqtSignal(float)
    def __init__(self, valueList):
        super(FixedValuesDial, self).__init__()
        self.valueList = valueList
        self.setMaximum(len(valueList) - 1)
        self.valueChanged.connect(self.computeFloat)
        self.currentFloatValue = self.value()

    def computeFloat(self, value):
        self.currentFloatValue = self.valueList[value]
        self.floatValueChanged.emit(self.currentFloatValue)

    def setFloatValue(self, value):
        try:
            # set the index, assuming the value is in the valueList
            self.setValue(self.valueList.index(value))
        except:
            # find the most closest index, based on the value
            index = bisect_left(self.valueList, value)
            if 0 < index < len(self.valueList):
                before = self.valueList[index - 1]
                after = self.valueList[index]
                # bisect_left returns the position where the value would be
                # added, assuming valueList is sorted; the resulting index is the
                # one preceding the one with a value greater or equal to the
                # provided value, so if the difference between the next value and
                # the current is greater than the difference between the previous
                # and the current, the index is closest to the previous
                if after - value > value - before:
                    index -= 1
            # note that the value -the new index- is actually updated only *if*
            # the new index is different from the current, otherwise there will
            # no valueChanged signal emission
            self.setValue(index)


class StepDial(QtWidgets.QDial):
    floatValueChanged = QtCore.pyqtSignal(float)
    def __init__(self, minimum, maximum, step):
        super(StepDial, self).__init__()
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
    def __init__(self, minimum, maximum, stepCount=1001):
        super(FloatDial, self).__init__()
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


class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        layout.addWidget(QtWidgets.QLabel('List based'), 0, 0)
        self.listDial = FixedValuesDial([0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2])
        layout.addWidget(self.listDial, 1, 0)
        self.listDial.floatValueChanged.connect(self.updateListValue)
        self.listSpin = QtWidgets.QDoubleSpinBox(
            minimum=0.02, maximum=0.2, singleStep=0.01)
        layout.addWidget(self.listSpin, 2, 0)
        self.listSpin.valueChanged.connect(self.listDial.setFloatValue)

        layout.addWidget(QtWidgets.QLabel('Step precision (0.02)'), 0, 1)
        self.stepDial = StepDial(0.02, 0.2, 0.02)
        layout.addWidget(self.stepDial, 1, 1)
        self.stepDial.floatValueChanged.connect(self.updateStepDisplay)
        self.stepSpin = QtWidgets.QDoubleSpinBox(
            minimum=0.02, maximum=0.2, singleStep=0.02)
        layout.addWidget(self.stepSpin, 2, 1)
        self.stepSpin.valueChanged.connect(self.stepDial.setFloatValue)

        layout.addWidget(QtWidgets.QLabel('Step count (21 steps)'), 0, 2)
        # see the FloatDial implementation above to understand the reason of odd
        # numbered steps
        self.floatDial = FloatDial(0.02, 0.2, 21)
        layout.addWidget(self.floatDial, 1, 2)
        self.floatDial.floatValueChanged.connect(self.updateFloatValue)
        self.floatSpin = QtWidgets.QDoubleSpinBox(
            minimum=0.02, maximum=0.2, decimals=5, singleStep=0.001)
        layout.addWidget(self.floatSpin, 2, 2)
        self.floatSpin.valueChanged.connect(self.floatDial.setFloatValue)

    def updateStepDisplay(self, value):
        # prevent the spinbox sending valuechanged while updating its value,
        # otherwise you might face an infinite recursion caused by the spinbox
        # trying to update the dial, which will correct the value and possibly
        # send the floatValueChanged back again to it; obviously, this applies
        # to the following slots
        self.stepSpin.blockSignals(True)
        self.stepSpin.setValue(value)
        self.stepSpin.blockSignals(False)

    def updateFloatValue(self, value):
        self.floatSpin.blockSignals(True)
        self.floatSpin.setValue(value)
        self.floatSpin.blockSignals(False)

    def updateListValue(self, value):
        self.listSpin.blockSignals(True)
        self.listSpin.setValue(value)
        self.listSpin.blockSignals(False)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())