import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.backends.qt_compat import QtCore, QtWidgets,  QtGui
#print(QtCore.qVersion())
#if is_pyqt5():
from matplotlib.backends.backend_qt5agg import FigureCanvas
#else:
#    from matplotlib.backends.backend_qt4agg import FigureCanvas


fig = plt.figure(figsize=(8, 6), dpi=100)



def update():
    n = n_slider.value()
    ax.clear()
    ax.plot(theta, 5 * np.cos(n * theta))
    fig.canvas.draw()

n_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
n_slider.setRange(3, 10)
n_slider.setSingleStep(1)
n_slider.setValue(4)
n_slider.setMinimumSize(QtCore.QSize(100, 15))
n_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
n_slider.setTickInterval(1)
n_slider.setFont(QtGui.QFont("Arial", 30))

n_slider.sliderReleased.connect(update)

def close():
    plt.close('all') 

button = QtWidgets.QPushButton("Quit")
button.setGeometry(QtCore.QRect(250, 0, 75, 25))


main = fig.canvas.parent()

hbox = QtWidgets.QVBoxLayout()
hbox.addWidget(n_slider)
hbox.addWidget(button)

vbox = QtWidgets.QHBoxLayout()
vspace = QtWidgets.QSpacerItem(1, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
vbox.addItem(vspace)
vbox.addSpacing(0)
vbox.addLayout(hbox)

ax = fig.add_subplot(111)
theta = np.arange(0., 2., 1. / 180.) * np.pi
ax.plot(theta, 5 * np.cos(4 * theta))
fig.canvas.setLayout(vbox)
fig.canvas.setSpacing(53)