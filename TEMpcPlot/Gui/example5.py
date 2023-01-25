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
ax = fig.add_subplot(111, projection='polar')
theta = np.arange(0., 2., 1. / 180.) * np.pi
ax.plot(theta, 5 * np.cos(4 * theta))


frame = QtWidgets.QFrame()
vbox2 = QtWidgets.QVBoxLayout(frame)

n_slider = QtWidgets.QSlider(frame)
n_slider.setRange(3, 10)
n_slider.setOrientation(QtCore.Qt.Horizontal)
n_slider.setObjectName("Slider")
vbox2.addWidget(n_slider)


def close():
    plt.close('all') 
button = QtWidgets.QPushButton("Quit", frame)
button.setGeometry(QtCore.QRect(250, 0, 75, 25))
vbox2.addWidget(button)

#hbox = QtWidgets.QVBoxLayout()


#hbox.addWidget(n_slider)
#hbox.addWidget(button)


vbox = QtWidgets.QHBoxLayout()
vspace = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
vbox.addItem(vspace)
vbox.addSpacing(400)
vbox.addWidget(frame)

fig.canvas.setLayout(vbox)
