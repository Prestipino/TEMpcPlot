from PyQt5.QtWidgets import QApplication, QLabel














app = QApplication([])
app.setStyle('Fusion')
label = QLabel('Hello World!')
label.show()
app.exec()