from PySide6.QtWidgets import QMainWindow, QPushButton, QLabel

class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("Object Detection System")
        self.detectButton = QPushButton("Detect", MainWindow)
        self.detectButton.setGeometry(50, 50, 100, 30)
        self.detectionLabel = QLabel(MainWindow)
        self.detectionLabel.setGeometry(50, 100, 640, 480)
        self.statsLabel = QLabel(MainWindow)
        self.statsLabel.setGeometry(700, 100, 300, 480)
