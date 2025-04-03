import sys

from PIL import Image
import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMainWindow, QGridLayout, \
 QPushButton, QApplication, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QImage, QPen, QPainter, QColor
from PyQt5.QtCore import Qt, QPoint
from numpy import argmax


from NeuralNetwork import NeuralNetwork


class ToolBox(QWidget):
    def __init__(self):
        super().__init__()

        self.setMaximumWidth(150)
        self.setMinimumWidth(150)

        self.vbox = QVBoxLayout()
        self.setLayout(self.vbox)

class CanvasWidget(QWidget):
    def __init__(self ):
        super().__init__()
        self.image = QImage(500,500,QImage.Format.Format_RGB32)
        self.image.fill(Qt.white)
        self.savedImage = QImage(500,500,QImage.Format.Format_RGB32)
        self.drawing = False
        self.brushColor =Qt.darkGray
        self.brushSize= 45
        self.brushStyle = Qt.SolidLine
        self.brushCap = Qt.RoundCap
        self.brushJoin = Qt.RoundJoin

        self.lastPoint = QPoint()

    pass


    def resizeEvent(self, event):
        self.image = self.image.scaled(self.width(), self.height())


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:

            painter = QPainter(self.image)
            painter.setRenderHint(QPainter.RenderHint.HighQualityAntialiasing)
            painter.setPen(QPen(self.brushColor, self.brushSize, self.brushStyle, self.brushCap, self.brushJoin))

            painter.drawPoint(event.pos())
            self.drawing = True
            self.lastPoint = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setRenderHint(QPainter.RenderHint.HighQualityAntialiasing)

            painter.setPen(QPen(self.brushColor, self.brushSize, self.brushStyle, self.brushCap, self.brushJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.savedImage = self.resizeSavedImage
            self.resizeSavedImage = self.image
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
pass


class Window(QMainWindow):

    def setExcpectedNumber(self, number):
        item_color = QTableWidgetItem()
        item_color.setBackground(self.successColor)
        self.table.setItem(number, 1, item_color)


    def get_rgb_from_hex(code):
        code_hex = code.replace("#", "")
        rgb = tuple(int(code_hex[i:i + 2], 16) for i in (0, 2, 4))
        return QColor.fromRgb(rgb[0], rgb[1], rgb[2])

    def clearWindow(self):
        for i in range(10):
            item_color = QTableWidgetItem()
            item_color.setBackground(self.baseColor)
            self.table.setItem(i, 1, item_color)
        self.imageArea.image.fill(Qt.white)
        self.imageArea.update()
        pass

    def guessNumber(self):
        for i in range(10):
            item_color = QTableWidgetItem()
            item_color.setBackground(self.baseColor)
            self.table.setItem(i, 1, item_color)

        prepImage = self.imageArea.image.scaled(28,28)
        imag_name = "image.png"
        prepImage.save(imag_name, format="PNG")

        image = Image.open(imag_name).convert("L")
        img_array = np.array(image)
        img_data = 255.0 - img_array.reshape(784)
        img_data = (img_data / 255.0 * 0.99) + 0.01

        res = argmax(self.n.query(img_data))
        self.setExcpectedNumber(res)

        pass

    def __init__(self, neuralNetwork):
        super().__init__()
        self.setWindowTitle("Neural Network OCR")
        self.setGeometry(100,100,900,600)
        self.setFixedSize(900,600)

        self.imageArea = CanvasWidget()
        self.grid = QGridLayout()
        self.table = QTableWidget()
        self.table.setRowCount(10)
        self.table.setColumnCount(2)
        self.baseColor = Qt.gray
        self.successColor =Qt.green

        self.clearButton = QPushButton("Clear")
        self.clearButton.clicked.connect(self.clearWindow)

        self.guessButton = QPushButton("Guess")
        self.guessButton.clicked.connect(self.guessNumber)
        self.n = NeuralNetwork(1,1,1,1)
        self.n= neuralNetwork

        for i in range(10):
            self.table.setItem(i,0,QTableWidgetItem("{}".format(i)))
            item_color = QTableWidgetItem()
            item_color.setBackground(self.baseColor)
            self.table.setItem(i,1, item_color)

        self.table.setHorizontalHeaderLabels(["Number","Result"])
        self.table.verticalHeader().setVisible(False)

        self.grid.addWidget(self.imageArea, 0, 0, 1, 6)
        self.grid.addWidget(self.table, 0, 6, 1, 2)
        self.grid.addWidget(self.clearButton, 1, 6, 1, 1)
        self.grid.addWidget(self.guessButton, 1, 7, 1, 1)


        win = QWidget()
        win.setLayout(self.grid)
        self.setCentralWidget(win)

    pass

class appGUI:
    def __init__(self, nueralNetwork):

        app = QApplication(sys.argv)
        window = Window(neuralNetwork=nueralNetwork)
        window.show()

        sys.exit(app.exec())


