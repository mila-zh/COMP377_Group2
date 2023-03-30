import sys
import cv2
import numpy as np
#import tensorflow as tf
from PyQt5 import QtWidgets, QtGui, QtCore

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window title and size
        self.setWindowTitle("Image Recognition App")
        self.setGeometry(100, 100, 800, 600)

        # Create a label to display the image
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(50, 50, 700, 500))

        # Create a button to load the image
        self.button = QtWidgets.QPushButton("Load Image", self)
        self.button.setGeometry(QtCore.QRect(350, 10, 100, 30))
        self.button.clicked.connect(self.load_image)

        # Load the model
        self.model = tf.keras.models.load_model("model.h5")

        # Load the labels
        with open("labels.txt", "r") as f:
            self.labels = f.read().splitlines()

        # Load the image
        self.image = None

    def load_image(self):
        # Open a file dialog to select an image file
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg)")
        if file_dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Read the selected image file and display it in the label
            filename = file_dialog.selectedFiles()[0]
            self.image = cv2.imread(filename)
            self.display_image()

            # Preprocess the image and make a prediction
            image = cv2.resize(self.image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            prediction = self.model.predict(image)
            label = self.labels[np.argmax(prediction)]
            QtWidgets.QMessageBox.information(self, "Prediction", f"The image is classified as {label}.")

    def display_image(self):
        # Convert the image to the RGB color space and resize it to fit the label
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.label.width(), self.label.height()))

        # Create a Qt image from the OpenCV image and display it in the label
        qimage = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
