import sys
import cv2
import numpy as np
import requests
import threading
import datetime
import time
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QTableWidget,
    QTableWidgetItem, QComboBox, QPushButton, QSlider, QFormLayout, QLabel, QFileDialog
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import urllib3

# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class VideoStreamWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.image_label = QLabel()
        self.fps_label = QLabel("FPS: 0")
        self.table_widget = QTableWidget()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.history_table = QTableWidget()
        self.model_selector = QComboBox()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.load_model_button = QPushButton("Load Model")
        self.batch_load_button = QPushButton("Load Batch Images")

        self.confidence_threshold = 0.5
        self.model_name = "SSD"

        self.setup_ui()
        self.setup_camera()
        self.setup_timers()
        self.prev_time = time.time()

        self.detection_results = []
        self.detection_history = []

        self.detection_thread = threading.Thread(target=self.detect_objects, daemon=True)
        self.detection_thread.start()

    def setup_ui(self):
        self.model_selector.addItems(["SSD", "YOLOv5", "YOLOv8"])

        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(int(self.confidence_threshold * 100))
        self.confidence_slider.valueChanged.connect(self.update_confidence_threshold)

        self.load_model_button.clicked.connect(self.load_model)
        self.batch_load_button.clicked.connect(self.load_batch_images)

        form_layout = QFormLayout()
        form_layout.addRow("Model:", self.model_selector)
        form_layout.addRow("Confidence Threshold:", self.confidence_slider)
        form_layout.addRow(self.load_model_button)
        form_layout.addRow(self.batch_load_button)

        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(["Time", "Objects", "Count"])

        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Class", "Count"])

        control_layout = QVBoxLayout()
        control_layout.addLayout(form_layout)
        control_layout.addWidget(self.table_widget)
        control_layout.addWidget(QLabel("Detection History"))
        control_layout.addWidget(self.history_table)

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.fps_label)
        video_layout.addWidget(self.image_label)
        video_layout.addWidget(self.canvas)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout, 2)
        main_layout.addLayout(control_layout, 1)

        self.setLayout(main_layout)

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)  # Capture from the first camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def setup_timers(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(500)  # 更新频率改为0.5秒

    def update_confidence_threshold(self, value):
        self.confidence_threshold = value / 100.0

    def load_model(self):
        self.model_name = self.model_selector.currentText()
        # Add logic to notify the backend to load the selected model
        print(f"Loaded model: {self.model_name}")

    def load_batch_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if files:
            self.batch_detect_objects(files)

    def batch_detect_objects(self, file_paths):
        for file_path in file_paths:
            try:
                image = cv2.imread(file_path)
                _, img_encoded = cv2.imencode('.jpg', image)
                response = requests.post(f"https://127.0.0.1:8000/detect/?model_name={self.model_name}",
                                         files={"file": img_encoded.tobytes()},
                                         verify=False)
                if response.status_code == 200:
                    detection_results = response.json().get('detections', [])
                    self.show_batch_detection_results(image, detection_results, file_path)
                else:
                    print(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    def show_batch_detection_results(self, image, detections, file_path):
        for detection in detections:
            box = detection['box']
            class_name = detection['class_name']
            confidence = detection['confidence']
            color = (0, 255, 0)  # 默认绿色
            if self.model_name == "SSD":
                color = (255, 0, 0)  # 蓝色
            elif self.model_name == "YOLOv5":
                color = (0, 255, 0)  # 绿色
            elif self.model_name == "YOLOv8":
                color = (0, 0, 255)  # 红色

            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(f"Detections in {file_path}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.draw_detections(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            self.update_statistics()

    def update_fps(self):
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_label.setText(f"FPS: {fps:.2f}")

    def draw_detections(self, frame):
        for detection in self.detection_results:
            box = detection['box']
            class_name = detection['class_name']
            confidence = detection['confidence']
            if confidence >= self.confidence_threshold:
                color = (0, 255, 0)  # 默认绿色
                if self.model_name == "SSD":
                    color = (255, 0, 0)  # 蓝色
                elif self.model_name == "YOLOv5":
                    color = (0, 255, 0)  # 绿色
                elif self.model_name == "YOLOv8":
                    color = (0, 0, 255)  # 红色

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def detect_objects(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                _, img_encoded = cv2.imencode('.jpg', frame)
                try:
                    response = requests.post(f"https://127.0.0.1:8000/detect/?model_name={self.model_name}",
                                             files={"file": img_encoded.tobytes()},
                                             verify=False)
                    if response.status_code == 200:
                        self.detection_results = response.json().get('detections', [])
                        self.update_history()
                    else:
                        print(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    print(f"Error sending request: {e}")

    def update_statistics(self):
        class_counts = {}
        for detection in self.detection_results:
            class_name = detection['class_name']
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        self.table_widget.setRowCount(len(class_counts))

        for row, (class_name, count) in enumerate(class_counts.items()):
            self.table_widget.setItem(row, 0, QTableWidgetItem(class_name))
            self.table_widget.setItem(row, 1, QTableWidgetItem(str(count)))

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(class_counts.keys(), class_counts.values())
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Object Detection Statistics')
        self.canvas.draw()

    def update_history(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        object_names = [d['class_name'] for d in self.detection_results]
        object_count = len(self.detection_results)

        self.detection_history.append((current_time, ", ".join(object_names), object_count))
        self.history_table.setRowCount(len(self.detection_history))

        for row, (time, objects, count) in enumerate(self.detection_history):
            self.history_table.setItem(row, 0, QTableWidgetItem(time))
            self.history_table.setItem(row, 1, QTableWidgetItem(objects))
            self.history_table.setItem(row, 2, QTableWidgetItem(str(count)))

    def closeEvent(self, event):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoStreamWidget()
    window.setWindowTitle("Object Detection")
    window.show()
    sys.exit(app.exec())
