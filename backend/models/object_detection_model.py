import cv2
import numpy as np

class ObjectDetectionModel:
    def __init__(self, model_name="SSD"):
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        if self.model_name == "SSD":
            self.net = cv2.dnn.readNetFromCaffe(
                "backend/models/MobileNetSSD_deploy.prototxt",
                "backend/models/MobileNetSSD_deploy.caffemodel"
            )
            self.class_names = {
                0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat",
                5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
                11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
                16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"
            }

        elif self.model_name == "YOLOv5":
            self.net = cv2.dnn.readNet("backend/models/yolov5s.onnx")
            self.class_names = {
                0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 4: "aeroplane",
                5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
                10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
                14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
                20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
                25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
                30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
                35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
                39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
                45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
                51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
                57: "sofa", 58: "pottedplant", 59: "bed", 60: "diningtable", 61: "toilet",
                62: "tvmonitor", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
                68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
                73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
                79: "toothbrush"
            }

        elif self.model_name == "YOLOv8":
            self.net = cv2.dnn.readNet("backend/models/yolov8s.onnx")
            self.class_names = {
                0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 4: "aeroplane",
                5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
                10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
                14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
                20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
                25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
                30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
                35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
                39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
                45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
                51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
                57: "sofa", 58: "pottedplant", 59: "bed", 60: "diningtable", 61: "toilet",
                62: "tvmonitor", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
                68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
                73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
                79: "toothbrush"
            }

    def detect_objects(self, frame):
        if self.model_name == "SSD":
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()
            h, w = frame.shape[:2]
            boxes = []
            confidences = []
            class_ids = []

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    boxes.append([startX, startY, endX, endY])
                    confidences.append(float(confidence))
                    class_ids.append(idx)

            return boxes, confidences, class_ids

        elif self.model_name == "YOLOv5" or self.model_name == "YOLOv8":
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
            self.net.setInput(blob)
            layer_outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
            boxes = []
            confidences = []
            class_ids = []
            h, w = frame.shape[:2]

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        box = detection[0:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            return boxes, confidences, class_ids
