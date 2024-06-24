from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from backend.models.object_detection_model import ObjectDetectionModel
import numpy as np
import cv2
import traceback

router = APIRouter()

# 初始化模型
ssd_model = ObjectDetectionModel(model_name='SSD')
yolov5_model = ObjectDetectionModel(model_name='YOLOv5')
yolov8_model = ObjectDetectionModel(model_name='YOLOv8')

def convert_to_standard_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_standard_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_standard_types(i) for i in obj]
    else:
        return obj

def clip_box_values(box, img_width, img_height):
    """Clip the box values to be within the image dimensions."""
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    return [x1, y1, x2, y2]

@router.post("/detect/")
async def detect_objects(file: UploadFile = File(...), model_name: str = Query("SSD")):
    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Unable to process image")

        img_height, img_width, _ = img.shape

        # 根据模型名称选择模型
        if model_name == 'YOLOv5':
            model = yolov5_model
        elif model_name == 'YOLOv8':
            model = yolov8_model
        else:
            model = ssd_model

        # 进行目标检测
        boxes, confidences, class_ids = model.detect_objects(img)
        detections = []
        for i in range(len(boxes)):
            box = clip_box_values(boxes[i], img_width, img_height)
            detections.append({
                "box": convert_to_standard_types(box),
                "confidence": float(confidences[i]),
                "class_id": int(class_ids[i]),
                "class_name": model.class_names[int(class_ids[i])]
            })

        return {"detections": detections}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
