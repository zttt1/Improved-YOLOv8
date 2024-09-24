from ultralytics import YOLO
from ultralytics.yolo.utils import YOLO

CFG = r'C:\Users\Administrator\Desktop\ultralytics-main\ultralytics\cfg\models\v8\yolov8-GAM.yaml'
SOURCE = r'C:\Users\Administrator\Desktop\ultralytics-main\ultralytics\assets\bus.jpg'


def test_model_forward():
    model = YOLO(CFG)
    model(SOURCE)
