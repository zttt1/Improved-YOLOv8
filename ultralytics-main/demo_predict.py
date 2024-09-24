from ultralytics import YOLO

yolo=YOLO("./yolov8n.pt",task="detect")

# 检测摄像头
# result=yolo(source="screen")
# 检测屏幕
# result=yolo (source="screen")
# save保存结果在run目录下,conf值越低框显示越多框取的越细化
# result=yolo (source="./ultralytics/assets/bus.jpg",save=True,conf=0.05)

# 检测文件图片
result=yolo (source="./ultralytics/assets/bus.jpg")
