from ultralytics import YOLO

# load a model
model = YOLO('yolov8-Concat_BiFPN.yaml')

# Train the model
model.train(data='yolo-Pill.yaml',workers=0,epochs=1000,batch=16)