from ultralytics import YOLO

# load a model
model = YOLO('runs/detect/train83/weights/last.pt')

# Train the model
model.val(data='yolo-Pill.yaml',workers=0,epochs=1000,batch=1)